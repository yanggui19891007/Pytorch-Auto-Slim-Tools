import torch
import torch.nn as nn
import numpy as np
from itertools import chain
from .dependency import *
from . import prune
import math
from scipy.spatial import distance

__all__ = ['Autoslim']


class Autoslim(object):
    def __init__(self, model, inputs, compression_ratio):
        self.model = model  # torchvision.models模型
        self.inputs = inputs  # 输入大小，torch.randn(1,3,224,224)
        self.compression_ratio = compression_ratio  # 期望压缩率
        self.DG = DependencyGraph()
        # 构建节点依赖关系
        self.DG.build_dependency(model, example_inputs=inputs)
        self.model_modules = list(model.modules())
        self.pruning_func = {
            'l1': self._base_l1_pruning,
            'fpgm': self._base_fpgm_pruning
        }

    def index_of_layer(self):
        dicts = {}
        for i, m in enumerate(self.model_modules):
            if isinstance(m, nn.modules.conv._ConvNd):
                dicts[i] = m
        return dicts

    def base_prunging(self, config):
        if not config['pruning_func'] in self.pruning_func:
            raise KeyError(
                "-[ERROR] {} not supported.".format((config['pruning_func'])))

        ori_output = {}
        for i, m in enumerate(self.model_modules):
            if isinstance(m, nn.modules.conv._ConvNd):
                ori_output[i] = m.out_channels

        if config['layer_compression_ratio'] is None and config['prune_shortcut'] == 1:
            config['layer_compression_ratio'] = self._compute_auto_ratios()

        prune_indexes = self.pruning_func[config['pruning_func']](config)

        for i, m in enumerate(self.model_modules):
            if i in prune_indexes and m.out_channels == ori_output[i]:
                pruning_plan = self.DG.get_pruning_plan(
                    m, prune.prune_conv, idxs=prune_indexes[i])
                if pruning_plan and config['prune_shortcut'] == 1:
                    pruning_plan.exec()
                elif not pruning_plan.is_in_shortcut:
                    pruning_plan.exec()

    def _base_fpgm_pruning(self, config):
        prune_indexes = {}
        for i, m in enumerate(self.model_modules):
            # _ConvNd包含卷积和反卷积
            if isinstance(m, nn.modules.conv._ConvNd):
                weight_torch = m.weight.detach().cuda()
                if isinstance(m, nn.modules.conv._ConvTransposeMixin):
                    weight_vec = weight_torch.view(weight_torch.size()[1], -1)
                    out_channels = weight_torch.size()[1]
                else:
                    weight_vec = weight_torch.view(
                        weight_torch.size()[0], -1)  # 权重[512,64,3,3] -> [512, 64*3*3]
                    out_channels = weight_torch.size()[0]

                if config['layer_compression_ratio'] and i in config['layer_compression_ratio']:
                    similar_pruned_num = int(
                        out_channels * config['layer_compression_ratio'][i])
                    # 全自动化压缩时，不剪跳连层
                else:
                    similar_pruned_num = int(
                        out_channels * self.compression_ratio)

                filter_pruned_num = int(
                    out_channels * (1 - config['norm_rate']))

                if config['dist_type'] == "l2" or "cos":
                    norm = torch.norm(weight_vec, 2, 1)
                    norm_np = norm.cpu().numpy()
                elif config['dist_type'] == "l1":
                    norm = torch.norm(weight_vec, 1, 1)
                    norm_np = norm.cpu().numpy()

                filter_large_index = []
                filter_large_index = norm_np.argsort()[filter_pruned_num:]

                indices = torch.LongTensor(filter_large_index).cuda()
                # weight_vec_after_norm.size=15
                weight_vec_after_norm = torch.index_select(
                    weight_vec, 0, indices).cpu().numpy()

                # for euclidean distance
                if config['dist_type'] == "l2" or "l1":
                    similar_matrix = distance.cdist(
                        weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
                elif config['dist_type'] == "cos":  # for cos similarity
                    similar_matrix = 1 - \
                        distance.cdist(weight_vec_after_norm,
                                       weight_vec_after_norm, 'cosine')

                # 将任意一个点与其他点的距离算出来，最后将距离相加，一共得到15组数据
                similar_sum = np.sum(np.abs(similar_matrix), axis=0)

                # for distance similar: get the filter index with largest similarity == small distance
                similar_large_index = similar_sum.argsort()[
                    similar_pruned_num:]
                similar_small_index = similar_sum.argsort()[
                    :similar_pruned_num]
                prune_index = [filter_large_index[i]
                               for i in similar_small_index]
                prune_indexes[i] = prune_index
        return prune_indexes

    def _base_l1_pruning(self, config):
        prune_indexes = {}
        # 全局阈值剪枝法（最好别用，效果不佳）
        if config['global_pruning']:
            filter_record = []
            for i, m in enumerate(self.model_modules):
                if isinstance(m, nn.modules.conv._ConvNd):
                    weight = m.weight.detach().cpu().numpy()
                    if isinstance(m, nn.modules.conv._ConvTransposeMixin):
                        L1_norm = np.sum(np.abs(weight), axis=(
                            0, 2, 3))  # 注：反卷积维数1对应输出维度
                    else:
                        L1_norm = np.sum(np.abs(weight), axis=(1, 2, 3))
                    filter_record.append(L1_norm.tolist())  # 记录每层卷积的l1_norm参数

            filter_record = list(chain.from_iterable(filter_record))
            total = len(filter_record)
            filter_record.sort()  # 全局排序
            thre_index = int(total * self.compression_ratio)
            thre = filter_record[thre_index]  # 根据裁剪率确定阈值
            for i, m in enumerate(self.model_modules):
                if isinstance(m, nn.modules.conv._ConvNd):
                    weight = m.weight.detach().cpu().numpy()
                    # _ConvTransposeMixin只包含反卷积
                    if isinstance(m, nn.modules.conv._ConvTransposeMixin):
                        L1_norm = np.sum(np.abs(weight), axis=(
                            0, 2, 3))  # 注：反卷积维数1对应输出维度
                    else:
                        L1_norm = np.sum(np.abs(weight), axis=(1, 2, 3))
                    num_pruned = min(int(max_ratio*len(L1_norm)),
                                     len(L1_norm[L1_norm < thre]))  # 不能全部减去
                    # 删除低于阈值的卷积核
                    prune_index = np.argsort(L1_norm)[:num_pruned].tolist()
                    prune_indexes.append(prune_index)

        # 局部阈值加指定层
        else:
            if config['layer_compression_ratio'] is None and config['prune_shortcut'] == 1:
                # 需要剪跳连层，并且未指定每一层的裁剪率
                config['layer_compression_ratio'] = self._compute_auto_ratios()

            for i, m in enumerate(self.model_modules):
                # 逐层裁剪
                # _ConvNd包含卷积和反卷积
                if isinstance(m, nn.modules.conv._ConvNd):
                    weight = m.weight.detach().cpu().numpy()
                    # _ConvTransposeMixin只包含反卷积
                    if isinstance(m, nn.modules.conv._ConvTransposeMixin):
                        out_channels = weight.shape[1]
                        L1_norm = np.sum(np.abs(weight), axis=(0, 2, 3))
                    else:
                        out_channels = weight.shape[0]
                        L1_norm = np.sum(
                            np.abs(weight), axis=(1, 2, 3))  # 计算卷积核的L1范式

                    # 自定义压缩或全自动化压缩时剪跳连层
                    if config['layer_compression_ratio'] and i in config['layer_compression_ratio']:
                        num_pruned = int(
                            out_channels * config['layer_compression_ratio'][i])
                    # 全自动化压缩时，不剪跳连层
                    else:
                        num_pruned = int(out_channels * self.compression_ratio)

                    # remove filters with small L1-Norm
                    prune_index = np.argsort(L1_norm)[:num_pruned].tolist()
                    prune_indexes.append(prune_index)
        return prune_indexes

    def _compute_auto_ratios(self):
        layer_compression_ratio = {}
        mid_value = self.compression_ratio

        one_value = (1-mid_value)/4 if mid_value >= 0.43 else mid_value/4
        values = [mid_value-one_value*3, mid_value-one_value*2, mid_value-one_value,
                  mid_value, mid_value+one_value, mid_value+one_value*2, mid_value+one_value*3]
        layer_cnt = 0
        for i, m in enumerate(self.model_modules):
            if isinstance(m, nn.modules.conv._ConvNd):
                layer_compression_ratio[i] = 0
                layer_cnt += 1
        layers_of_class = layer_cnt/7
        conv_cnt = 0
        for i, m in enumerate(self.model_modules):
            if isinstance(m, nn.modules.conv._ConvNd):
                layer_compression_ratio[i] = values[math.floor(
                    conv_cnt/layers_of_class)]
                conv_cnt += 1
        return layer_compression_ratio


if __name__ == "__main__":
    from resnet_small import resnet_small
    model = resnet_small()
    slim = Autoslim(model, inputs=torch.randn(
        1, 3, 224, 224), compression_ratio=0.5)
    slim.l1_norm_pruning()
    print(model)
