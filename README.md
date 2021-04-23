# pytorch-Autoslim2.0

A pytorch toolkit for structured neural network pruning automatically

完全自动化的模型剪枝工具
## 1 Introduction 项目介绍

### ① Architecture 系统架构

**用户层**：人人都会用的剪枝工具，仅需二行代码即可完成全自动化剪枝

**中间层**：提供统一接口，让开发者可以自己封装SOTA剪枝算法，不断更新工具

**系统底层**：自动分析网络结构并构建剪枝关系



## 2 Support 支持度

### ① Supported Models 支持的模型

|模型类型|<center>支持</center>|<center>已测试</center> |
| --- | --- | --- |
| 分类模型 |√  |AlexNet,VGG，ResNet系列等  |
| 检测模型 |√  |CenterNet，YOLO系列等  |
| 分割模型 |√ | 正在测试 |

### ② Pruning Algorithm 剪枝算法

|函数名|<center>剪枝算法</center>|
| --- | --- |
| l1_norm_pruning |[Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)|
| l2_norm_pruning |[Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)|
| fpgm_pruning |[Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/abs/1811.00250)|


在原始剪枝算法上，做了部分调整。此外，后续会支持更多的剪枝算法。
## 3 Installation 安装

```bash
pip install -e ./Autoslim
```

## 4 Instructions 使用介绍

**model可以来源于torchvision，也可以是自己在Pytorch中构建的model**

### Automatic Pruning 自动化剪枝

```python
import torch_pruning as pruning
from torchvision.models import resnet18
import torch

# 模型建立
model = resnet18()
flops_raw, params_raw = pruning.get_model_complexity_info(
    model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)  
print('-[INFO] before pruning flops:  ' + flops_raw)
print('-[INFO] before pruning params:  ' + params_raw)
# 选择裁剪方式
mod = 'fpgm'

# 剪枝引擎建立
slim = pruning.Autoslim(model, inputs=torch.randn(
    1, 3, 224, 224), compression_ratio=0.5)

if mod == 'fpgm':
    config = {
        'layer_compression_ratio': None,
        'norm_rate': 1.0, 'prune_shortcut': 1,
        'dist_type': 'l1', 'pruning_func': 'fpgm'
    }
elif mod == 'l1':
    config = {
        'layer_compression_ratio': None,
        'norm_rate': 1.0, 'prune_shortcut': 1,
        'global_pruning': False, 'pruning_func': 'l1'
    }
slim.base_prunging(config)
flops_new, params_new = pruning.get_model_complexity_info(
    model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)  
print('\n-[INFO] after pruning flops:  ' + flops_new)
print('-[INFO] after pruning params:  ' + params_new)

```

## 5 Examples 使用案例

### ①Resnet-cifar10

#### Train 训练

```bash
python prune_resnet18_cifar10.py --mode train --round 0
```
#### Pruning 剪枝

```bash
python prune_resnet18_cifar10.py --mode prune --round 1 --total_epochs 60
```

#### Train 微调

```bash
python cifar100_prune.py --mode train --round 2 --total_epochs 10 --batch_size 512
```

## 6 致谢

感谢以下仓库：[https://github.com/TD-wzw/Autoslim](https://github.com/TD-wzw/Autoslim)