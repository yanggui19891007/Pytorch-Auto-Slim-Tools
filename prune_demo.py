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
