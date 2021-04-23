import torch_pruning as pruning
import torch
from torch_quanting import AutoQuant
from torchvision.models import resnet18
model = resnet18()
flops_raw, params_raw = pruning.get_model_complexity_info(
    model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)  
print('\n-[INFO] before pruning flops:  ' + flops_raw)
print('-[INFO] before pruning params:  ' + params_raw)
torch.save(model, 'a.pth')
config_list = [{
    'quant_types': ['weight'],
    'quant_bits': {
        'weight': 8,
    }, # 这里可以仅使用 `int`，因为所有 `quan_types` 使用了一样的位长，参考下方 `ReLu6` 配置。
    'op_types':['Conv2d', 'Linear']
}]
quantizer = AutoQuant(model, config_list)
model = quantizer.compress()
