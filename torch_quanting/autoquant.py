import torch
import torch.nn as nn
import copy
from .quantizer import Quantizer

__all__ = ['AutoQuant']

class AutoQuant(Quantizer):
    """quantize weight to 8 bits
    """

    def __init__(self, model, config_list):
        super().__init__(model, config_list)
        self.layer_scale = {}

    def quantize_weight(self, wrapper, **kwargs):
        weight = copy.deepcopy(wrapper.module.old_weight.data)
        new_scale = weight.abs().max() / 127
        scale = max(self.layer_scale.get(wrapper.name, 0), new_scale)
        self.layer_scale[wrapper.name] = scale
        orig_type = weight.type()  # TODO: user layer
        weight = weight.div(scale).type(torch.int8).type(orig_type).mul(scale)
        wrapper.module.weight = weight
        return weight

