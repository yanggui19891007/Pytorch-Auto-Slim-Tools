import torch


class LayerInfo:
    def __init__(self, name, module):
        self.module = module
        self.name = name
        self.type = type(module).__name__


def _setattr(model, name, module):
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)


def _check_weight(module):
    try:
        return isinstance(module.weight.data, torch.Tensor)
    except AttributeError:
        return False


class QuantizerModuleWrapper(torch.nn.Module):
    def __init__(self, module, module_name, module_type, config, quantizer):

        super().__init__()
        # origin layer information
        self.module = module
        self.name = module_name
        self.type = module_type
        # config and pruner
        self.config = config
        self.quantizer = quantizer

        if 'weight' in config['quant_types']:
            if not _check_weight(self.module):
                _logger.warning(
                    'Module %s does not have parameter "weight"', self.name)
            else:
                self.module.register_parameter(
                    'old_weight', torch.nn.Parameter(self.module.weight))
                delattr(self.module, 'weight')
                self.module.register_buffer('weight', self.module.old_weight)

    def forward(self, *inputs):
        if 'input' in self.config['quant_types']:
            inputs = self.quantizer.quant_grad.apply(
                inputs,
                QuantType.QUANT_INPUT,
                self)

        if 'weight' in self.config['quant_types'] and _check_weight(self.module):
            self.quantizer.quant_grad.apply(
                self.module.old_weight,
                QuantType.QUANT_WEIGHT,
                self, inputs[0])
            result = self.module(*inputs)
        else:
            result = self.module(*inputs)

        if 'output' in self.config['quant_types']:
            result = self.quantizer.quant_grad.apply(
                result,
                QuantType.QUANT_OUTPUT,
                self)
        return result


class Compressor(object):
    """
    Abstract base PyTorch compressor
    """

    def __init__(self, model, config_list):

        assert isinstance(model, torch.nn.Module)

        self.bound_model = model
        self.config_list = config_list

        self.modules_to_compress = None
        self.modules_wrapper = []
        self.is_wrapped = False

        self._fwd_hook_handles = {}
        self._fwd_hook_id = 0

        self.reset()

    def reset(self, checkpoint=None):
        """
        reset model state dict and model wrapper
        """
        self._unwrap_model()
        if checkpoint is not None:
            self.bound_model.load_state_dict(checkpoint)

        self.modules_to_compress = None
        self.modules_wrapper = []

        for layer, config in self._detect_modules_to_compress():
            wrapper = self._wrap_modules(layer, config)
            self.modules_wrapper.append(wrapper)

        self._wrap_model()

    def _detect_modules_to_compress(self):
        """
        detect all modules should be compressed, and save the result in `self.modules_to_compress`.
        The model will be instrumented and user should never edit it after calling this method.
        """
        if self.modules_to_compress is None:
            self.modules_to_compress = []
            for name, module in self.bound_model.named_modules():
                if module == self.bound_model:
                    continue
                layer = LayerInfo(name, module)
                config = self.select_config(layer)
                if config is not None:
                    self.modules_to_compress.append((layer, config))
        return self.modules_to_compress

    def _wrap_model(self):
        """
        wrap all modules that needed to be compressed

        """
        for wrapper in reversed(self.get_modules_wrapper()):
            _setattr(self.bound_model, wrapper.name, wrapper)
        self.is_wrapped = True

    def _unwrap_model(self):
        """
        unwrap all modules that needed to be compressed

        """
        for wrapper in self.get_modules_wrapper():
            _setattr(self.bound_model, wrapper.name, wrapper.module)
        self.is_wrapped = False

    def compress(self):
        """
        Compress the model with algorithm implemented by subclass.

        The model will be instrumented and user should never edit it after calling this method.
        `self.modules_to_compress` records all the to-be-compressed layers

        Returns
        -------
        torch.nn.Module
            model with specified modules compressed.
        """
        return self.bound_model

    def set_wrappers_attribute(self, name, value):
        """
        To register attributes used in wrapped module's forward method.
        If the type of the value is Torch.tensor, then this value is registered as a buffer in wrapper,
        which will be saved by model.state_dict. Otherwise, this value is just a regular variable in wrapper.

        Parameters
        ----------
        name : str
            name of the variable
        value: any
            value of the variable
        """
        for wrapper in self.get_modules_wrapper():
            if isinstance(value, torch.Tensor):
                wrapper.register_buffer(name, value.clone())
            else:
                setattr(wrapper, name, value)

    def get_modules_to_compress(self):
        """
        To obtain all the to-be-compressed modules.

        Returns
        -------
        list
            a list of the layers, each of which is a tuple (`layer`, `config`),
            `layer` is `LayerInfo`, `config` is a `dict`
        """
        return self.modules_to_compress

    def get_modules_wrapper(self):
        """
        To obtain all the wrapped modules.

        Returns
        -------
        list
            a list of the wrapped modules
        """
        return self.modules_wrapper

    def select_config(self, layer):
        """
        Find the configuration for `layer` by parsing `self.config_list`

        Parameters
        ----------
        layer : LayerInfo
            one layer

        Returns
        -------
        config or None
            the retrieved configuration for this layer, if None, this layer should
            not be compressed
        """
        ret = None
        for config in self.config_list:
            config = config.copy()
            # expand config if key `default` is in config['op_types']
            if 'op_types' in config and 'default' in config['op_types']:
                expanded_op_types = []
                for op_type in config['op_types']:
                    if op_type == 'default':
                        expanded_op_types.extend(
                            default_layers.weighted_modules)
                    else:
                        expanded_op_types.append(op_type)
                config['op_types'] = expanded_op_types

            # check if condition is satisified
            if 'op_types' in config and layer.type not in config['op_types']:
                continue
            if 'op_names' in config and layer.name not in config['op_names']:
                continue

            ret = config
        if ret is None or 'exclude' in ret:
            return None
        return ret

    def update_epoch(self, epoch):
        """
        If user want to update model every epoch, user can override this method.
        This method should be called at the beginning of each epoch

        Parameters
        ----------
        epoch : num
            the current epoch number
        """
        pass

    def _wrap_modules(self, layer, config):
        """
        This method is implemented in the subclasses, i.e., `Pruner` and `Quantizer`

        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the compression operation
        config : dict
            the configuration for compressing this layer
        """
        raise NotImplementedError()

    def add_activation_collector(self, collector):
        self._fwd_hook_id += 1
        self._fwd_hook_handles[self._fwd_hook_id] = []
        for wrapper in self.get_modules_wrapper():
            handle = wrapper.register_forward_hook(collector)
            self._fwd_hook_handles[self._fwd_hook_id].append(handle)
        return self._fwd_hook_id

    def remove_activation_collector(self, fwd_hook_id):
        if fwd_hook_id not in self._fwd_hook_handles:
            raise ValueError("%s is not a valid collector id" %
                             str(fwd_hook_id))
        for handle in self._fwd_hook_handles[fwd_hook_id]:
            handle.remove()
        del self._fwd_hook_handles[fwd_hook_id]


class Quantizer(Compressor):
    """
    Base quantizer for pytorch quantizer
    """

    def __init__(self, model, config_list):
        super().__init__(model, config_list)
        self.quant_grad = QuantGrad

    def quantize_weight(self, wrapper, **kwargs):
        """
        quantize should overload this method to quantize weight.
        This method is effectively hooked to :meth:`forward` of the model.
        Parameters
        ----------
        wrapper : QuantizerModuleWrapper
            the wrapper for origin module
        """
        raise NotImplementedError('Quantizer must overload quantize_weight()')

    def quantize_output(self, output, wrapper, **kwargs):
        """
        quantize should overload this method to quantize output.
        This method is effectively hooked to :meth:`forward` of the model.
        Parameters
        ----------
        output : Tensor
            output that needs to be quantized
        wrapper : QuantizerModuleWrapper
            the wrapper for origin module
        """
        raise NotImplementedError('Quantizer must overload quantize_output()')

    def quantize_input(self, *inputs, wrapper, **kwargs):
        """
        quantize should overload this method to quantize input.
        This method is effectively hooked to :meth:`forward` of the model.
        Parameters
        ----------
        inputs : Tensor
            inputs that needs to be quantized
        wrapper : QuantizerModuleWrapper
            the wrapper for origin module
        """
        raise NotImplementedError('Quantizer must overload quantize_input()')

    def _wrap_modules(self, layer, config):
        """
        Create a wrapper forward function to replace the original one.
        Parameters
        ----------
        layer : LayerInfo
            the layer to instrument the mask
        config : dict
            the configuration for quantization
        """
        assert 'quant_types' in config, 'must provide quant_types in config'
        assert isinstance(config['quant_types'],
                          list), 'quant_types must be list type'
        assert 'quant_bits' in config, 'must provide quant_bits in config'
        assert isinstance(config['quant_bits'], int) or isinstance(
            config['quant_bits'], dict), 'quant_bits must be dict type or int type'

        if isinstance(config['quant_bits'], dict):
            for quant_type in config['quant_types']:
                assert quant_type in config['quant_bits'], 'bits length for %s must be specified in quant_bits dict' % quant_type

        return QuantizerModuleWrapper(layer.module, layer.name, layer.type, config, self)


class QuantType:
    """
    Enum class for quantization type.
    """
    QUANT_INPUT = 0
    QUANT_WEIGHT = 1
    QUANT_OUTPUT = 2

QType_Dict = {
    0: "input",
    1: "weight",
    2: "output"
}

class QuantGrad(torch.autograd.Function):
    """
    Base class for overriding backward function of quantization operation.
    """
    @classmethod
    def _quantize(cls, x, scale, zero_point):
        """
        Reference function for quantizing x -- non-clamped.
        Parameters
        ----------
        x : Tensor
            tensor to be quantized
        scale : Tensor
            scale for quantizing x
        zero_point : Tensor
            zero_point for quantizing x
        Returns
        -------
        tensor
            quantized x without clamped
        """
        return ((x / scale) + zero_point).round()
    @classmethod
    def get_bits_length(cls, config, quant_type):
        """
        Get bit for quantize config
        Parameters
        ----------
        config : Dict
            the configuration for quantization
        quant_type : str
            quant type
        Returns
        -------
        int
            n-bits for quantization configuration
        """
        if isinstance(config["quant_bits"], int):
            return config["quant_bits"]
        else:
            return config["quant_bits"].get(quant_type)

    @staticmethod
    def quant_backward(tensor, grad_output, quant_type, scale, zero_point, qmin, qmax):
        """
        This method should be overrided by subclass to provide customized backward function,
        default implementation is Straight-Through Estimator
        Parameters
        ----------
        tensor : Tensor
            input of quantization operation
        grad_output : Tensor
            gradient of the output of quantization operation
        scale : Tensor
            the type of quantization, it can be `QuantType.QUANT_INPUT`, `QuantType.QUANT_WEIGHT`, `QuantType.QUANT_OUTPUT`,
            you can define different behavior for different types.
        zero_point : Tensor
            zero_point for quantizing tensor
        qmin : Tensor
            quant_min for quantizing tensor
        qmax : Tensor
            quant_max for quantizng tensor
        Returns
        -------
        tensor
            gradient of the input of quantization operation
        """
        return grad_output

    @staticmethod
    def forward(ctx, tensor, quant_type, wrapper, input_tensor=None, **kwargs):
        if quant_type == QuantType.QUANT_INPUT:
            output = wrapper.quantizer.quantize_input(tensor, wrapper, **kwargs)
        elif quant_type == QuantType.QUANT_WEIGHT:
            output = wrapper.quantizer.quantize_weight(wrapper, input_tensor=input_tensor, **kwargs)
        elif quant_type == QuantType.QUANT_OUTPUT:
            output = wrapper.quantizer.quantize_output(tensor, wrapper, **kwargs)
        else:
            raise ValueError("unrecognized QuantType.")


        bits = QuantGrad.get_bits_length(wrapper.config, QType_Dict[quant_type])
        qmin, qmax = torch.Tensor([0]).to(tensor.device), torch.Tensor([(1 << bits) - 1]).to(tensor.device)
        if hasattr(wrapper.module, 'scale') and hasattr(wrapper.module, 'zero_point'):
            scale = wrapper.module.scale
            zero_point = wrapper.module.zero_point
        else:
            scale, zero_point = None, None
        ctx.save_for_backward(tensor, torch.Tensor([quant_type]), scale, zero_point, qmin, qmax)
        return output

    @classmethod
    def backward(cls, ctx, grad_output):
        tensor, quant_type, scale, zero_point, qmin, qmax = ctx.saved_variables
        output = cls.quant_backward(tensor, grad_output, quant_type, scale, zero_point, qmin, qmax)
        return output, None, None, None
