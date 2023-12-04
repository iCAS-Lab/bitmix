import bitmix
import torch
import torch.nn as nn
import copy

import transformers
def quantization_aware_model(model: nn.Module, q_config, copy_model=False, warmup_steps=100):
    if copy_model:
        model = copy.deepcopy(model)
    for quant_group in q_config:
        modules = quant_group["modules"]
        group_quant_config = copy.deepcopy(quant_group)
        del group_quant_config["modules"]
        for name in modules:
            module, parent, module_name = bitmix.utils.get_module_parents(model, name)
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)\
                or isinstance(module, transformers.pytorch_utils.Conv1D):
                conv = bitmix.qat.QuantizationWrapper(module, **group_quant_config, warmup_steps=warmup_steps)
                setattr(parent, module_name, conv)
            elif isinstance(model, nn.Tanh):
                tanh = bitmix.qat.Tanh(**group_quant_config, warmup_steps=warmup_steps)
                setattr(parent, module_name, tanh)

    return model
