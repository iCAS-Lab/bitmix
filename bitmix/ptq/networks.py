import bitmix
import torch
import torch.nn as nn
import copy

import transformers

def get_calibration_model(model: nn.Module, q_config, copy_model=False):
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

                conv = bitmix.ptq.CoreModule(module, **group_quant_config)
                setattr(parent, module_name, conv)
    return model

def quantize_calibrated_model(model: nn.Module, smoothing_factor=-1):
    q_modules = []
    for name, _ in model.state_dict().items():
        module, module_name = bitmix.utils.get_module(model, name, depth=2)
        if isinstance(module, bitmix.ptq.CoreModule):
            module.quantize_module(smoothing_factor=smoothing_factor)
            q_modules.append(module)
    assert len(q_modules)>0, "No quantized modules found. Create using bitmix.ptq.get_calibration_model"
    
