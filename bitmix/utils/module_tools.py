import bitmix

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_module(module, name, depth=1):
    if len(name.split("."))==depth:
        return module, name
    mod_name = name.split(".")[0]
    name = '.'.join(name.split(".")[1:])
    module = getattr(module, mod_name)
    return get_module(module, name, depth=depth)

def replace_modules(model, module_name_list, replace_fn):
    for name in module_name_list:
        module, parent, module_name = get_module_parents(model, name)
        replace_module = replace_fn(module, parent, module_name)
        setattr(parent, module_name, replace_module)

def get_module_parents(model, name):
    parent, module_name = bitmix.utils.get_module(model, name, depth=2)
    module_name = module_name.split(".")[0]
    module, _ = bitmix.utils.get_module(model, name)
    return module, parent, module_name

def get_paths_of_instance(model: torch.nn.Module, instance_type=nn.Conv2d):
    return_modules = []
    for name, _ in model.state_dict().items():
        parent, module_name = get_module(model, name, depth=2)
        module_name = module_name.split(".")[0]
        module, param_name = get_module(model, name)
        if "weight" in param_name and isinstance(module, instance_type):
            return_modules.append((name))
    return return_modules

def get_paths_of_activations(model, instance_type, path="model"):
    return_modules = []
    for module_name in dir(model):
        if isinstance(getattr(model, module_name), nn.Module):
            module = getattr(model, module_name)
            if module==model:
                continue
            elif isinstance(module, instance_type):
                return_modules.append(path)
            else:
                return_modules += get_paths_of_activations(module, instance_type, path=path+"."+module_name)
    return return_modules