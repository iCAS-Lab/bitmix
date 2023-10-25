import torch
import numpy as np

DEFUALT_INPUT_NBITS = 8
DEFUALT_OUTPUT_NBITS = 8
DEFUALT_WEIGHT_NBITS = 8

def quantize_tensor(tensor, scale, zero_point, dtype=torch.int32):
        tensor = torch.floor(tensor/scale)
        tensor += zero_point
        tensor = tensor.type(dtype)
        return tensor

def get_quant_params(x, symmetric=True, per_axis=False, n_bits=8):
    max_value = 2**n_bits
    if symmetric:
        if per_axis:
            scale = (2*torch.amax(torch.abs(x), dim=(1,2,3)))/(max_value-1)
            zero_point = torch.tensor(0)
        else:
            scale = (2*torch.max(torch.abs(x)))/(max_value-1)
            zero_point = torch.tensor(0)
    else:
        if per_axis:
            raise NotImplementedError
        scale = (torch.max(x) - torch.min(x))/(max_value-1)
        zero_point = -(torch.floor(torch.min(x)/scale)+(2**(n_bits-1)))
    return scale, zero_point

def get_quant_act_params(x, n_bits=8, device=None):
    max_value = 2**n_bits
    min_val = torch.min(torch.tensor([torch.min(a) for a in x]))
    max_val = torch.max(torch.tensor([torch.max(a) for a in x]))
    scale = (min_val - max_val)/(max_value-1)
    zero_point = -(torch.floor(min_val/scale)+(2**(n_bits-1)))
    if not device is None:
        scale, zero_point = scale.to(device), zero_point.to(device)
    return scale, zero_point

def get_quant_config(group, in_bits=DEFUALT_INPUT_NBITS, weight_bits=DEFUALT_WEIGHT_NBITS, out_bits=DEFUALT_OUTPUT_NBITS):
    return {"modules":group,
            "in_bits":in_bits,
            "weight_bits":weight_bits,
            "out_bits":out_bits,
            "quantize_output":False,
            "quantize_input":True}