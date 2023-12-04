import torch
import torch.nn as nn
from bitmix.utils import quantize_tensor, get_quant_params, get_quant_act_params

class QuantTanH(nn.Module):
    def __init__(self, in_bits=8, weight_bits=8,
                 out_bits=8, quantize_input=True,
                 quantize_output=True, warmup_steps=100):
        super(QuantTanH, self).__init__()
        self.alpha = 0.99

        self.quantize_input = True
        self.quantize_output = False
        self.weight_bits = weight_bits
        self.in_bits = in_bits
        self.out_bits = out_bits

        self.main_module = module
        self.quantized = False
        self.step = 0
        self.warmup_steps = warmup_steps

        self.input_rep_data = []
        self.output_rep_data = []

    def forward(self, x):
        if self.steps >= self.warmup_steps:
            self.quantized=True
        if not self.quantized:
            return torch.Tanh(x)
        self.steps = self.steps + 1

        s = 1/(2**(self.out_bits-1))
        x = torch.round(x/s)*s
        return x