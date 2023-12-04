import torch
import torch.nn as nn
from bitmix.utils import quantize_tensor, get_quant_params, get_quant_act_params

class QuantizationWrapper(nn.Module):
    def __init__(self, module, in_bits=8, weight_bits=8,
                 out_bits=8, quantize_input=True,
                 quantize_output=True, warmup_steps=100, alpha=0.999):
        super(QuantizationWrapper, self).__init__()
        

        self.alpha = 0.99

        self.quantize_input = True
        self.quantize_output = False
        self.weight_bits = weight_bits
        self.in_bits = in_bits
        self.out_bits = out_bits

        self.quantize_input=quantize_input
        self.quantize_output=quantize_output

        self.main_module = module
        self.quantized = False
        self.step = 0
        self.warmup_steps = warmup_steps

        self.input_rep_data = []
        self.output_rep_data = []
    
    def quantize(self):
        self.quantized = True
        _device = self.main_module.weight.device
        
        self.min_inp = torch.min(torch.tensor([torch.min(x) for x in self.input_rep_data])).to(_device)
        self.max_inp = torch.max(torch.tensor([torch.max(x) for x in self.input_rep_data])).to(_device)
        self.min_out = torch.min(torch.tensor([torch.min(x) for x in self.output_rep_data])).to(_device)
        self.max_out = torch.max(torch.tensor([torch.max(x) for x in self.output_rep_data])).to(_device)

    @property
    def weight(self):
        return self.main_module.weight

    def forward(self, x):
        if self.step>=self.warmup_steps and not self.quantized:
            self.quantize()

        self.step+=1

        if not self.quantized:
            self.input_rep_data.append(torch.unsqueeze(x.cpu(), dim=0))
            x = self.main_module(x)
            self.output_rep_data.append(torch.unsqueeze(x.cpu(), dim=0))
            return x

        if self.quantize_input:
            with torch.no_grad():
                curr_max = torch.max(x)
                curr_min = torch.min(x)

                self.min_inp = self.min_inp*self.alpha + curr_min*(1-self.alpha)
                self.max_inp = self.max_inp*self.alpha + curr_max*(1-self.alpha)

                x = torch.clamp(x, min=self.min_inp, max=self.max_inp)
                scale = (self.min_inp - self.max_inp)/(2**self.in_bits-1)
                x = torch.round(x/scale)*scale

        with torch.no_grad():
            s,z = get_quant_params(self.main_module.weight.data, n_bits=self.weight_bits,
                                   per_axis=isinstance(self.main_module, nn.Conv2d))
            weights = self.main_module.weight.data
            if len(self.weight.shape)>2:
                q_weights = torch.round(weights/s.view(weights.shape[0],1,1,1))*s.view(weights.shape[0],1,1,1)
            else:
                q_weights = torch.round(weights/s)*s
            fp_weights = self.main_module.weight.data
        
        setattr(self.main_module.weight, "data", q_weights)
        x = self.main_module(x)
        setattr(self.main_module.weight, "data", fp_weights)

        if self.quantize_output:
            with torch.no_grad():
                curr_max = torch.max(x)
                curr_min = torch.min(x)

                self.min_out = self.min_out*self.alpha + curr_min*(1-self.alpha)
                self.max_out = self.max_out*self.alpha + curr_max*(1-self.alpha)

                x = torch.clamp(x, min=self.min_out, max=self.max_out)
                scale = (self.min_out - self.max_out)/(2**self.out_bits-1)
                x = torch.round(x/scale)*scale
            
        return x