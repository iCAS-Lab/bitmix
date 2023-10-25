from bitmix.utils import quantize_tensor, get_quant_params, get_quant_act_params
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CoreModule(nn.Module):
    def __init__(self, module, in_bits=8, weight_bits=8,
                 out_bits=8, quantize_input=True,
                 quantize_output=True):
        super(CoreModule, self).__init__()
        self.input_rep_data = []
        self.output_rep_data = []

        self.quantize_input = quantize_input
        self.quantize_output = quantize_output

        self.main_module = module
        self.in_bits = in_bits
        self.weight_bits = weight_bits
        self.out_bits = out_bits
        self.quantized = False
        self.smooth_quant = False

    def quantize_module(self, smoothing_factor=-1):
        if self.quantized:
            return
        assert not len(self.input_rep_data)==0, "You must calibrate the model before quantizing"
        assert not len(self.output_rep_data)==0, "You must calibrate the model before quantizing"

        with torch.no_grad():
            _device = self.main_module.weight.device
            self.smooth_quant = not (smoothing_factor==-1)

            if self.smooth_quant:
                # SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
                # https://arxiv.org/pdf/2211.10438.pdf
                assert smoothing_factor>=0 and smoothing_factor<=1
                
                x_max = torch.amax(torch.stack([torch.amax(torch.abs(x.view(-1,x.shape[-1])),dim=0) for x in self.input_rep_data]), dim=0).to(_device)
                w_max = torch.amax(torch.abs(self.main_module.weight.data), dim=0).to(_device)

                x_max=x_max**(smoothing_factor)
                w_max=w_max**(1-smoothing_factor)

                s = x_max/w_max
                s[s==0] = 1e-4
                diag_s = torch.eye(s.shape[0]).to(_device)*s
                inv_diag_s = torch.inverse(diag_s)

                s_weights = torch.matmul(diag_s,self.main_module.weight.data.T).T
                setattr(self.main_module.weight, "data", s_weights.type(torch.float32))
                self.input_rep_data = [torch.matmul(x.to(_device),inv_diag_s).cpu() for x in self.input_rep_data]
                self.output_rep_data = [self.main_module(x.to(_device)).cpu() for x in self.input_rep_data]
                self.smooth_x = inv_diag_s


            self.min_inp = torch.min(torch.tensor([torch.min(x) for x in self.input_rep_data])).to(_device)
            self.max_inp = torch.max(torch.tensor([torch.max(x) for x in self.input_rep_data])).to(_device)
            self.min_out = torch.min(torch.tensor([torch.min(x) for x in self.output_rep_data])).to(_device)
            self.max_out = torch.max(torch.tensor([torch.max(x) for x in self.output_rep_data])).to(_device)

            self.input_quant_params = get_quant_act_params(self.input_rep_data, n_bits=self.in_bits, device=_device)
            self.output_quant_params = get_quant_act_params(self.output_rep_data, n_bits=self.out_bits, device=_device)
            self.weight_quant_params = get_quant_params(self.main_module.weight.data, n_bits=self.weight_bits,
                                                        per_axis=isinstance(self.main_module, nn.Conv2d))
                

            s1,z1 = self.input_quant_params # input scale and zero point
            s2,z2 = self.weight_quant_params # weight scale and zero point
            s3,z3 = self.output_quant_params # output scale and zero point

            sb,zb = s1*s2, torch.tensor([0]).to(_device) # bias scale and zero point

            weights = self.main_module.weight.data
            if len(self.weight.shape)>2:
                q_weights = quantize_tensor(weights, s2.view(weights.shape[0],1,1,1), z2)
            else:
                q_weights = quantize_tensor(weights, s2, z2)
                
            setattr(self.main_module.weight, "data", q_weights.type(torch.float32))

            if self.main_module.bias is not None:
                bias = self.main_module.bias.data
                q_bias = quantize_tensor(bias, torch.squeeze(sb), zb, dtype=torch.int32)
                setattr(self.main_module.bias, "data", q_bias.type(torch.float32))

            q2 = torch.sum(q_weights, [i for i in range(1,len(q_weights.shape))])
            self.q2z1 = q2*z1
            if len(self.weight.shape)>2:
                self.q2z1 = self.q2z1.view(1,self.q2z1.shape[0],1,1)

            self.m = (s1*s2)/s3
            if len(self.weight.shape)>2:
                self.m = self.m.view(1,self.m.shape[0],1,1)
            self.quantized = True
    
    @property
    def weight(self):
        return self.main_module.weight
    
    def forward(self, x):
        if not self.quantized:
            self.input_rep_data.append(torch.unsqueeze(x.cpu(), dim=0))
            x = self.main_module(x)
            self.output_rep_data.append(torch.unsqueeze(x.cpu(), dim=0))
            return x

        s1,z1 = self.input_quant_params
        s3,z3 = self.output_quant_params

        if self.quantize_input:
            if self.smooth_quant:
                x=torch.matmul(x,self.smooth_x)
            x = torch.clamp(x, min=self.min_inp, max=self.max_inp)
            x = quantize_tensor(x, s1, z1).type(torch.float32)

        
        x = self.main_module(x)
        x = (x - self.q2z1)*self.m + z3

        if not self.quantize_output:
            x = s3*(x - z3)
            x = torch.clamp(x, min=self.min_out, max=self.max_out)
        return x


class Conv2d(CoreModule):
    def __init__(self, module, in_bits=8, weight_bits=8,
                 out_bits=8, quantize_input=True,
                 quantize_output=True):
        super(Conv2d, self).__init__(module, in_bits=in_bits,
                                     weight_bits=weight_bits,
                                     out_bits=out_bits,
                                     quantize_input=quantize_input,
                                     quantize_output=quantize_output)