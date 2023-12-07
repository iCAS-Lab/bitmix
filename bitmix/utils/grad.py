import torch
import torch.nn.functional as F
from bitmix.utils import get_quant_params
class QuantLinearGrad(torch.autograd.Function):
    """
    Forward and backwards pass for a quant FC layer
    """
    @staticmethod
    def forward(ctx, x, weights, bias, wbits):
        q_weights = weights
        if not wbits==32:
            s,_ = get_quant_params(weights, n_bits=wbits, per_axis=False)
            q_weights = torch.round(weights/s)*s
            
        ctx.save_for_backward(x, weights, q_weights)
        return F.linear(x, q_weights, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x,weights,ternary_w = ctx.saved_tensors
 
        d_input = F.linear(grad_output, torch.transpose(ternary_w,0,1))
        d_bias = grad_output.sum(dim=0)
        d_weight = torch.transpose(torch.matmul(torch.transpose(x, 0, 1), grad_output),0,1)

        return d_input, d_weight, d_bias, None



class QuantConv2DGrad(torch.autograd.Function):
    """
    Forward and backwards pass for a quantized conv
    """
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, groups, dilation, wbits):
        ctx.stride=stride
        ctx.padding=padding
        ctx.groups=groups
        ctx.dilation = dilation
        q_weights = weight
        if not wbits==32:
            s,_ = get_quant_params(q_weights, n_bits=wbits, per_axis=True)
            q_weights = torch.round(q_weights/s.view(q_weights.shape[0],1,1,1))*s.view(q_weights.shape[0],1,1,1)

        
        ctx.save_for_backward(input, weight, q_weights, bias)
        return F.conv2d(input, q_weights, bias, stride=stride, padding=padding, groups=groups, dilation=dilation)

    @staticmethod
    def backward(ctx, grad_output):
        input,weight,q_weights,bias = ctx.saved_tensors
        stride=ctx.stride
        padding=ctx.padding
        groups=ctx.groups
        dilation=ctx.dilation

        n_bias = bias.size()[0]
        d_bias = torch.reshape(grad_output, (n_bias,-1)).sum(dim=1)

        d_input = torch.nn.grad.conv2d_input(input.shape, q_weights, grad_output, stride=stride, padding=padding, groups=groups, dilation=dilation)
        d_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, groups=groups, dilation=dilation)
        return d_input, d_weight, d_bias, None, None, None, None, None

class QuantConv2DGradNoBias(torch.autograd.Function):
    """
    Forward and backwards pass for a ternary conv w/o bias
    """
    @staticmethod
    def forward(ctx, input, weight, stride, padding, groups, wbits):

        ctx.stride=stride
        ctx.padding=padding
        ctx.groups=groups
        ctx.dilation = dilation
        
        q_weights = weight
        if not wbits==32:
            s,_ = get_quant_params(weights, n_bits=wbits, per_axis=True)
            q_weights = torch.round(weights/s.view(q_weights.shape[0],1,1,1))*s.view(q_weights.shape[0],1,1,1)

        ctx.save_for_backward(input, weight, q_weights)
        return F.conv2d(input, q_weights, None, stride=stride, padding=padding, groups=groups, dilation=dilation)

    @staticmethod
    def backward(ctx, grad_output):
        input,weight,q_weights = ctx.saved_tensors
        stride=ctx.stride
        padding=ctx.padding
        groups=ctx.groups
        dilation=ctx.dilation

        d_input = torch.nn.grad.conv2d_input(input.shape, q_weights, grad_output, stride=stride, padding=padding, groups=groups, dilation=dilation)
        d_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding, groups=groups, dilation=dilation)
        return d_input, d_weight, None, None, None, None, None