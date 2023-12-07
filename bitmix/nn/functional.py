from bitmix.utils import QuantLinearGrad, QuantConv2DGrad, QuantConv2DGradNoBias
def linear(x, weights, wbits=32, bias=None):
    return QuantLinearGrad.apply(x, weights, bias, wbits)

def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, wbits=32):
    if bias is None:
        return QuantConv2DGradNoBias.apply(x, weight, stride, padding, groups, dilation, wbits)
    return QuantConv2DGrad.apply(x, weight, bias, stride, padding, groups, dilation, wbits)