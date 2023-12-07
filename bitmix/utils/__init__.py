from bitmix.utils.quant import quantize_tensor, get_quant_params, get_quant_config, get_quant_act_params
from bitmix.utils.module_tools import get_module, get_paths_of_instance, get_module_parents, replace_modules
from bitmix.utils.grouping import group_by_regex, group_by_substrings
from bitmix.utils.grad import QuantLinearGrad, QuantConv2DGrad, QuantConv2DGradNoBias