from param_config import *

# estimate cdma speed with ref data
cdma_linear_rate_ref = (M_ref * K_ref * byte_size_ref)/13564 #cycles/bytes
cdma_speed_ref = 1/cdma_linear_rate_ref


# compute cdma cycles for fp32 data
def cdma_cycles_fp32(x, y):
    return x * y * byte_size_fp32 / cdma_linear_rate_ref


# compute cdma cycles for fp16 data
def cdma_cycles_fp16(x, y):
    return x * y * byte_size_fp16 / cdma_linear_rate_ref
