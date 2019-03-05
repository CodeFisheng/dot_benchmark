from param_config import *

# estimate cdma speed with ref data
cdma_linear_rate_ref = (M_ref * K_ref * byte_size_ref)/13564 #cycles/bytes
cdma_speed_ref = 1/cdma_linear_rate_ref


def cdma_cycles(x, y, args):
    if args.dtype == 0:
        return cdma_cycles_fp16(x, y)
    elif args.dtype == 1:
        return cdma_cycles_fp32(x, y)


# compute cdma cycles for fp32 data
def cdma_cycles_fp32(x, y):
    cdma_cycle = x * y * byte_size_fp32 / cdma_linear_rate_ref / cdma_efficiency
    return cdma_cycle


# compute cdma cycles for fp16 data
def cdma_cycles_fp16(x, y):
    cdma_cycle = x * y * byte_size_fp16 / cdma_linear_rate_ref / cdma_efficiency
    return cdma_cycle
