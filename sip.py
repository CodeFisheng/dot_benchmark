from param_config import *

# compute ref sip power
sip_rate_old = (9123454 - 9091999 - 7000) / 2 / M_ref / K_ref / N_ref #cycles/fpcnt
sip_rate_new = sip_rate_old / 2


# sip_cycles_legacy
# def sip_cycles_fp16(m, k, n):
#     flop = 2 * m * k * n
#     sip_cycles = flop * sip_rate_new / sip_base_eff / 4
#     return sip_cycles
def sip_cycles(m, k, n, args):
    if args.dtype == 0:
        return sip_cycles_fp16(m, k, n)
    elif args.dtype == 1:
        return sip_cycles_fp32(m, k, n)


def sip_cycles_fp32(m, k, n):
    flop = 2 * m * k * n
    sip_cycles = flop * sip_rate_new / sip_base_eff
    return sip_cycles


def sip_cycles_fp16(m, k, n):
    flop = 2 * m * k * n
    sip_cycles = flop * sip_rate_new / sip_base_eff / 4
    adjusted_cycles = sip_cycles / sip_code_efficiency_fp16(m, k, n) /\
                      sip_dma_efficiency_fp16(m, k, n)
    return adjusted_cycles


def sip_code_efficiency_fp16(m, k, n):
    x = m / 16 / 8
    y = n / 16
    z = k / 4
    flop = x * y * z
    ret = flop / (flop + 8)
    return ret


def sip_dma_efficiency_fp16(m, k, n):
    x = m / 16 / 8
    y = n / 16
    z = k / 4
    tmp = 109 * x * y / 128 / (x + y)
    ret = min(1.0, tmp)
    return ret
