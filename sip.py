from param_config import *

# compute ref sip power
sip_rate_old = (9123454 - 9091999 - 7000) / 2 / M_ref / K_ref / N_ref #cycles/fpcnt
sip_rate_new = sip_rate_old / 2


def sip_cycles_fp32(m, k, n):
    flop = 2 * m * k * n
    sip_cycles = flop * sip_rate_new / sip_base_eff
    return sip_cycles


def sip_cycles_fp16(m, k, n):
    flop = 2 * m * k * n
    sip_cycles = flop * sip_rate_new / sip_base_eff / 4
    return sip_cycles