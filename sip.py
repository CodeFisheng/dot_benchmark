from utils import *
import numpy as np

# compute ref sip power
sip_rate_old = (9123454 - 9091999 - 7000) / 2 / M_ref / K_ref / N_ref #cycles/fpcnt
sip_rate_new = sip_rate_old / 2


# sip_cycles_legacy
# def sip_cycles_fp16(m, k, n):
#     flop = 2 * m * k * n
#     sip_cycles = flop * sip_rate_new / sip_base_eff / 4
#     return sip_cycles
def sip_cycles(m, k, n, args, flag):
    if args.dtype == 0:
        return sip_cycles_fp16(m, k, n, flag)
    elif args.dtype == 1:
        return sip_cycles_fp32(m, k, n, flag)


def sip_cycles_fp32(m, k, n, flag):
    flop = 2 * m * k * n
    sip_cycles = flop * sip_rate_new / sip_base_eff
    return sip_cycles


def sip_cycles_fp16(m, k, n, flag):
    x_list, y_list = get_sipblock_search_list(m, n)
    z = np.ceil(k / 4/ 8)
    eff0 = 0
    eff1 = 0
    eff2 = 0
    xx, yy, zz = 0, 0, 0
    eff_final = 0
    for x in x_list:
        for y in y_list:
            sip_code_eff_ = sip_code_efficiency_fp16(m, k, n, x, y, z)
            sip_dma_eff_ = sip_dma_efficiency_fp16(m, k, n, x, y, z, flag)
            sip_leading_eff_ = sip_leading_efficiency_fp16(m, k, n, x, y, z, flag)
            #print("total efficiency = ", sip_code_eff_ * sip_dma_eff_ * sip_leading_eff_)
            eff_final_ = sip_code_eff_ * sip_dma_eff_ * sip_leading_eff_
            if eff_final_ > eff_final:
                xx, yy, zz = x, y, z
                eff0 = sip_code_eff_
                eff1 = sip_dma_eff_
                eff2 = sip_leading_eff_
                eff_final = eff_final_

    p00 = sip_base_power * sip_base_eff
    p0 = p00 * eff0
    p1 = p0 * eff1
    p2 = p1 * eff2
    flop = 2 * m * k * n
    if p2 == 0:
        print('p2=0')
        cycles_tmp = 10000000
    else:
        cycles_tmp = flop * 4 * 1.35 / p2/ 1000
    print('cycles = ', cycles_tmp)
    print('adjusted_sip_power = ', p2)
    print('efficiency = ', eff0 * eff1 * eff2)

    return [cycles_tmp, [xx, yy, zz, eff0, p0, eff1, p1, eff2, p2]]


def sip_code_efficiency_fp16(m, k, n, x, y, z):
    flop = x * y * z
    ret = flop / (flop + 8)
    # print('eff0 = ', ret)
    return ret


def sip_dma_efficiency_fp16(m, k, n, x, y, z, flag):
    if flag == 1:
        # print('2x + y')
        tmp = x * y / (128 * (x + y) / 109 + 128 * x / ns.reshape_bandwidth)
    else:
        # print('2x + 2y')
        tmp = x * y / (128 * (x + y) / 109 + 128 * (x + y) / ns.reshape_bandwidth)
    ret = min(1.0, tmp)
    # print('eff1 = ', ret)
    return ret


def sip_leading_efficiency_fp16(m, k, n, x, y, z, flag):
    if flag == 1:
        leading_cycles = (x + y)*z*128/109 + x * z * 128 / ns.reshape_bandwidth
    else:
        leading_cycles = (x + y)*z*128/109 + (x + y)*z*128/ns.reshape_bandwidth
    iter_cnt = np.ceil(k / 4 / z)
    iter_cycles = iter_cnt * (x * y * z + 8) / sip_dma_efficiency_fp16(m, k, n, x, y,
                                                                       z, flag)
    ret = iter_cycles / (iter_cycles + leading_cycles)
    print('eff2 = ', ret)
    return ret
