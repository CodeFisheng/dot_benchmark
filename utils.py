import numpy as np
from param_config import *


def find_k(M, K, N, args):
    klen_fp16 = (cluster_buffer_size_limit - 2 * M * N * byte_size_fp16) / \
           (2 * byte_size_fp16 * M + 2 * byte_size_fp16 * N)
    klen_fp16 = int(klen_fp16)
    klen_fp32 = (cluster_buffer_size_limit - 2 * M * N * byte_size_fp32) / \
                (2 * byte_size_fp32 * M + 2 * byte_size_fp32 * N)
    klen_fp32 = int(klen_fp32)
    if args.dtype == 0:
        for i in range(1, 2001):
            tmp = np.ceil(K/i)
            if tmp < 256:
                return klen_fp16
            if tmp < klen_fp16:
                return tmp
    elif args.dtype == 1:
        for i in range(1, 2001):
            tmp = np.ceil(K/i)
            if tmp < 500:
                return klen_fp32
            if tmp < klen_fp32:
                return tmp


def param_download_cycles():
    return param_download_


def cqm_count():
    return cqm_count_


def cqm_cycles():
    return cqm_cycles_


def get_freq():
    return frequency_


def get_turbo_freq():
    return turbo_frequency_


def sip_delay_cycles():
    return sip_delay_cycles_


def if_static(M, N, K, args):
    if args.dtype == 1:
        return if_static_fp32(M, N, K)
    elif args.dtype == 0:
        return if_static_fp16(M, N, K)


def if_static_fp16(M, N, K):
    if M * K * 2 * byte_size_fp16 + N * K * byte_size_fp16 + 2 * M * N * byte_size_fp16 <\
            cluster_buffer_size_limit:
        return True
    else:
        return False


def if_static_fp32(M, N, K):
    if M * K * 2 * byte_size_fp32 + N * K * byte_size_fp32 + 2 * M * N * byte_size_fp32 <\
            cluster_buffer_size_limit:
        return True
    else:
        return False


def if_dynamic(M, N, K, args):
    if args.dtype == 1:
        return if_dynamic_fp32(M, N, K)
    elif args.dtype == 0:
        return if_dynamic_fp16(M, N, K)


def if_dynamic_fp16(M, N, K):
    if M * K * 2 * byte_size_fp16 + N * K * 2 * byte_size_fp16 + 2 * M * N * \
            byte_size_fp16 < cluster_buffer_size_limit:
        return True
    else:
        return False


def if_dynamic_fp32(M, N, K):
    if M * K * 2 * byte_size_fp32 + N * K * 2 * byte_size_fp32 + 2 * M * N * \
            byte_size_fp32 < cluster_buffer_size_limit:
        return True
    else:
        return False


def get_search_list(m, n):
    tmp_m = min(m, 512)
    tmp_n = min(n, 512)
    m_unit = 128
    n_unit = 16
    m_times = int((tmp_m) / m_unit)
    n_times = int((tmp_n) / n_unit)
    print(tmp_m)
    print(tmp_n)
    m_list = [m_unit * (j+1) for j in range(m_times)]
    n_list = [n_unit * (j+1) for j in range(n_times)]
    if tmp_n == 8:
        n_list = [16]
    if tmp_m == 8:
        m_list = [16]
    if len(n_list) == 0:
        n_list.append(n)
    return m_list, n_list