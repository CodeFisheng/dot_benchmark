import numpy as np

# other constants
_param_download = 1500
_cqm_count = 3
_cqm_cycles = 500
_frequency = 1.2
_turbo_frequency = 1.35
_sip_delay_cycles = 7000
byte_size_fp32 = 4
byte_size_fp16 = 2
cluster_buffer_size_limit = 4*1024*1024 #bytes

def find_k(klen, K):
    klen = int(klen)
    for i in range(1, 2001):
        tmp = np.ceil(K/i)
        if tmp < 256:
            return klen
        if tmp < klen:
            return tmp

def param_download_cycles():
    return _param_download

def cqm_count():
    return _cqm_count

def cqm_cycles():
    return _cqm_cycles

def get_freq():
    return _frequency

def get_turbo_freq():
    return _turbo_frequency

def sip_delay_cycles():
    return _sip_delay_cycles

def if_static(M, N, K):
    if M * K * 2 * byte_size_fp16 + N * K * byte_size_fp16 + 2 * M * N * byte_size_fp16 <\
            cluster_buffer_size_limit:
        return True
    else:
        return False