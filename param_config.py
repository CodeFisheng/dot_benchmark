# size of matrix
M_ref = 1152
K_ref = 136
N_ref = 144
byte_size_ref = 4
byte_size_fp32 = 4
byte_size_fp16 = 2

# other constants
param_download_ = 1500
cqm_count_ = 3
cqm_cycles_ = 500
frequency_ = 1.35
turbo_frequency_ = 1.35
sip_delay_cycles_ = 7000
byte_size_fp32 = 4
byte_size_fp16 = 2
cluster_buffer_size_limit = 4*1024*1024 #bytes
sip_base_eff = 1.0
pipeline_efficiency = 0.7
cdma_efficiency = 1.0
mode_x = 4
mode_y = 8
sip_base_power = 86.4 * 1.35 / 1.35

class Namespace: pass

global ns
ns = Namespace()
def set_strategy2():
    ns.reshape_bandwidth = 64 # if m_unit == 64, reshape_bandwidth use 64, otherwise 8
    ns.cdma_reshape_eff = 46 / 32
    ns.m_unit = 64
    ns.n_unit = 32

def set_strategy1():
    ns.reshape_bandwidth = 8 # if m_unit == 64, reshape_bandwidth use 64, otherwise 8
    ns.cdma_reshape_eff = 1
    ns.m_unit = 128
    ns.n_unit = 16
