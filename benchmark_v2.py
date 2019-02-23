import argparse
# size of matrix
M_bench = 1152
K_bench = 136
N_bench = 144
byte_size = 4

# TODO check capacity

cdma_linear_rate = (M_bench * K_bench * byte_size)/13564 #cycles/bytes
fp_cnt_ref = 2 * M_bench * K_bench * N_bench
CQM_gap_cnt = 5
sip_launch_rate = (9123454 - 9091999 - 7000) / fp_cnt_ref #cycles/fpcnt
new_sip_launch_rate = 0.5 * sip_launch_rate #cycles/fpcnt 

# variables
# constant parameters
freq = 1.2 # cycles/second
cluster_buffer_size_limit = 2*1024*1024*byte_size #bytes cbuffer limit 2MB
sip_buffer_size_limit = 240*1024*byte_size #bytes sip_buffer limit 240KB

launch1 = 2000 #cycles
launch1_times = 1
paraminit = 1500
time_space_min = 500 #cycles
time_space_max = 500 #cycles
time_space = (time_space_min + time_space_max) / 2 #cycles

# time 1 = CDMA(a) + CDMA(b) + launch8 + launch1 + CDMA(o)
# time 2-k = CDMA(a) + launch8 + launch1 + CMDA(o)
# k = 64
# averagely: time each loop = CDMA(a) + CDMA(b)/64 + launch8 + launch1 + CDMA(o) + 5CQM_GAPS
# launch sip, c2s slice + s2s reshape can be ignore with launch8 estimate

def calc_cdma_linear(row, col):
    # return cycles/bytes * bytes = cycles
    return row * col * byte_size / cdma_linear_rate

def calc_sip_launch(row, col, tail):
    # cycles
    return row * col * tail * 2 * new_sip_launch_rate + 7000

def calc_sip_launch(fp_cnt):
    # cycles
    return fp_cnt * new_sip_launch_rate + 7000
    # return 4500 + 7000
def run(M, N, args):
    cluster_buffer_size_limit = 2*1024*1024 #bytes cbuffer limit 2MB
    klen = (cluster_buffer_size_limit - M * N * 4)  / (4 * M + 4 * N)
    if klen < 256:
        return;
    print('M = ', M, '; N = ', N)
    print("k_block_length = ", klen)
    # where this 2* cycles comes from, 128? but I saw cycles launch8 counted
    fp_cnt_block = 2 * M * klen * N
    in0 = calc_cdma_linear(M, klen)
    in1 = calc_cdma_linear(klen, N)
    out = calc_cdma_linear(M, N)
    sip8 = calc_sip_launch(fp_cnt_block)

    # TODO, corrected, I forgot 1/64
    cycles = in0 + in1 + sip8 + (CQM_gap_cnt-2) * time_space + out 
    
    if args.show:
        if args.show == 1:
            print("calculation is averaged over 2048*2048 X 2048*64\n")
            print("in0[32, 2048] with 4 blocks, cdma linear copy from hbm to cluster, \n\
                take = ", calc_cdma_linear(M, klen), " cycles\n") 
            print("in1[2048, 64], cdma linear copy from hbm to cluster, \n\
                take = ", calc_cdma_linear(klen, N), " cycles\n") 
            print("launch8 for matrix 32*1024 X 1024*64 \ntakes: ", \
                calc_sip_launch(fp_cnt_block), " cycles\n")
            print("average delay in CQM vector align takes: ", time_space, \
                " cycles ; total cycles = ", \
                (CQM_gap_cnt-2) * time_space, "\n")
            print("out[32, 64], cdma linear copy from cluster to hbm, \ntake = ", \
                calc_cdma_linear(M, N), " cycles\n") 
            print("each cycles is: ", cycles)

    # TODO, need 4 bytes or not
    # 4 cores
    total_cap = (1.35/1.2) * 4 * fp_cnt_block * freq / cycles / 1000
    # print("[freq correction] \ncalculation capability = ", total_cap, " TFlops")

    pipeline_rate = 2 
    print("pipeline rate = ", pipeline_rate)
    pipelined_total_cap = pipeline_rate * total_cap
    print("final calculation capacity after pipeline: ", pipelined_total_cap)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type = int, help = 'display integer')
    parser.add_argument("--cqm", type = float, help = 'cqm improvement rate')
    parser.add_argument("--cdma", type = float, help = 'cdma improvement rate')
    args = parser.parse_args()
    M_list = [128, 256, 384, 512, 640, 768, 896, 1024, 1024]
    N_list = [128, 256, 384, 512, 640, 768, 896, 1024, 16]

    for M in M_list:
        for N in N_list:
            run(M, N, args)
