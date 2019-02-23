import argparse


# size of matrix
N = 128
M = 2048
K = 2048
#N = 700
#M = 1024
#K = 512
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
freq = 1.2*1024*1024 # cycles/second
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type = int, help = 'display integer')
    parser.add_argument("--cqm", type = float, help = 'cqm improvement rate')
    parser.add_argument("--cdma", type = float, help = 'cdma improvement rate')
    parser.add_argument("case", type = int, help = "select test cases\
        \n    1 2048 2048 32\
        \n    2 2048 2048 64\
        \n    3 2048 2048 128")
    args = parser.parse_args()
    

    if args.case == 1:
        M = 2048
        K = 2048
        N = 32
    elif args.case == 2:
        M = 2048
        K = 2048
        N = 64
    elif args.case == 3:
        M = 2048
        K = 2048
        N = 128
    # where this 2* cycles comes from, 128? but I saw cycles launch8 counted
    iters = M/32/4 # 2 blocks, 32*2048, 2048*64, 32*1024*32
    M_each = M/iters
    fp_cnt_total = 2 * M * K * N
    fp_cnt_each = 2 * M_each * K * N
        
    in0 = calc_cdma_linear(M_each, K)
    in1 = calc_cdma_linear(K, N)
    out = calc_cdma_linear(M_each, N)
    launch8 = calc_sip_launch(fp_cnt_each)

    if args.cqm:
        time_space /= args.cqm

    if args.cdma:
         in0 /= args.cdma
         in1 /= args.cdma
         out /= args.cdma

    # TODO, corrected, I forgot 1/64
    each_cycles = in0 + launch8 + (CQM_gap_cnt-2) * time_space + out 
    total_cycles = iters * each_cycles + paraminit + 2 * time_space + calc_cdma_linear(K, N)
    
    if args.show:
        if args.show == 1:
            print("calculation is averaged over 2048*2048 X 2048*64\n")
            print("parameter launch, take = ", launch1 , " cycles\n") 
            print("in0[32, 2048] with 4 blocks, cdma linear copy from hbm to cluster, \n\
                take = ", calc_cdma_linear(M_each, K), " cycles\n") 
            print("in1[2048, 64], cdma linear copy from hbm to cluster, \n\
                take = ", calc_cdma_linear(K, N)/iters, " cycles\n") 
            print("launch8 for matrix 32*1024 X 1024*64 \ntakes: ", \
                calc_sip_launch(fp_cnt_each), " cycles\n")
            print("average delay in CQM vector align takes: ", time_space, \
                " cycles ; total cycles = ", \
                (CQM_gap_cnt-2) * time_space + time_space * 2 / iters, "\n")
            print("out[32, 64], cdma linear copy from cluster to hbm, \ntake = ", \
                calc_cdma_linear(M_each, N), " cycles\n") 

            print("each cycles is: ", each_cycles)
            print("total cycles is: ", total_cycles, \
                ", for data size 32*2048 X 2048*64\n")

    #TODO, just one loop, is it OK 
    # print("total calculation FP is: ", fp_cnt_total, " float point operations\n")

    # TODO, need 4 bytes or not
    # 4 cores
    total_cap = (1.35/1.2) * 4 * fp_cnt_total * freq / total_cycles / 1024 / 1024 / 1024
    # print("[freq correction] \ncalculation capability = ", total_cap, " TFlops")

    sip_cycles = calc_sip_launch(fp_cnt_each) - 7000
    non_sip_cycles = each_cycles - sip_cycles
    pipeline_rate = (sip_cycles + non_sip_cycles) / max(sip_cycles, non_sip_cycles)
    # print("pipeline rate = ", pipeline_rate)
    pipelined_total_cap = pipeline_rate * total_cap
    print("final calculation capacity after pipeline: ", pipelined_total_cap)

