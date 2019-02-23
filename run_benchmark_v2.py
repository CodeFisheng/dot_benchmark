import argparse
import numpy as np
import pandas as pd

# for scenario 1, in1 + in0_block + output can be put in cbuffer

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
global freq
freq = 1.2 # cycles/second
cluster_buffer_size_limit = 2*1024*1024*byte_size #bytes cbuffer limit 2MB
sip_buffer_size_limit = 240*1024*byte_size #bytes sip_buffer limit 240KB

global paraminit
paraminit = 1500
time_space_min = 500 #cycles
time_space_max = 500 #cycles

# time 1 = CDMA(a) + CDMA(b) + launch8 + launch1 + CDMA(o)
# time 2-k = CDMA(a) + launch8 + launch1 + CMDA(o)
# k = 64
# averagely: time each loop = CDMA(a) + CDMA(b)/64 + launch8 + launch1 + CDMA(o) + 5CQM_GAPS
# launch sip, c2s slice + s2s reshape can be ignore with launch8 estimate

global cqm, cdma, show
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

def run(M, K, N, args):
    global time_space
    time_space = (time_space_min + time_space_max) / 2 #cycles
    cluster_buffer_size_limit = 2*1024*1024 #bytes cbuffer limit 2MB
    klen = (cluster_buffer_size_limit - M * N * 4)  / (4 * M + 4 * N)
    if klen < 32:
        return 0
    # print('M = ', M, '; N = ', N)
    # print("k_block_length = ", klen)
    # where this 2* cycles comes from, 128? but I saw cycles launch8 counted

    fp_cnt_block = 2 * M * klen * N
    in0 = calc_cdma_linear(M, klen)
    in1 = calc_cdma_linear(klen, N)
    out = calc_cdma_linear(M, N)
    sip8 = calc_sip_launch(fp_cnt_block)
    delay = (CQM_gap_cnt-2) * time_space
    if args.cqm:
        time_space /= args.cqm
    if args.cdma:
        in0 /= args.cdma
        in1 /= args.cdma
        out /= args.cdma
    # TODO, corrected, I forgot 1/64
    cycles = in0 + in1 + sip8 + delay + out

    if args.show and args.show == 1:
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

    pipeline_rate = pipeline(in0, in1, sip8, delay, out)
    # print("pipeline rate = ", pipeline_rate)
    pipelined_total_cap = pipeline_rate * total_cap
    # print("final calculation capacity after pipeline: ", pipelined_total_cap)
    return pipelined_total_cap

def pipeline(in0, in1, sip8, delay, out):
    cycles = {'in0':in0, 'in1':in1, 'sip8':sip8, 'delay':delay, 'out':out}
    group = {'in0':0, 'in1':1, 'sip8':2, 'delay':3, 'out':1}
    N = 10000
    stream1 = []
    stream2 = []
    for i in range(N):
        stream1.append('in0')
        stream1.append('in1')
        stream1.append('sip8')
        stream1.append('delay')
        stream1.append('out')
        stream2.append('in0')
        stream2.append('in1')
        stream2.append('sip8')
        stream2.append('delay')
        stream2.append('out')
    total_time = 2 * N * ( cycles['in0'] + cycles['in1'] + cycles['sip8'] + cycles['delay'] + cycles['out'] )
    #print(total_time)

    pipelined_time = 0
    last = True
    exe1 = []
    exe2 = []
    cycle1 = 0
    cycle2 = 0
    rnd = 0
    while True:
        if len(stream1) == 0:
            break
        if len(stream2) == 0:
            break
        # load node
        if rnd % 1:
            print('round = ', rnd)
            print(len(stream1))
            print(len(stream2))
        rnd += 1
        #if rnd > 10:
            #break
        if not exe1 and not exe2:
            exe1.append(stream1[0])
            exe2.append(stream2[0])
            stream1.pop(0)
            stream2.pop(0)
            cycle1 = cycles[exe1[0]]
            cycle2 = cycles[exe2[0]]
        elif exe1:
            exe2.append(stream2.pop(0))
            cycle2 = cycles[exe2[0]]
        elif exe2 and not exe1:
            exe1.append(stream1.pop(0))
            cycle1 = cycles[exe1[0]]
        if (group[exe1[0]] == group[exe2[0]])&(group[exe1[0]] != 3):
            if last == True:
                now = exe1.pop(0)
                last = False
                pipelined_time += cycle1
                cycle1 = 0
            elif last == False:
                now = exe2.pop(0)
                last = True
                pipelined_time += cycle2
                cycle2 = 0
        else:
            if cycle1 < cycle2:
                now = exe1.pop(0)
                last = False
                pipelined_time += cycle1
                cycle2 -= cycle1
                cycle1 = 0
            elif cycle1 > cycle2:
                now = exe2.pop(0)
                last = True
                pipelined_time += cycle2
                cycle1 -= cycle2
                cycle2 = 0
            elif cycle1 == cycle2:
                now = exe2.pop(0)
                exe1.pop(0)
                last = True
                pipelined_time += cycle2
                cycle1 = 0
                cycle2 = 0

    while stream1:
        exe = stream1.pop(0)
        pipelined_time += cycles[exe]
    while stream2:
        exe = stream2.pop(0)
        pipelined_time += cycles[exe]
    print('pipeline rate = ', total_time/pipelined_time, '\n')
    return total_time/pipelined_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type = int, help = 'display integer')
    parser.add_argument("--cqm", type = float, help = 'cqm improvement rate')
    parser.add_argument("--cdma", type = float, help = 'cdma improvement rate')
    args = parser.parse_args()
    # M_list = [128, 256, 384, 512, 640, 768, 896, 1024, 1024]
    # N_list = [128, 256, 384, 512, 640, 768, 896, 1024, 16]

    # for M in M_list:
    #     for N in N_list:
    #         run(M, N, args)
    dataset = np.array(pd.read_csv('DeepBench_NV_V100.csv'))
    result = []
    failres = []
    for i in dataset:
        cap = run(i[0], i[2], i[1], args)
        if cap == 0:
            tmp = [i[0], i[2], i[1], args]
            failres.append(tmp)
            continue
        tmp = [i[0], i[2], i[1], i[4], cap]
        result.append(tmp)
        print("M = ",i[0], " N = ", i[2], " K = ",i[1])
        print("     NV cap = ", i[4], ";   our cap = ", cap, " TFlops")
    pd.DataFrame(result).to_csv('DeepBench_enflame_passes_large.csv', header = 1, index = 0)
    pd.DataFrame(failres).to_csv('DeepBench_enflame_failures_large.csv', header = 1, index = 0)
