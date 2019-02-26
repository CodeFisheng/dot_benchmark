import argparse
import numpy as np
import pandas as pd
import time

# TODO: all in one method, follow leo's ppt
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
cluster_buffer_size_limit = 4*1024*1024 #bytes cbuffer limit 2MB

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

def calc_sip_launch(fp_cnt):
    # cycles
    return fp_cnt * new_sip_launch_rate
    # return 4500 + 7000

def find_k(klen, K):
    klen = int(klen)
    for i in range(21):
        tmp = int(K/i)
        if tmp < 500:
            return klen
        if tmp < klen:
            return tmp

def run(M, K, N, args):
    global time_space
    time_space = (time_space_min + time_space_max) / 2 #cycles
    klen = (cluster_buffer_size_limit - 2 * M * N * byte_size)  / (2 * byte_size * M + 2 * byte_size * N)
    # print('M = ', M, '; N = ', N)
    # print("k_block_length = ", klen)
    # where this 2* cycles comes from, 128? but I saw cycles launch8 counted

    k_block = find_k(klen, K)
    k_times = int(K/k_block) + 1
    print('normal: block size = ', M, ' * ', k_block, '(', K, ')', ' * ', N)
    if klen < 500:
        print('normal: k_block < 500, error')
        return [0, 0, 0, 0, 0, 0]
    fp_cnt_block = 2 * M * k_block * N
    fp_cnt_base = 2 * M * K * N
    in0 = calc_cdma_linear(M, k_block)
    in1 = calc_cdma_linear(k_block, N)
    out = calc_cdma_linear(M, N)
    sip8 = calc_sip_launch(fp_cnt_block)
    cqm = (CQM_gap_cnt-2) * time_space
    delay = 7000
    if args.cqm:
        time_space /= args.cqm
    if args.cdma:
        in0 /= args.cdma
        in1 /= args.cdma
        out /= args.cdma
    cycles = in0 + in1 + sip8 + delay + cqm + out

    # if args.show and args.show == 1:
    #     print("calculation is averaged over 2048*2048 X 2048*64\n")
    #     print("in0[32, 2048] with 4 blocks, cdma linear copy from hbm to cluster, \n\
    #         take = ", calc_cdma_linear(M, klen), " cycles\n")
    #     print("in1[2048, 64], cdma linear copy from hbm to cluster, \n\
    #         take = ", calc_cdma_linear(klen, N), " cycles\n")
    #     print("launch8 for matrix 32*1024 X 1024*64 \ntakes: ", \
    #         calc_sip_launch(fp_cnt_block), " cycles\n")
    #     print("average delay in CQM vector align takes: ", time_space, \
    #         " cycles ; total cycles = ", \
    #         (CQM_gap_cnt-2) * time_space, "\n")
    #     print("out[32, 64], cdma linear copy from cluster to hbm, \ntake = ", \
    #         calc_cdma_linear(M, N), " cycles\n")
    #     print("each cycles is: ", cycles)

    # 4 cores

    ################ version I, consider tails
    total_cap = (1.35/1.2) * 4 * fp_cnt_block * freq * fp_cnt_base / fp_cnt_block / k_times / cycles / 1000
    ################ version II, not consider tails
    # total_cap = (1.35/1.2) * 4 * fp_cnt_block * freq / cycles / 1000

    # print("[freq correction] \ncalculation capability = ", total_cap, " TFlops")

    pipeline_rate1 = pipeline(in0, in1, sip8, delay, cqm, out, 0, 0, 1)
    pipeline_rate2 = pipeline(in0, in1, sip8, delay, cqm, out, 0, 1, 0)
    pipeline_rate3 = pipeline(in0, in1, sip8, delay, cqm, out, 1, 0, 0)
    if (pipeline_rate3 > pipeline_rate1)&(pipeline_rate3 > pipeline_rate2):
            c1 = 1
            c2 = 0
            c3 = 0
    elif (pipeline_rate2 > pipeline_rate1)&(pipeline_rate2 > pipeline_rate3):
            c1 = 0
            c2 = 1
            c3 = 0
    elif (pipeline_rate1 > pipeline_rate2)&(pipeline_rate1 > pipeline_rate3):
            c1 = 0
            c2 = 0
            c3 = 1

    pipeline_rate = max(pipeline_rate1, max(pipeline_rate2, pipeline_rate3))
    #print('pipeline rate = ', pipeline_rate, '\n')
    # print("pipeline rate = ", pipeline_rate)
    pipelined_total_cap = pipeline_rate * total_cap
    print("normal: pipe = ", pipeline_rate, "; and power = ", pipelined_total_cap)
    return [c1, c2, c3, pipeline_rate, pipelined_total_cap, k_block]

def run_stable(M, K, N, args):
    global time_space
    time_space = (time_space_min + time_space_max) / 2 #cycles
    # print('M = ', M, '; N = ', N)
    # print("k_block_length = ", klen)
    # where this 2* cycles comes from, 128? but I saw cycles launch8 counted

    M_block = M
    N_block = N
    k_block = K
    print('stable: block size = ', M_block, ' * ', k_block, ' * ', N_block)
    if M_block * k_block * 2 + k_block * N_block + 2 * M_block * N_block > cluster_buffer_size_limit:
        print('stable: block size > 4MB, error')
        return [0, 0, 0, 0, 0, 0]
    fp_cnt_block = 2 * M_block * k_block * N_block
    in0 = calc_cdma_linear(M_block, k_block)
    in1 = calc_cdma_linear(k_block, N_block)
    out = calc_cdma_linear(M_block, N_block)
    sip8 = calc_sip_launch(fp_cnt_block)
    cqm = (CQM_gap_cnt-2) * time_space
    delay = 7000
    if args.cqm:
        time_space /= args.cqm
    if args.cdma:
        in0 /= args.cdma
        in1 /= args.cdma
        out /= args.cdma
    cycles = in0 + sip8 + delay + cqm + out

    # if args.show and args.show == 1:
    #     print("calculation is averaged over 2048*2048 X 2048*64\n")
    #     print("in0[32, 2048] with 4 blocks, cdma linear copy from hbm to cluster, \n\
    #         take = ", calc_cdma_linear(M, k_block), " cycles\n")
    #     print("in1[2048, 64], cdma linear copy from hbm to cluster, \n\
    #         take = ", calc_cdma_linear(k_block, N), " cycles\n")
    #     print("launch8 for matrix 32*1024 X 1024*64 \ntakes: ", \
    #         calc_sip_launch(fp_cnt_block), " cycles\n")
    #     print("average delay in CQM vector align takes: ", time_space, \
    #         " cycles ; total cycles = ", \
    #         (CQM_gap_cnt-2) * time_space, "\n")
    #     print("out[32, 64], cdma linear copy from cluster to hbm, \ntake = ", \
    #         calc_cdma_linear(M, N), " cycles\n")
    #     print("each cycles is: ", cycles)

    # 4 cores
    total_cap = (1.35/1.2) * 4 * fp_cnt_block * freq / cycles / 1000
    # print("[freq correction] \ncalculation capability = ", total_cap, " TFlops")

    pipeline_rate1 = pipeline_stable(in0, sip8, delay, cqm, out)
    pipeline_rate2 = pipeline_stable(in0, sip8, delay, cqm, out)
    pipeline_rate3 = pipeline_stable(in0, sip8, delay, cqm, out)
    pipeline_rate = max(pipeline_rate1, max(pipeline_rate2, pipeline_rate3))
    #print('pipeline rate = ', pipeline_rate, '\n')
    # print("pipeline rate = ", pipeline_rate)
    pipelined_total_cap = pipeline_rate * total_cap
    print("stable: pipe = ", pipeline_rate, "; and power = ", pipelined_total_cap)
    return [0, 0, 1, pipeline_rate, pipelined_total_cap, k_block]

def pipeline_stable(in0, sip8, delay, cqm, out):
    cycles = {'in0':in0, 'sip8':sip8, 'delay':delay, 'cqm':cqm, 'out':out}
    group = {'in0':1, 'sip8':2, 'delay':3, 'cqm':4, 'out':5}
    N = 10000
    stream1 = []
    stream2 = []
    for i in range(N):
        stream1.append('cqm')
        stream1.append('in0')
        stream1.append('cqm')
        stream1.append('sip8')
        stream1.append('delay')
        stream1.append('cqm')
        stream1.append('out')

        stream2.append('cqm')
        stream2.append('in0')
        stream2.append('cqm')
        stream2.append('sip8')
        stream2.append('delay')
        stream2.append('cqm')
        stream2.append('out')
    total_time = 2 * N * ( cycles['in0'] + cycles['sip8'] + cycles['delay'] + 3*cycles['cqm'] + cycles['out'] )
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
        if (group[exe1[0]] == group[exe2[0]])&(group[exe1[0]] != 3)&(group[exe1[0]] != 4):
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
    return total_time/pipelined_time

def pipeline(in0, in1, sip8, delay, cqm, out, in0c, in1c, outc):
    cycles = {'in0':in0, 'in1':in1, 'sip8':sip8, 'delay':delay, 'cqm':cqm, 'out':out}
    group = {'in0':in0c, 'in1':in1c, 'sip8':2, 'delay':3, 'cqm':4, 'out':outc}
    N = 10000
    stream1 = []
    stream2 = []
    for i in range(N):
        stream1.append('cqm')
        stream1.append('in0')
        stream1.append('cqm')
        stream1.append('in1')
        stream1.append('cqm')
        stream1.append('sip8')
        stream1.append('delay')
        stream1.append('cqm')
        stream1.append('out')

        stream2.append('cqm')
        stream2.append('in0')
        stream2.append('cqm')
        stream2.append('in1')
        stream2.append('cqm')
        stream2.append('sip8')
        stream2.append('delay')
        stream2.append('cqm')
        stream2.append('out')
    total_time = 2 * N * ( cycles['in0'] + cycles['in1'] + cycles['sip8'] + cycles['delay'] + 4*cycles['cqm'] + cycles['out'] )
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
        if (group[exe1[0]] == group[exe2[0]])&(group[exe1[0]] != 3)&(group[exe1[0]] != 4):
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
    return total_time/pipelined_time

def if_stable(M, N, K):
    if M * K * 2 * byte_size + N * K * byte_size + 2 * M * N * byte_size < cluster_buffer_size_limit:
        return True
    else:
        return False

def go(i, args):
    tmp_m = min(i[0], 512)
    tmp_n = min(i[1], 512)
    m_unit = 128
    n_unit = 16
    m_remain = tmp_m % m_unit
    m_times = int((tmp_m) / m_unit)
    n_remain = tmp_n % n_unit
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
        n_list.append(i[1])
    pipe = 0
    cap = 0
    m_ = 0
    n_ = 0
    k_ = 0
    f1orf2 = 1
    global_f = 0
    c1 = -1
    c2 = -1
    c3 = -1
    cc1 = -1
    cc2 = -1
    cc3 = -1
    kk = 0
    for mm in m_list:
        for nn in n_list:
            if if_stable(mm, nn, i[2]):
                [c1, c2, c3, pipe_tmp, cap_tmp, kk] = run_stable(mm, i[2], nn, args)
                f1orf2 = 1
            else:
                [c1, c2, c3, pipe_tmp, cap_tmp, kk] = run(mm, i[2], nn, args)
                f1orf2 = 2
            if cap_tmp == 0:
                break
            if cap_tmp > cap:
                #print("new cap = ", cap_tmp, "new pipe = ", pipe_tmp)
                global_f = f1orf2
                pipe = pipe_tmp
                cap = cap_tmp
                cc1 = c1
                cc2 = c2
                cc3 = c3
                m_ = mm
                n_ = nn
                k_ = kk
    print(" **** normal using F", f1orf2, ": final cap = ", cap, "final pipe = ", pipe)
    if cap == 0:
        tmp = [i[0], i[2], i[1], m_, k_, n_, cc1, cc2, cc3, 'fail', pipe, i[4], cap]
    else:
        if i[1] == 8:
            cap /= 2
        if global_f == 1:
            tmp = [i[0], i[2], i[1], m_, k_, n_, cc1, cc2, cc3, 'F1', pipe, i[4], cap]
        else:
            tmp = [i[0], i[2], i[1], m_, k_, n_, cc1, cc2, cc3, 'F2', pipe, i[4], cap]
    return tmp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type = int, help = 'display integer')
    parser.add_argument("--cqm", type = float, help = 'cqm improvement rate')
    parser.add_argument("--cdma", type = float, help = 'cdma improvement rate')
    args = parser.parse_args()
    dataset = np.array(pd.read_csv('./input/DeepBench_NV_V100_mini.csv'))
    #dataset = np.array(pd.read_csv('DeepBench_NV_V100_mini.csv'))
    result = []
    time1 = time.time()
    for i in dataset:
        tmp = []
        print('size = ', i[0], ' * ', i[2], ' * ', i[1])
        tmp = go(i, args)
        result.append(tmp)
    pd.DataFrame(result).to_csv('./res_tail/DeepBench_enflame_v3_mini.csv', header = 1, index = 0)
    time2 = time.time()
    print('time = ', time2 - time1)
    #pd.DataFrame(result).to_csv('DeepBench_enflame_v3_mini_tail.csv', header = 1, index = 0)

