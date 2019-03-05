import argparse
import pandas as pd
import time

from cdma import *
from sip import *
from pipeline import *
from utils import *

# TODO: all in one method, follow leo's ppt
# for scenario 1, in1 + in0_block + output can be put in cbuffer


def run(M, K, N, args):
    klen = (cluster_buffer_size_limit - 2 * M * N * byte_size_fp16)  /\
           (2 * byte_size_fp16 * M + 2 * byte_size_fp16 * N)
    k_block = find_k(klen, K)
    print('normal: block size = ', M, ' * ', k_block, '(', K, ')', ' * ', N)
    if klen < 256:
        print('normal: k_block < 500, error')
        return [0, 0, 0, 0, 0, 0]
    fp_cnt_block = 2 * M * N * k_block
    in0 = cdma_cycles_fp16(M, k_block)
    in1 = cdma_cycles_fp16(k_block, N)
    out = cdma_cycles_fp16(M, N)
    sip8 = sip_cycles_fp16(M, k_block, N)
    cqm = cqm_count() * cqm_cycles()
    delay = sip_delay_cycles()
    cycles = in0 + in1 + sip8 + delay + cqm + out

    total_cap = (get_turbo_freq() / get_freq()) * byte_size_fp32 * fp_cnt_block * \
                get_freq() / cycles / 1000

    # print breakdown cycles
    if args.show and args.show == 1:
        print("in0 = ", in0 , " cycles")
        print("in1 = ", in1, " cycles")
        print("sip8 = ", sip8, " cycles")
        print("out = ", out, " cycles")
        print("total cycles = ", cycles, " cycles")

    # find best pipeline solution
    pipeline_rate1 = pipeline(in0, in1, sip8, delay, cqm, out, 0, 0, 1)
    pipeline_rate2 = pipeline(in0, in1, sip8, delay, cqm, out, 0, 1, 0)
    pipeline_rate3 = pipeline(in0, in1, sip8, delay, cqm, out, 1, 0, 0)
    if (pipeline_rate3 >= pipeline_rate1)&(pipeline_rate3 >= pipeline_rate2):
            c1 = 1
            c2 = 0
            c3 = 0
    elif (pipeline_rate2 >= pipeline_rate1)&(pipeline_rate2 >= pipeline_rate3):
            c1 = 0
            c2 = 1
            c3 = 0
    elif (pipeline_rate1 >= pipeline_rate2)&(pipeline_rate1 >= pipeline_rate3):
            c1 = 0
            c2 = 0
            c3 = 1
    pipeline_rate = max(pipeline_rate1, max(pipeline_rate2, pipeline_rate3))

    # for conservation, pipeline real efficiency
    pipeline_rate = 1.0 + 1.0 * (pipeline_rate - 1.0)

    # output results
    pipelined_total_cap = pipeline_rate * total_cap
    print("normal: pipe = ", pipeline_rate, "; and power = ", pipelined_total_cap)
    return [c1, c2, c3, pipeline_rate, pipelined_total_cap, k_block]

def run_stable(M, K, N, args):
    global time_space
    # print('M = ', M, '; N = ', N)
    # print("k_block_length = ", klen)
    # where this 2* cycles comes from, 128? but I saw cycles launch8 counted

    M_block = M
    N_block = N
    k_block = K
    print('stable: block size = ', M_block, ' * ', k_block, ' * ', N_block)
    if M_block * k_block * 2 + k_block * N_block + 2 * M_block * N_block > \
        cluster_buffer_size_limit / byte_size_fp16:
        print('stable: block size > 4MB, error')
        return [0, 0, 0, 0, 0, 0]
    fp_cnt_block = 2 * M_block * k_block * N_block
    in0 = cdma_cycles_fp16(M_block, k_block)
    in1 = cdma_cycles_fp16(k_block, N_block)
    out = cdma_cycles_fp16(M_block, N_block)
    sip8 = sip_cycles_fp16(M_block, k_block, N_block)
    cqm = cqm_count() * cqm_cycles()
    delay = sip_delay_cycles()
    cycles = in0 + sip8 + delay + cqm + out

    # total_cap = (1.35/1.2) * 4 * fp_cnt_block * freq / cycles / 1000
    total_cap = (get_turbo_freq() / get_freq()) * byte_size_fp32 * fp_cnt_block * \
                get_freq() / cycles / 1000

    # print breakdown cycles
    if args.show and args.show == 1:
        print("in0 = ", in0 , " cycles")
        print("in1 = ", in1, " cycles")
        print("sip8 = ", sip8, " cycles")
        print("out = ", out, " cycles")
        print("total cycles = ", cycles, " cycles")

    # find best pipeline solution
    pipeline_rate1 = pipeline_stable(in0, sip8, delay, cqm, out)
    pipeline_rate2 = pipeline_stable(in0, sip8, delay, cqm, out)
    pipeline_rate3 = pipeline_stable(in0, sip8, delay, cqm, out)
    pipeline_rate = max(pipeline_rate1, max(pipeline_rate2, pipeline_rate3))

    # for conservation, pipeline real efficiency
    pipeline_rate = 1.0 + 1.0 * (pipeline_rate - 1.0)

    # output results
    pipelined_total_cap = pipeline_rate * total_cap
    print("stable: pipe = ", pipeline_rate, "; and power = ", pipelined_total_cap, '\n')
    return [0, 0, 1, pipeline_rate, pipelined_total_cap, k_block]


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
            if if_static(mm, nn, i[2]):
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
        tmp = [i[5], i[0], i[2], i[1], m_, k_, n_, cc1, cc2, cc3, 'fail', pipe, i[4], cap]
    else:
        if i[1] == 8:
            cap /= 2
        if global_f == 1:
            tmp = [i[5], i[0], i[2], i[1], m_, k_, n_, cc1, cc2, cc3, 'F1', pipe, i[4], cap]
            print('m = ', m_, 'k_', k_, 'n_', n_)
        else:
            tmp = [i[5], i[0], i[2], i[1], m_, k_, n_, cc1, cc2, cc3, 'F2', pipe, i[4], cap]
            print('m = ', m_, 'k_', k_, 'n_', n_)
    return tmp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type = int, help = 'display integer')
    parser.add_argument("--data", type = int)
    parser.add_argument("--m", type = int)
    parser.add_argument("--n", type = int)
    parser.add_argument("--k", type = int)
    args = parser.parse_args()
    # dataset = np.array(pd.read_csv('DeepBench_NV_V100_mini.csv'))
    dataset = np.array(pd.read_csv('./input/DeepBench_NV_V100.csv'))
    result = []
    time1 = time.time()
    if args.data == 1:
        tmp_size = [args.m, args.n, args.k, 0, 0, "test"]
        print('size = ', tmp_size[0], ' * ', tmp_size[2], ' * ', tmp_size[1])
        tmp = go(tmp_size, args)
        result.append(tmp)
    else:
        for i in dataset:
            tmp = []
            print('size = ', i[0], ' * ', i[2], ' * ', i[1])
            tmp = go(i, args)
            result.append(tmp)
            pd.DataFrame(result).to_csv('./res_norm/DeepBench_enflame_final.csv', \
                                        header = 1, index = 0)
    time2 = time.time()
    print('time = ', time2 - time1)
