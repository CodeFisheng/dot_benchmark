def pipeline(in0, in1, sip8, delay, cqm, out, in0c, in1c, outc):
    cycles = {'in0':in0, 'in1':in1, 'sip8':sip8, 'delay':delay, 'cqm':cqm, 'out':out}
    group = {'in0':in0c, 'in1':in1c, 'sip8':2, 'delay':3, 'cqm':4, 'out':outc}
    N = 100
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


def pipeline_static(in0, sip8, delay, cqm, out):
    cycles = {'in0':in0, 'sip8':sip8, 'delay':delay, 'cqm':cqm, 'out':out}
    group = {'in0':1, 'sip8':2, 'delay':3, 'cqm':4, 'out':5}
    N = 100
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    argparse.add_argument('--in0', type = float)
