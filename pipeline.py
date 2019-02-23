import argparse
# default
in0 = 23000
in1 = 46000
sip8 = 72000
delay = 7000
out = 11000

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

def simulate():
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
    print(total_time/pipelined_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in0', type = int, help = 'in0 cycles')
    parser.add_argument('--in1', type = int, help = 'in1 cycles')
    parser.add_argument('--sip8', type = int, help = 'sip8 cycles')
    parser.add_argument('--delay', type = int, help = 'delay cycles')
    parser.add_argument('--out', type = int, help = 'out cycles')
    args = parser.parse_args()
    if args.in0:
        in0 = args.in0
    if args.in1:
        in1 = args.in1
    if args.sip8:
        sip8 = args.sip8
    if args.delay:
        delay = args.delay
    if args.out:
        out = args.out

    simulate()
