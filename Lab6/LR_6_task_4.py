import numpy as np
np.asfarray = np.asarray

import neurolab as nl

target = [[1, 0, 0, 0, 1,
           1, 1, 0, 0, 1,
           1, 0, 1, 0, 1,
           1, 0, 0, 1, 1,
           1, 0, 0, 0, 1],
          [1, 1, 1, 1, 1,
           1, 0, 0, 0, 0,
           1, 1, 1, 1, 1,
           1, 0, 0, 0, 0,
           1, 1, 1, 1, 1],
          [1, 1, 1, 1, 0,
           1, 0, 0, 0, 1,
           1, 1, 1, 1, 0,
           1, 0, 0, 1, 0,
           1, 0, 0, 0, 1],
          [0, 1, 1, 1, 0,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1,
           0, 1, 1, 1, 0]]

chars = ['N', 'E', 'R', 'O']
target = np.asfarray(target)
target[target == 0] = -1

net = nl.net.newhop(target)

output = net.sim(target)
print("Test on train samples (must be True for all):")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())

print("\nTest on defaced N:")
test_n = np.asfarray([0, 0, 0, 0, 0,
                      1, 1, 0, 0, 1,
                      1, 1, 0, 0, 1,
                      1, 0, 1, 1, 1,
                      0, 0, 0, 1, 1])
test_n[test_n == 0] = -1
out = net.sim([test_n])
print(f"Result for N: {(out[0] == target[0]).all()}, Sim. steps: {len(net.layers[0].outs)}")

print("\nTest on defaced E:")
test_e = np.asfarray([1, 1, 1, 1, 1,
                      1, 0, 0, 0, 0,
                      1, 1, 0, 1, 1,
                      1, 0, 0, 0, 0,
                      1, 1, 1, 1, 1])
test_e[test_e == 0] = -1
out_e = net.sim([test_e])
print(f"Result for E: {(out_e[0] == target[1]).all()}, Sim. steps: {len(net.layers[0].outs)}")