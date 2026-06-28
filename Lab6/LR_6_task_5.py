import numpy as np
np.asfarray = np.asarray
import neurolab as nl


P = [1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1]

Ye = [0, 1, 1, 1, 0,
      1, 0, 0, 0, 0,
      1, 1, 1, 0, 0,
      1, 0, 0, 0, 0,
      0, 1, 1, 1, 0]

V = [1, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 0]

target = np.asfarray([P, Ye, V])
target[target == 0] = -1

chars = ['П', 'Є', 'В']


net = nl.net.newhop(target)


print("Test on train samples (must be True):")
output = net.sim(target)
for i in range(len(target)):
    print(f"Буква {chars[i]}: {(output[i] == target[i]).all()}")


print("\nTest with errors (1-2 pixels changed):")


test_p = np.array(P); test_p[24] = 1

test_ye = np.array(Ye); test_ye[12] = 1

test_v = np.array(V); test_v[4] = 1

tests = [test_p, test_ye, test_v]
tests = np.asfarray(tests)
tests[tests == 0] = -1

results = net.sim(tests)

for i in range(len(tests)):
    is_correct = (results[i] == target[i]).all()
    print(f"Пошкоджена {chars[i]}: {is_correct}, кроків: {len(net.layers[0].outs)}")