import numpy as np
import neurolab as nl

target = [[-1, 1, -1, -1, 1, -1, -1, 1, -1],
          [1, 1, 1, 1, -1, 1, 1, -1, 1],
          [1, -1, 1, 1, 1, 1, 1, -1, 1],
          [1, 1, 1, 1, -1, -1, 1, -1, -1],
          [-1, -1, -1, -1, 1, -1, -1, -1, -1]]

input_data = [[-1, -1, 1, 1, 1, 1, 1, -1, 1],
              [-1, -1, 1, -1, 1, -1, -1, -1, -1],
              [-1, -1, -1, -1, 1, -1, -1, 1, -1]]

net = nl.net.newhem(target)

output = net.sim(target)
print("Test on train samples (must be [0, 1, 2, 3, 4])")
print(np.argmax(output, axis=1))

output_rec = net.sim([input_data[0]])
print("Outputs on recurrent cycle:")
print(np.array(net.layers[1].outs))

output_test = net.sim(input_data)
print("Outputs on test sample:")
print(output_test)