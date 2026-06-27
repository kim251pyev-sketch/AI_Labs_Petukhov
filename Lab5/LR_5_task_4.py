import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

text = np.loadtxt('data_simple_nn.txt')
data = text[:, :2]
labels = text[:, 2:]

plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.title('Вхідні дані')

min_val = data.min(axis=0)
max_val = data.max(axis=0)

num_neurons = 2
net = nl.net.newp([[min_val[0], max_val[0]], [min_val[1], max_val[1]]], num_neurons)

error_progress = net.train(data, labels, epochs=100, show=20, lr=0.03)

plt.figure()
plt.plot(error_progress)
plt.title('Прогрес помилки навчання')
plt.xlabel('Кількість епох')
plt.ylabel('Помилка навчання')

test_data = np.array([[0.4, 0.2], [0.3, 0.7], [0.8, 0.4]])
predicted_labels = net.sim(test_data)

print("\nTest results:")
for i, point in enumerate(test_data):
    print(f"{point} --> {predicted_labels[i]}")

plt.show()