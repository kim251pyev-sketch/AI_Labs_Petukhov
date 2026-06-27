import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

text = np.loadtxt('data_perceptron.txt')

data = text[:, :2]
labels = text[:, 2].reshape(-1, 1)

plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.title('Вхідні дані')
plt.xlabel('Розмірність 1')
plt.ylabel('Розмірність 2')

dim1_min, dim1_max = data[:, 0].min(), data[:, 0].max()
dim2_min, dim2_max = data[:, 1].min(), data[:, 1].max()

nn = nl.net.newp([[dim1_min, dim1_max], [dim2_min, dim2_max]], 1)

error_progress = nn.train(data, labels, epochs=100, show=20, lr=0.03)

plt.figure()
plt.plot(error_progress)
plt.title('Зміна помилки навчання')
plt.xlabel('Кількість епох')
plt.ylabel('Помилка навчання')
plt.grid()
plt.show()