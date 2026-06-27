import numpy as np
np.Inf = np.inf
np.asfarray = np.asarray

import matplotlib.pyplot as plt
import neurolab as nl

min_val = -15
max_val = 15
num_points = 130
x = np.linspace(min_val, max_val, num_points)
y = 2 * np.square(x) + 5
y /= np.linalg.norm(y)

data = x.reshape(num_points, 1)
labels = y.reshape(num_points, 1)

plt.figure(figsize=(8, 4))
plt.scatter(data, labels, color='blue', s=10)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Вхідні дані')
plt.grid(True)

nn = nl.net.newff([[min_val, max_val]], [3, 1])
nn.trainf = nl.train.train_gd

error_progress = nn.train(data, labels, epochs=2000, show=200, goal=0.01)

plt.figure(figsize=(8, 4))
plt.plot(error_progress)
plt.xlabel('Кількість епох')
plt.ylabel('Помилка')
plt.title('Динаміка помилки під час навчання')
plt.grid(True)

x_dense = np.linspace(min_val, max_val, num_points * 2)
y_dense_pred = nn.sim(x_dense.reshape(x_dense.size, 1)).reshape(x_dense.size)
y_pred = nn.sim(data).reshape(num_points)

plt.figure(figsize=(8, 4))
plt.plot(x_dense, y_dense_pred, '-', label="Прогноз мережі (щільний)")
plt.plot(x, y, '.', label="Реальні дані")
plt.plot(x, y_pred, 'p', label="Навчальна вибірка")
plt.legend()
plt.title('Фактичні vs передбачені дані')
plt.grid(True)

plt.show()