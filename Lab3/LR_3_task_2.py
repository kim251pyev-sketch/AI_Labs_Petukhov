import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

input_file = 'data_regr_2.txt'

data = np.loadtxt(input_file, delimiter=',')
data = data[data[:, 0].argsort()] # Сортування за першим стовпцем (X)
X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

y_test_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test, color='blue', label='Тестові дані')
plt.plot(X_test, y_test_pred, color='red', linewidth=3, label='Прогноз')
plt.title('Лінійна регресія: data_regr_2.txt')
plt.legend()
plt.show()

print("Linear regressor performance (data_regr_2):")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))