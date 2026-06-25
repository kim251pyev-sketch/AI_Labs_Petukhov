import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

m = 100
np.random.seed(42)
X = 6 * np.random.rand(m, 1) - 3
y = np.sin(X).flatten() + np.random.uniform(-0.5, 0.5, m)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_pred_poly = poly_reg.predict(X_new_poly)

print(f"Лінійна модель: y = {lin_reg.intercept_:.2f} + {lin_reg.coef_[0]:.2f}*x")
print(f"Поліноміальна модель: y = {poly_reg.intercept_:.2f} + {poly_reg.coef_[0]:.2f}*x + {poly_reg.coef_[1]:.2f}*x^2")

plt.scatter(X, y, color='blue', alpha=0.5, label='Випадкові дані')
plt.plot(X, y_pred_lin, color='green', label='Лінійна регресія')
plt.plot(X_new, y_pred_poly, color='red', label='Поліноміальна регресія (deg=2)')
plt.legend()
plt.title('Порівняння лінійної та поліноміальної регресії')
plt.show()