import sys

from sklearn.linear_model import LinearRegression

assert sys.version_info >= (3, 7)

import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

np.random.seed(42)

m = 100  # số mẫu
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

# Vẽ dữ liệu
plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
plt.title("Dữ liệu giả lập")
plt.show()

# Giai bai tap 1a
x_b = sklearn.preprocessing.add_dummy_feature(X)
theta_best = np.linalg.inv(x_b.T@x_b)@x_b.T@y
print("theta_best =", theta_best)


#Giai bai tap 1b

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print("Intercept:", lin_reg.intercept_)
print("Coef:", lin_reg.coef_)

#Giai bai tap 1c
X_new = np.array([[0], [2]])
y_predict = lin_reg.predict(X_new)

plt.figure(figsize=(6, 4))
plt.plot(X_new, y_predict, "r-", label = "Prediction")
plt.plot(X, y, "b")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
plt.legend()
plt.show()