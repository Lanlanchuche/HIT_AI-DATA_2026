from bai1 import x_b, y, X
from bai2 import m
import numpy as np
from sklearn.linear_model import SGDRegressor

import matplotlib as mpl
import sklearn.preprocessing
import matplotlib.pyplot as plt

#Bai 3a

n_epochs = 50
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

np.random.seed(42)
theta = np.random.randn(2, 1)

t = 0

for epoch in range(n_epochs):
    for iteration in range(m):
        # YOUR CODE HERE
        random_index = np.random.randint(m)

        #cắt lát từng mẻ một
        xi = x_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi) # công thức sgd thì k chia gradient cho m
        # t là tổng số bước lặp đã thực hiện từ đầu đến giờ
        t += 1
        eta = learning_schedule(t)
        theta = theta - eta * gradients


print("SGD theta =", theta)

#Bai 3b


# YOUR CODE HERE
sgd_reg = SGDRegressor(max_iter=1000, tol = 1e-5, penalty= None, eta0=0.01, n_iter_no_change=100, random_state = 42)
sgd_reg.fit(X, y.ravel())

print("intercept:", sgd_reg.intercept_)
print("coef:", sgd_reg.coef_)

#Bai 3c

n_epochs = 50
batch_size = 20
t0, t1 = 200, 1000

def learning_schedule_mb(t):
    return t0 / (t + t1)

np.random.seed(42)
theta = np.random.randn(2, 1)
t = 0

for epoch in range(n_epochs):
    # YOUR CODE HERE
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = x_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    for i in range(0, m, batch_size):
        xi = X_b_shuffled[i: i + batch_size] #lay tu i den i + 20
        yi = y_shuffled[i: i + batch_size]

        t += 1
        gradients = 2/m * xi.T @ (xi @ theta - yi)
        eta = learning_schedule_mb(t)
        theta = theta - eta * gradients






print("Mini-batch GD theta =", theta)



