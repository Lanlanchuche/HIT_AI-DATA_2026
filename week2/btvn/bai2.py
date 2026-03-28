from bai1 import x_b, y, X
import numpy as np
import matplotlib as mpl
import sklearn.preprocessing
import matplotlib.pyplot as plt




#Bai 2a
eta = 0.1  # learning rate
n_epochs = 1000
m = len(x_b)  # đảm bảo X_b đã được tạo ở Bài 1a

np.random.seed(42)
theta = np.random.randn(2, 1)

for epoch in range(n_epochs):

    gradients = 2/m * x_b.T @ (x_b @ theta - y)
    theta = theta - eta * gradients
    pass

print("theta =", theta)

#Bai 2b
X_new = np.array([[0], [2]])
X_new_b = sklearn.preprocessing.add_dummy_feature(X_new)

def plot_gradient_descent(theta, eta):
    m = len(x_b)
    plt.plot(X, y, "b.")
    n_epochs = 1000
    n_shown = 20

    for epoch in range(n_epochs):
        if epoch < n_shown:
            y_predict = X_new_b @ theta
            color = mpl.colors.rgb2hex(plt.cm.OrRd(epoch / n_shown + 0.15))
            plt.plot(X_new, y_predict, linestyle="solid", color=color)

        # YOUR CODE HERE — tính gradients và cập nhật theta
        gradients = 2/m * x_b.T @ (x_b @ theta - y)
        theta = theta - eta * gradients
    plt.xlabel("$x_1$")
    plt.axis([0, 2, 0, 15])
    plt.grid()
    plt.title(fr"$\eta = {eta}$")

np.random.seed(42)
theta = np.random.randn(2, 1)

plt.figure(figsize=(10, 4))
# YOUR CODE HERE — vẽ 3 subplot với eta = 0.02, 0.1, 0.5

plt.subplot(1, 3, 1)
plot_gradient_descent(theta, eta=0.02)

plt.subplot(1, 3, 2)
plot_gradient_descent(theta, eta=0.1)

plt.subplot(1, 3, 3)
plot_gradient_descent(theta, eta=0.5)

plt.tight_layout()

plt.show()