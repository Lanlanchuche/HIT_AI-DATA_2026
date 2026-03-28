from bai4 import iris
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


#Bai 5a

X_softmax = iris["data"][["petal length (cm)", "petal width (cm)"]].values
y_softmax = iris["target"].values

softmax_reg = LogisticRegression(solver="lbfgs", C=10, random_state=42)
softmax_reg.fit(X_softmax, y_softmax)

#Bai 5b

sample = np.array([[5,2]])
print("Predicted class:", softmax_reg.predict(sample))
print("Probabilities:", softmax_reg.predict_proba(sample))

#Bai 5c
custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])

x0_grid = np.linspace(0, 7, 500)      # Trục hoành (Petal length) từ 0 đến 7
x1_grid = np.linspace(0, 3.5, 200)    # Trục tung (Petal width) từ 0 đến 3.5
x0_mesh, x1_mesh = np.meshgrid(x0_grid, x1_grid)

X_grid = np.c_[x0_mesh.ravel(), x1_mesh.ravel()]

y_pred_grid = softmax_reg.predict(X_grid).reshape(x0_mesh.shape)

plt.figure(figsize=(10, 6))

plt.contourf(x0_mesh, x1_mesh, y_pred_grid, cmap=custom_cmap, alpha=0.8)

plt.plot(X_softmax[y_softmax==0, 0], X_softmax[y_softmax==0, 1], "yo", label="Iris setosa")
plt.plot(X_softmax[y_softmax==1, 0], X_softmax[y_softmax==1, 1], "bs", label="Iris versicolor")
plt.plot(X_softmax[y_softmax==2, 0], X_softmax[y_softmax==2, 1], "g^", label="Iris virginica")

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.legend(loc="center left")
plt.title("Softmax Regression Decision Boundary")
plt.axis([0, 7, 0, 3.5])
plt.grid(alpha=0.3)
plt.show()