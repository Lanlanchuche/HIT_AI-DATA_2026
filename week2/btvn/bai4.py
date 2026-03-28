from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np


# Iris dataset có sẵn trong sklearn, không cần download
iris = load_iris(as_frame=True)
print("Feature names:", list(iris.data.columns))
print("Target names:", list(iris.target_names))
print("Shape:", iris.data.shape)

#Bai 4a
X_iris =  iris["data"][["petal width (cm)"]].values
y_iris = (iris.target== 2).astype(int) #astype(int) de chuyen ve 0 va 1

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_iris, y_iris)

#Bai 4b
X_new_iris = np.linspace(0,3,1000).reshape(-1, 1)

y_proba = log_reg.predict_proba(X_new_iris)

plt.figure(figsize=(8, 4))
plt.plot(X_new_iris, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")
plt.plot(X_new_iris, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")
plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.legend()
plt.grid()
plt.show()
