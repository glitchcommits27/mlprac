import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("Social_Network_Ads.csv")
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

xx, yy = np.meshgrid(
    np.linspace(-3, 3, 300),
    np.linspace(-3, 3, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]

plt.figure(figsize=(11, 4.5))

for i, kernel in enumerate(["linear", "rbf"]):
    model = SVC(kernel=kernel).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    Z = model.predict(grid).reshape(xx.shape)

    plt.subplot(1, 2, i + 1)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(
        X_test[:, 0], X_test[:, 1],
        c=y_test, cmap="coolwarm", edgecolors="k"
    )
    plt.title(f"{kernel} SVM (accuracy {acc:.2f})")
    plt.xlabel("Age (scaled)")
    plt.ylabel("Salary (scaled)")
    print(f"{kernel} SVM test accuracy = {acc:.3f}")

plt.tight_layout()
plt.show()