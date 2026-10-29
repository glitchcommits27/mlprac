import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("iris.csv")
X = df.drop(columns=["species"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

depths = [2, 3, 4, 5]
train_acc, test_acc = [], []

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)

    train_acc.append(tree.score(X_train, y_train))
    test_acc.append(tree.score(X_test, y_test))

    print(f"Depth {d}: train = {train_acc[-1]:.3f} "
          f"test = {test_acc[-1]:.3f}")

best_depth = depths[test_acc.index(max(test_acc))]
print("Best depth:", best_depth)

plt.plot(depths, train_acc, "o-", label="Train")
plt.plot(depths, test_acc, "s-", label="Test")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Depth vs Accuracy")
plt.legend()
plt.tight_layout()
plt.show()