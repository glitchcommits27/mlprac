import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("iris.csv")
X = df.drop(columns=["species"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

plt.figure(figsize=(16, 6))

for i, criterion in enumerate(["gini", "entropy"]):
    tree = DecisionTreeClassifier(
        criterion=criterion, max_depth=3, random_state=42
    ).fit(X_train, y_train)

    y_pred = tree.predict(X_test)

    print(f"\n{criterion.upper()} accuracy = {accuracy_score(y_test, y_pred):.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(export_text(tree, feature_names=list(X.columns)))

    plt.subplot(1, 2, i + 1)
    plot_tree(
        tree, feature_names=list(X.columns),
        class_names=list(tree.classes_), filled=True, fontsize=7
    )
    plt.title(criterion.upper())

plt.tight_layout()
plt.show()