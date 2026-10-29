import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)

df = pd.read_csv("Social_Network_Ads.csv")
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

models = [
    ("Logistic Regression", LogisticRegression()),
    ("RBF SVM", SVC(kernel="rbf"))
]

for name, model in models:
    y_pred = model.fit(X_train, y_train).predict(X_test)

    print(f"\n{name}")
    print("Confusion matrix:", confusion_matrix(y_test, y_pred).tolist())
    print(f"Accuracy = {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision = {precision_score(y_test, y_pred):.3f}")
    print(f"Recall = {recall_score(y_test, y_pred):.3f}")
    print(f"F1 = {f1_score(y_test, y_pred):.3f}")