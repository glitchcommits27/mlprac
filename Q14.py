import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("Social_Network_Ads.csv")
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

default = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
default_cv = cross_val_score(default, X_train, y_train, cv=5).mean()
default.fit(X_train, y_train)

params = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(
    make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    params, cv=5
)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

for name, model, cv in [
    ("Default", default, default_cv),
    ("Tuned", grid, grid.best_score_)
]:
    y_pred = model.predict(X_test)
    print(f"{name:8s} CV accuracy = {cv:.3f} "
          f"Test accuracy = {accuracy_score(y_test, y_pred):.3f} "
          f"F1 = {f1_score(y_test, y_pred):.3f}")