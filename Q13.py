import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

df = pd.read_csv("Social_Network_Ads.csv")
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

for name, Xtr, Xte in [
    ("Unscaled", X_train, X_test),
    ("Standardised", X_train_s, X_test_s)
]:
    model = SVC(kernel="rbf").fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    print(f"\n{name}")
    print("Confusion matrix:", confusion_matrix(y_test, y_pred).tolist())
    print(f"Accuracy = {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1 = {f1_score(y_test, y_pred):.3f}")