import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("Social_Network_Ads.csv")
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

for cols in [["Age"], ["EstimatedSalary"], ["Age", "EstimatedSalary"]]:
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train[cols])
    Xte = sc.transform(X_test[cols])

    model = SVC(kernel="rbf").fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    print(f"{' + '.join(cols):22s} "
          f"Accuracy = {accuracy_score(y_test, y_pred):.3f} "
          f"Precision = {precision_score(y_test, y_pred):.3f} "
          f"Recall = {recall_score(y_test, y_pred):.3f} "
          f"F1 = {f1_score(y_test, y_pred):.3f}")