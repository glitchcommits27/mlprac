import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score

df = pd.read_csv("Social_Network_Ads.csv")
X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sc = StandardScaler()
model = SVC(kernel="rbf").fit(sc.fit_transform(X_train), y_train)
y_pred = model.predict(sc.transform(X_test))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"TN = {tn} FP = {fp} FN = {fn} TP = {tp}")

result = X_test.copy()
result["Actual"] = y_test
result["Predicted"] = y_pred

print("\nFalse positives (predicted buy, did not buy):")
print(result[(result["Actual"] == 0) & (result["Predicted"] == 1)])

print("\nFalse negatives (predicted no, but bought):")
print(result[(result["Actual"] == 1) & (result["Predicted"] == 0)])

print(f"Precision = {precision_score(y_test, y_pred):.3f}")
print(f"Recall = {recall_score(y_test, y_pred):.3f}")