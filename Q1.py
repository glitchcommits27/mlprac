# Q1. Best combination of advertising channels
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load, inspect, clean
df = pd.read_csv("Advertising.csv")
print(df.head())
print(df.describe().round(2))
print("Missing values:", df.isnull().sum().sum(), "| Duplicates:", df.duplicated().sum())
df = df.dropna().drop_duplicates()

# 2. Split ONCE so every model uses the same test rows
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. One Linear Regression per feature combination
combos = [["TV"], ["Radio"], ["Newspaper"],
["TV", "Radio"], ["TV", "Newspaper"], ["Radio", "Newspaper"],
["TV", "Radio", "Newspaper"]]
results = []
for cols in combos:
    model = LinearRegression().fit(X_train[cols], y_train)
    y_pred = model.predict(X_test[cols])
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append([" + ".join(cols), rmse, r2_score(y_test, y_pred)])

# 4. Compare (lowest RMSE = best)
results = pd.DataFrame(results, columns=["Features", "RMSE", "R2"]).sort_values("RMSE")
print("Comparing RMSE and R2 for different feature combinations:")
print(results.round(3).to_string(index=False))
plt.barh(results["Features"], results["RMSE"])
plt.xlabel("RMSE (lower is better)")
plt.title("Advertising Features vs Prediction Error")
plt.tight_layout()
plt.show()