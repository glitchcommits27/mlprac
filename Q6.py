
# Q6: Comparing regression algorithms for used-car prices

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load and clean data
df = pd.read_csv("used_cars.csv")
print("Original rows:", len(df))
print("Missing values:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())

df = df.drop_duplicates().dropna()
df = df[["year", "km_driven", "transmission", "owner", "selling_price"]]

# Step 2: Encode categorical features
df["transmission"] = df["transmission"].map(
    {"Manual": 0, "Automatic": 1}
)
df = pd.get_dummies(df, columns=["owner"], drop_first=True, dtype=int)

# Step 3: Define features and target
X = df.drop(columns=["selling_price"])
y = df["selling_price"]

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train and evaluate models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append([name, rmse, r2])

# Step 6: Compare results and identify the best model
results = pd.DataFrame(results, columns=["Model", "RMSE", "R2"])
results = results.sort_values("RMSE")

print("\nModel Comparison:")
print(results.round(3).to_string(index=False))

best = results.iloc[0]
print("\nBest Model:", best["Model"])
print("Best RMSE:", round(best["RMSE"], 3))
print("Best R2:", round(best["R2"], 3))
