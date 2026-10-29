import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("used_cars.csv").drop_duplicates().dropna()
df = df[["year", "km_driven", "transmission", "owner", "selling_price"]]
df = pd.get_dummies(df, columns=["transmission", "owner"], drop_first=True, dtype=int)

X = df.drop(columns=["selling_price"])
y = df["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

base = DecisionTreeRegressor(random_state=42)
base_cv = -cross_val_score(
    base, X_train, y_train, cv=5,
    scoring="neg_root_mean_squared_error"
).mean()
base.fit(X_train, y_train)

params = {
    "max_depth": [2, 3, 4, 5, 7, 10, None],
    "min_samples_split": [2, 10, 20, 50],
    "min_samples_leaf": [1, 5, 10, 20]
}

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    params, cv=5, scoring="neg_root_mean_squared_error"
)
grid.fit(X_train, y_train)
best = grid.best_estimator_

print("Best parameters:", grid.best_params_)
print(f"Baseline: CV RMSE = {base_cv:,.0f} "
      f"Test R2 = {r2_score(y_test, base.predict(X_test)):.3f}")
print(f"Tuned: CV RMSE = {-grid.best_score_:,.0f} "
      f"Test R2 = {r2_score(y_test, best.predict(X_test)):.3f}")