
# Q3: Evaluating the contribution of Newspaper advertising

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("Advertising.csv")

# Step 2: Define features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Step 3: Split the data once
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Compare models with and without Newspaper
n = len(y_test)

for cols in [
    ["TV", "Radio", "Newspaper"],
    ["TV", "Radio"]
]:
    model = LinearRegression().fit(X_train[cols], y_train)
    y_pred = model.predict(X_test[cols])

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    p = len(cols)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(
        f"{' + '.join(cols):24s} "
        f"RMSE = {rmse:.4f} "
        f"R2 = {r2:.4f} "
        f"Adjusted R2 = {adj_r2:.4f}"
    )
    print("Coefficients:", model.coef_.round(4))
