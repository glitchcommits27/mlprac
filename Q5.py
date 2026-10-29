
# Q5: Effect of training data size on sales prediction

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("Advertising.csv")

# Step 2: Select features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Step 3: Create one fixed test set
X_pool, X_test, y_pool, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train using different data sizes
for frac in [0.5, 0.7, 0.8]:
    n = int(frac * len(df))

    model = LinearRegression().fit(
        X_pool.iloc[:n], y_pool.iloc[:n]
    )
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(
        f"{int(frac*100)}% data ({n} rows): "
        f"RMSE = {rmse:.4f}, R2 = {r2:.4f}"
    )
