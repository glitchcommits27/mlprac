
# Q2: Linear versus Polynomial Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("Advertising.csv")

# Step 2: Select features and target
X = df[["TV", "Radio"]]
y = df["Sales"]

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Linear Regression
linear = LinearRegression().fit(X_train, y_train)

# Step 5: Train Polynomial Regression of degree 2
poly = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)
poly.fit(X_train, y_train)

# Step 6: Compare both models
for name, model in [
    ("Linear Regression", linear),
    ("Polynomial Regression", poly)
]:
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: RMSE = {rmse:.3f}, R2 = {r2:.3f}")
