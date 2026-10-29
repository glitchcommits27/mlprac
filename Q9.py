import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("used_cars.csv").drop_duplicates().dropna()

X = pd.get_dummies(
    df[["year", "km_driven", "transmission", "owner"]],
    columns=["transmission", "owner"],
    drop_first=True,
    dtype=int
)
y = df["selling_price"]

print("Encoded columns:", list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for name, cols in [
    ("Numerical only", ["year", "km_driven"]),
    ("Numerical + encoded", list(X.columns))
]:
    model = LinearRegression().fit(X_train[cols], y_train)
    y_pred = model.predict(X_test[cols])
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name:20s} RMSE = {rmse:,.0f} "
          f"R2 = {r2_score(y_test, y_pred):.3f}")