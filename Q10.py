import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("used_cars.csv").drop_duplicates().dropna()
df = df[["year", "km_driven", "transmission", "owner", "selling_price"]]
df = pd.get_dummies(df, columns=["transmission", "owner"], drop_first=True, dtype=int)

X = df.drop(columns=["selling_price"])
y = df["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outlier = (y_train < lower) | (y_train > upper)

print(f"Q1 = {Q1:,.0f}, Q3 = {Q3:,.0f}")
print(f"Lower = {lower:,.0f}, Upper = {upper:,.0f}")
print("Outliers in training data:", outlier.sum(), "of", len(y_train))

for name, Xt, yt in [
    ("Original", X_train, y_train),
    ("Outliers removed", X_train[~outlier], y_train[~outlier])
]:
    model = LinearRegression().fit(Xt, yt)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name:17s} RMSE = {rmse:,.0f} "
          f"R2 = {r2_score(y_test, y_pred):.3f}")