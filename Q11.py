import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("used_cars.csv").drop_duplicates().dropna()

df["brand"] = df["name"].str.split().str[0]
df = df.drop(columns=["name"])

for col in ["brand", "fuel", "seller_type", "transmission", "owner"]:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=["selling_price"])
y = df["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)
importance = pd.Series(
    rf.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("Feature importances:")
print(importance.round(3))

top3 = list(importance.index[:3])
rf3 = RandomForestRegressor(random_state=42).fit(X_train[top3], y_train)

for name, model, cols in [
    ("All features", rf, list(X.columns)),
    ("Top 3 " + str(top3), rf3, top3)
]:
    y_pred = model.predict(X_test[cols])
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name}: RMSE = {rmse:,.0f} "
          f"R2 = {r2_score(y_test, y_pred):.3f}")