
# Q4: Identifying unusual sales predictions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("Advertising.csv")

# Step 2: Select features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train the model and predict
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))
print("R2:", round(r2_score(y_test, y_pred), 3))

# Step 5: Calculate residuals
result = X_test.copy()
result["Actual"] = y_test
result["Predicted"] = y_pred
result["Residual"] = result["Actual"] - result["Predicted"]
result["AbsResidual"] = result["Residual"].abs()

# Step 6: Find the five largest errors
top5 = result.sort_values("AbsResidual", ascending=False).head(5)
print("\nFive largest prediction errors:")
print(top5.round(3).to_string())

# Step 7: Plot actual versus predicted Sales
plt.scatter(y_test, y_pred, label="Test observations")
plt.scatter(top5["Actual"], top5["Predicted"],
            color="orange", s=80, label="Top 5 errors")

lims = [y_test.min() - 1, y_test.max() + 1]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.tight_layout()
plt.show()
