import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("Stores.csv")

np.random.seed(42)
data['sales'] = np.random.randint(1000, 5000, size=len(data))

data_encoded = pd.get_dummies(data[['store_type', 'country']], drop_first=True)

X = data_encoded
y = data['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

future_features = X.sample(10, random_state=42) 
future_pred = model.predict(future_features)

plt.figure(figsize=(8,6))
plt.plot(range(1, 11), future_pred, marker='o', color='green')
plt.xlabel("Future Periods")
plt.ylabel("Forecasted Sales")
plt.title("Future Sales Forecast")
plt.show()