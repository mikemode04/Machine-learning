import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Last ned valutadata (EUR/USD som eksempel)
data = yf.download("EURUSD=X", start="2020-01-01", end="2024-12-31")
data = data.dropna()

# Lag features: moving averages og returns
data["Return"] = data["Close"].pct_change()
data["MA5"] = data["Close"].rolling(5).mean()
data["MA20"] = data["Close"].rolling(20).mean()
data = data.dropna()

X = data[["Return", "MA5", "MA20"]]
y = data["Close"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Modell: Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Random Forest MSE (Forex): {mse:.4f}")

# Plot
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test, label="Actual EUR/USD")
plt.plot(y_test.index, y_pred, label="Predicted EUR/USD")
plt.legend()
plt.title("EUR/USD Prediction with Random Forest")
plt.show()
