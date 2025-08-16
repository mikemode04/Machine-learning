import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Last ned naturgass-data (Henry Hub spot proxy via ETF UNG)
data = yf.download("UNG", start="2018-01-01", end="2024-12-31")

# Feature: lagged returns
data["Return"] = data["Close"].pct_change()
data["Lag1"] = data["Return"].shift(1)
data["Lag2"] = data["Return"].shift(2)
data = data.dropna()

X = data[["Lag1", "Lag2"]]
y = data["Return"]

# Lineær regresjon
model = LinearRegression()
model.fit(X, y)

data["Predicted"] = model.predict(X)

# Plot
plt.figure(figsize=(12,6))
plt.plot(data.index, data["Return"], label="Actual Return", alpha=0.6)
plt.plot(data.index, data["Predicted"], label="Predicted Return", alpha=0.8)
plt.legend()
plt.title("Gas ETF (UNG) - Return Prediction using Linear Regression")
plt.show()

score = model.score(X, y)
print(f"R² score for return prediction: {score:.4f}")
