# Forex Price Prediction (EUR/USD)

This script applies **machine learning** to predict Forex prices (EUR/USD) using historical Yahoo Finance data.

## Features
- Download Forex data via `yfinance`
- Feature engineering: returns + moving averages
- Train/test split (time-series friendly)
- Random Forest regression model
- Evaluation using Mean Squared Error (MSE)
- Visualization of predicted vs. actual prices

## Requirements
```bash
pip install pandas numpy matplotlib scikit-learn yfinance
