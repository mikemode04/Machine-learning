# Gas Price Analysis

This project applies a **linear regression model** on natural gas prices (via ETF `UNG` as a proxy).

## Features
- Download commodity data with `yfinance`
- Create lagged features (Lag1, Lag2 returns)
- Fit linear regression for return prediction
- Plot actual vs. predicted returns

## Requirements
```bash
pip install pandas matplotlib scikit-learn yfinance
