# FX Direction Classifier (STK-style Logistic Regression)

A compact, **statistics-first** ML example on **financial time series**:
we predict whether **tomorrow's FX return** (EUR/USD) will be positive or negative using
a **logistic regression** with **L2 regularization**, built in a proper
`scikit-learn` pipeline and evaluated with **time-series cross-validation**.


## Data
- Source: Yahoo Finance via `yfinance`
- Instrument: `EURUSD=X` (you can switch to `USDNOK=X`, `GBPUSD=X`, etc.)
- Frequency: daily close, `2015-01-01 → latest`

---

## Features (no leakage)
- `ret1`, `ret5`: 1- and 5-day returns  
- `mom`: moving-average ratio `(MA5/MA20 − 1)` (momentum proxy)  
- `vol20`: rolling std of `ret1` (risk proxy)  
- Day-of-week dummies (Mon is baseline)  
- **Target**: `y = 1[ next_day_return > 0 ]`

---

## Method
- Split data by **time**: last 20% is a holdout test set (no shuffling).
- On the training window: **GridSearchCV** with **TimeSeriesSplit** (expanding window) to tune `C` in logistic regression.
- Pipeline: `StandardScaler → LogisticRegression(L2)`.
- Report: **AUC**, **Brier**, **Accuracy**, **Confusion matrix**, and top coefficients.

---

## Run
```bash
pip install pandas numpy yfinance scikit-learn matplotlib
python fx_direction_logit.py
