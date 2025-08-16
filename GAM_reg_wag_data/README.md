# GAM Regression on Wage Dataset

This project applies a **Generalized Additive Model (GAM)** to the Wage dataset from STK2100.

## Dataset Source  
URL: [wage.csv (UiO STK2100)](https://www.uio.no/studier/emner/matnat/math/STK2100/v25/oblig/wage.csv) :contentReference[oaicite:1]{index=1}

## Features & Model  
- Categorical features encoded as dummy variables.  
- Applies spline terms to **year** and **age** (smooth effects), with factor terms for other variables.  
- Train/test split: 80/20.  
- Evaluation via Mean Squared Error (MSE).  
- Prints GAM summary for coefficient insights.

## Usage
```bash
pip install pandas pygam scikit-learn
python gam_regression_wage.py
