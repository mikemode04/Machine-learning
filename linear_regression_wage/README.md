# Linear Regression on Wage Dataset

This project applies **Linear Regression** on the `wage.csv` dataset from UiO (STK2100).

## 📊 Dataset
- Source: [UiO Wage dataset](https://www.uio.no/studier/emner/matnat/math/STK2100/v25/oblig/wage.csv)  
- Variables: education, jobclass, health, health insurance, race, marital status, etc.  
- Target: `wage` (hourly wage).  

Categorical variables are encoded as dummy variables before fitting the regression model.

## ⚙️ Method
1. Clean and preprocess categorical variables.
2. Convert them to dummy variables.
3. Split dataset into training (80%) and test (20%).
4. Fit a linear regression model using `sklearn`.
5. Evaluate with Mean Squared Error (MSE).

## ▶️ How to run
Clone the repo and run:

```bash
python linear_regression_wage.py
