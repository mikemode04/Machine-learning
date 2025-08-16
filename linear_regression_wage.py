import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Last inn data
url = "https://www.uio.no/studier/emner/matnat/math/STK2100/v25/oblig/wage.csv"
wage = pd.read_csv(url)

# Rens kategoriske variabler
for col in ['education', 'jobclass', 'health', 'health_ins']:
    wage[col] = wage[col].apply(lambda x: x.split('. ', 1)[1] if '. ' in x else x)

# Konverter kategoriske variabler til dummy-variabler
categorical_cols = ['maritl', 'race', 'education', 'jobclass', 'health', 'health_ins']
wage_dummies = pd.get_dummies(wage, columns=categorical_cols, drop_first=True)

# Definer X og y
X = wage_dummies.drop(columns=['wage'])
y = wage_dummies['wage']

# Del datasettet i trenings- og testsett (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline (gjennomsnittsprediksjon)
baseline_pred = np.mean(y_train)
baseline_mse = mean_squared_error(y_test, [baseline_pred]*len(y_test))

# Lineær regresjon
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluer modellen
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2_train = model.score(X_train, y_train)

print(f"Forklaringsgrad (R^2) på treningssettet: {r2_train:.4f}")
print(f"Mean Squared Error på testsettet (modell): {mse:.2f}")
print(f"Mean Squared Error baseline (gj.snitt): {baseline_mse:.2f}")

# Plot topp 10 koeffisienter
coefficients = pd.Series(model.coef_, index=X.columns)
top10 = coefficients.abs().sort_values(ascending=False).head(10).index

plt.figure(figsize=(10, 6))
coefficients[top10].sort_values().plot(kind="barh")
plt.title("Topp 10 viktigste variabler (Lineær regresjon)")
plt.xlabel("Koeffisientverdi")
plt.tight_layout()
plt.show()
