import pandas as pd
from sklearn.model_selection import train_test_split
from pygam import LinearGAM, s, f
from sklearn.metrics import mean_squared_error

# Last inn data (Wage dataset, STK2100)
url = "https://www.uio.no/studier/emner/matnat/math/STK2100/v25/oblig/wage.csv"
data = pd.read_csv(url)

# Kategoriske variabler som dummies (dropp first)
cat_cols = ['maritl', 'race', 'education', 'jobclass', 'health', 'health_ins']
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

X = data.drop(columns='wage')
y = data['wage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Next two columns should correspond to 'year' and 'age'
X_train = X_train[X_train.columns]
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Build GAM with splines on first two features (assumed year, age)
terms = s(0) + s(1)
for i in range(2, X_train.shape[1]):
    terms += f(i)

gam = LinearGAM(terms).fit(X_train.values, y_train)
y_pred = gam.predict(X_test.values)
mse = mean_squared_error(y_test, y_pred)

print(f"Test MSE (GAM): {mse:.2f}")
print(gam.summary())
