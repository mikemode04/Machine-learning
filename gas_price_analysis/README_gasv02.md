# Gas ETF (UNG) - Return Prediction (Lagged Linear Regression)

Dette prosjektet viser hvordan man kan bruke **lineær regresjon** til å modellere
daglige avkastninger på naturgass-ETF (UNG) ved hjelp av **laggede returer**.

---

## Data
- Kilde: Yahoo Finance (`yfinance`)
- Instrument: UNG (United States Natural Gas Fund) som proxy for Henry Hub spot
- Periode: 2018-01-01 til 2024-12-31
- Frekvens: daglig close

---

## Features
- `Lag1`: gårsdagens retur  
- `Lag2`: retur to dager tilbake  
- **Target**: dagens retur  

---

## Modell
- **Lineær regresjon** (OLS)
- Treningsoppsett: enkel fit på hele datasettet (ingen CV i denne enkle versjonen)
- Evaluering:
  - R² (forklaringsgrad)
  - MSE (gjennomsnittlig kvadratfeil)
- Visualisering av faktiske vs predikerte returer

---

## Run
```bash
pip install pandas yfinance matplotlib scikit-learn
python gas2.py
