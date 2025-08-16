"""
FX Direction Classifier (EURUSD) — STK-style ML
- Målet er å predikere om neste dags avkastning blir positiv (y=1) eller negativ (y=0)
- Modellen er en logistisk regresjon med L2-regularisering i en sklearn-pipeline
- Vi bruker tidsserie-splitt (uten shuffling) for hyperparameter-tuning for å unngå lekkasje
- Evaluerer med ROC AUC, Brier score, accuracy og konf.matrise
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, accuracy_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt


RANDOM_STATE = 42
PAIR = "EURUSD=X"           # lett å bytte til f.eks. "USDNOK=X"
START = "2015-01-01"
END = None                  # til siste tilgjengelige dato
TEST_SIZE = 0.2             # siste 20% som out-of-sample holdout
TS_SPLITS = 5               # expanding window CV på treningssettet

def load_fx_close(ticker: str, start: str, end: str = None) -> pd.Series:
    """Hent daglige close-priser fra Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end, progress=False)["Close"].dropna()
    return data

def make_features(px: pd.Series) -> pd.DataFrame:
    """Konstruer en liten, STK-vennlig feature-matrise uten lekkasje."""
    df = pd.DataFrame(index=px.index)
    df["close"] = px
    df["ret1"]  = px.pct_change(1)
    df["ret5"]  = px.pct_change(5)
    # Momentum / glidende snitt-ratio
    df["ma5"]   = px.rolling(5).mean()
    df["ma20"]  = px.rolling(20).mean()
    df["mom"]   = df["ma5"] / df["ma20"] - 1
    # Rullerende volatilitet (20-dagers std av daglig avkastning)
    df["vol20"] = df["ret1"].rolling(20).std()

    # Target: neste dags retning
    df["y_next"] = df["ret1"].shift(-1)  # morgendagens ret1
    df["y"] = (df["y_next"] > 0).astype(int)

    # Day-of-week (dummy-variabler uten første kategori for identifiability)
    dow = df.index.dayofweek  # 0=Mon
    df = df.assign(dow=dow)
    df = pd.get_dummies(df, columns=["dow"], drop_first=True)

    # Fjern rader med NaN fra rullerende beregninger og target-shift
    df = df.dropna()

    # Sett X og y
    feature_cols = [c for c in df.columns if c not in ["y", "y_next"]]
    X = df[feature_cols]
    y = df["y"]
    return X, y, feature_cols

def train_eval(X: pd.DataFrame, y: pd.Series, feature_names):
    """Train/val med tidsserie-CV på treningssett + slutt-test på holdout."""
    # Holdout: siste 20% av tid
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logit",  LogisticRegression(penalty="l2", solver="lbfgs", max_iter=200))
    ])

    # Hyperparam: C (regulariseringsstyrke)
    param_grid = {"logit__C": [0.1, 0.5, 1.0, 2.0, 5.0]}
    tscv = TimeSeriesSplit(n_splits=TS_SPLITS)

    grid = GridSearchCV(
        pipe, param_grid=param_grid, cv=tscv,
        scoring="roc_auc", n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train)

    print(f"Best C (time-series CV): {grid.best_params_['logit__C']}")
    print(f"Mean CV AUC: {grid.best_score_:.3f}")

    # Test på holdout
    best = grid.best_estimator_
    proba = best.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    auc   = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    acc   = accuracy_score(y_test, pred)
    cm    = confusion_matrix(y_test, pred)

    print("\n=== Holdout performance ===")
    print(f"AUC:   {auc:.3f}")
    print(f"Brier: {brier:.3f}  (lavere er bedre)")
    print(f"Acc:   {acc:.3f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_test, pred, digits=3))

    # En enkel koeffisient-visning (etter fit på hele treningssettet)
    # NB: For pipeline må vi hente ut koeffisienter fra siste steg
    logit = best.named_steps["logit"]
    coef = pd.Series(logit.coef_[0]_
