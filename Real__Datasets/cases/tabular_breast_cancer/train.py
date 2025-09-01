#!/usr/bin/env python3
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, RocCurveDisplay
from joblib import dump
import matplotlib.pyplot as plt
from common.utils import ART_MODELS, ART_REPORTS, save_json, save_fig
from rich import print as rprint

def main():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 1) Logistic Regression pipeline
    logit = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ])
    logit.fit(Xtr, ytr)

    # 2) RandomForest
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(Xtr, ytr)

    models = {"logreg": logit, "rf": rf}
    results = {}

    for name, mdl in models.items():
        yhat = mdl.predict(Xte)
        yproba = mdl.predict_proba(Xte)[:, 1] if hasattr(mdl, "predict_proba") else None
        acc = accuracy_score(yte, yhat)
        f1  = f1_score(yte, yhat)
        auc = roc_auc_score(yte, yproba) if yproba is not None else None

        # 5-fold CV på hele datasettet for robusthet
        cv = cross_val_score(mdl, X, y, cv=5, scoring="accuracy").mean()

        results[name] = {"accuracy": acc, "f1": f1, "auc": auc, "cv_acc_5fold": cv}
        dump(mdl, ART_MODELS / f"breast_{name}.joblib")

        # ROC-plot for modeller med sannsynlighet
        if yproba is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_predictions(yte, yproba, ax=ax)
            ax.set_title(f"ROC — {name}")
            save_fig(fig, ART_REPORTS / f"breast_{name}_roc.png")

    rprint("[bold]Breast Cancer results[/bold]", results)
    save_json(results, ART_REPORTS / "breast_results.json")

if __name__ == "__main__":
    main()
