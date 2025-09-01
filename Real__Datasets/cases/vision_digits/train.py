#!/usr/bin/env python3
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from joblib import dump
from common.utils import ART_MODELS, ART_REPORTS, save_json, save_fig
from rich import print as rprint

def main():
    d = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(
        d.data, d.target, test_size=0.2, random_state=42, stratify=d.target
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", gamma="scale"))
    ])
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)
    acc = accuracy_score(yte, yhat)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(yte, yhat, ax=ax, colorbar=False)
    ax.set_title("Digits — Confusion Matrix")
    save_fig(fig, ART_REPORTS / "digits_confusion.png")

    dump(pipe, ART_MODELS / "digits_svc.joblib")
    save_json({"accuracy": acc}, ART_REPORTS / "digits_results.json")
    rprint(f"[bold]Digits accuracy:[/bold] {acc:.3f}")

if __name__ == "__main__":
    main()
