#!/usr/bin/env python3
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from common.utils import ART_MODELS, ART_REPORTS, save_json, save_fig
from joblib import dump
from rich import print as rprint

def main():
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    Xtr, Xte, ytr, yte = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))),
        ("clf", LinearSVC())
    ])
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)

    acc = accuracy_score(yte, yhat)
    report = classification_report(
        yte, yhat, target_names=data.target_names, output_dict=True
    )

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        yte, yhat, ax=ax, xticks_rotation=90, colorbar=False
    )
    ax.set_title("20 Newsgroups — Confusion Matrix")
    save_fig(fig, ART_REPORTS / "20ng_confusion.png")

    dump(pipe, ART_MODELS / "20ng_linearsvc.joblib")
    save_json({"accuracy": acc, "classification_report": report}, ART_REPORTS / "20ng_results.json")
    rprint(f"[bold]20 Newsgroups accuracy:[/bold] {acc:.3f}")

if __name__ == "__main__":
    main()
