
---

# 4) `common/utils.py`
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401 (holdes for ev. styling)
from rich import print as rprint

ART_BASE = Path("artifacts")
ART_MODELS = ART_BASE / "models"
ART_REPORTS = ART_BASE / "reports"
for p in (ART_BASE, ART_MODELS, ART_REPORTS):
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    rprint(f"[green]Saved[/green] JSON -> {path}")

def save_fig(fig, path, dpi=160):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    rprint(f"[green]Saved[/green] fig -> {path}")
