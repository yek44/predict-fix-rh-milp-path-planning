"""Small helpers shared by the experiment scripts."""

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECKPOINT = ROOT / "checkpoints" / "sidegru.pt"


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def read_csv(path):
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


def reduction(baseline, method):
    """Percent reduction of total time."""
    return 100.0 * (1.0 - np.sum(method) / np.sum(baseline))


def bootstrap_reduction(baseline, method, draws=10_000, seed=2026):
    """95% paired bootstrap interval of the total-time reduction."""
    baseline, method = np.asarray(baseline, float), np.asarray(method, float)
    idx = np.random.RandomState(seed).randint(0, len(baseline), size=(draws, len(baseline)))
    boot = 100.0 * (1.0 - method[idx].sum(axis=1) / baseline[idx].sum(axis=1))
    return np.percentile(boot, [2.5, 97.5])
