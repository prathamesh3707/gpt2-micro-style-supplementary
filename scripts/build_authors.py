"""Rebuild the four "author" corpora used in the paper from AG News.

AG News text is not redistributed here. This script re-applies the notebook's
selection rule and checks the result against data/author_splits.json.
"""
import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
SPLITS = json.loads((ROOT / "data" / "author_splits.json").read_text())


def load_authors():
    ds = load_dataset(SPLITS["dataset"], split=SPLITS["split"])
    texts = ds["text"]
    authors = {}
    for name, indices in SPLITS["indices"].items():
        authors[name] = [texts[i] for i in indices]
    return authors


def verify_selection():
    """Re-run the notebook's filter and confirm it picks the same indices."""
    ds = load_dataset(SPLITS["dataset"], split=SPLITS["split"])
    labels, texts = ds["label"], ds["text"]
    for label in range(4):
        picked = [
            i for i in range(len(ds))
            if labels[i] == label and 20 < len(texts[i]) <= 200
        ][:300]
        assert picked == SPLITS["indices"][f"author_{label}"], f"mismatch for author_{label}"
    print("Selection matches data/author_splits.json for all 4 authors.")


if __name__ == "__main__":
    verify_selection()
    authors = load_authors()
    for name, texts in authors.items():
        print(f"{name} ({SPLITS['label_names'][name]}): {len(texts)} texts")
        print(f"  e.g. {texts[0][:100]}")
