"""Recompute the style capacity curve and style confusion matrix.

Uses the released generations in generations/ and the held-out real texts
(slice [150:250] of each author). Writes CSVs to results/ and figures to figures/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from build_authors import load_authors

ROOT = Path(__file__).resolve().parent.parent
DATA_SIZES = [25, 50, 100]

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def style_similarity(real, gen):
    real_emb = embedder.encode(real)
    gen_emb = embedder.encode(gen)
    return cosine_similarity(gen_emb, real_emb).mean()


def load_generations(author, size):
    return json.loads((ROOT / "generations" / f"{author}_{size}.json").read_text())


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    authors = load_authors()
    author_list = list(authors)
    held_out = {a: authors[a][150:250] for a in author_list}

    # Style capacity curve
    rows = []
    for a in author_list:
        for size in DATA_SIZES:
            rows.append({
                "author": a,
                "train_samples": size,
                "style_similarity": float(style_similarity(held_out[a], load_generations(a, size))),
            })
    capacity = pd.DataFrame(rows)
    capacity.to_csv(ROOT / "results" / "capacity_curve.csv", index=False)

    fig = plt.figure(figsize=(8, 5))
    for a in author_list:
        sub = capacity[capacity.author == a]
        sns.lineplot(x=sub.train_samples, y=sub.style_similarity, marker="o", label=a)
    plt.xlabel("Training Samples")
    plt.ylabel("Style Similarity")
    plt.title("Style Capacity Curve")
    plt.grid(True)
    save(fig, "capacity_curve")

    # Style confusion matrix (models trained on 100 samples)
    n = len(author_list)
    matrix = np.zeros((n, n))
    for i, a in enumerate(author_list):
        gen = load_generations(a, 100)
        for j, b in enumerate(author_list):
            matrix[i][j] = style_similarity(held_out[b], gen)
    confusion = pd.DataFrame(matrix, index=author_list, columns=author_list)
    confusion.index.name = "generated_from"
    confusion.to_csv(ROOT / "results" / "confusion_matrix.csv")

    fig = plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, xticklabels=author_list, yticklabels=author_list)
    plt.xlabel("Compared Author")
    plt.ylabel("Generated From")
    plt.title("Style Confusion Matrix")
    save(fig, "confusion_matrix")

    print(capacity.pivot(index="author", columns="train_samples", values="style_similarity").round(4))
    print(confusion.round(4))


if __name__ == "__main__":
    main()
