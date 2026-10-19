"""Draw the experiment pipeline diagram (figures/architecture.png and .pdf)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

INK = "#1f2933"
MUTED = "#52606d"
LINE = "#9aa5b1"
STAGE_FILL = "#f5f7fa"
STAGE_EDGE = "#cbd2d9"
ACCENT = "#2f6fb0"
# One color per "author" (AG News label), colorblind-safe
AUTHOR_COLORS = ["#4477aa", "#ee6677", "#228833", "#ccbb44"]
AUTHOR_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

W, H = 8.0, 3.7
Y0 = 0.17  # crop empty space under the bottom label
fig = plt.figure(figsize=(W, H - Y0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(Y0, H)
ax.axis("off")

COL_W, GAP, LEFT = 1.36, 0.25, 0.08
TOP, BOTTOM = 3.28, 0.62  # stage panel extent


def col_x(i):
    return LEFT + i * (COL_W + GAP)


def panel(i, title):
    x = col_x(i)
    ax.add_patch(FancyBboxPatch(
        (x, BOTTOM), COL_W, TOP - BOTTOM,
        boxstyle="round,pad=0,rounding_size=0.06",
        fc=STAGE_FILL, ec=STAGE_EDGE, lw=0.8))
    ax.text(x + 0.1, TOP + 0.14, str(i + 1), fontsize=8, fontweight="bold",
            color="white", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.25", fc=ACCENT, ec="none"))
    ax.text(x + 0.24, TOP + 0.14, title, fontsize=8.5, fontweight="bold",
            color=INK, ha="left", va="center")


def card(x, y, w, h, title, lines, fs=6.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04",
        fc="white", ec=STAGE_EDGE, lw=0.7))
    ax.text(x + 0.07, y + h - 0.08, title, fontsize=fs + 0.6, fontweight="bold",
            color=INK, ha="left", va="top")
    for k, line in enumerate(lines):
        ax.text(x + 0.07, y + h - 0.27 - k * 0.15, line, fontsize=fs,
                color=MUTED, ha="left", va="top")


def arrow(p, q, style="-|>", color=LINE, lw=1.0, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=8, color=color, lw=lw,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# 1. Data
panel(0, "Data")
x = col_x(0) + 0.08
cw = COL_W - 0.16
card(x, 2.40, cw, 0.78, "AG News",
     ["train split, 120k items", "keep 20 < len ≤ 200 chars", "first 300 per label"])
arrow((x + cw / 2, 2.40), (x + cw / 2, 2.28))
ax.text(x + 0.07, 2.24, "4 “authors”", fontsize=7.2, fontweight="bold",
        color=INK, va="top")
for k, (name, c) in enumerate(zip(AUTHOR_NAMES, AUTHOR_COLORS)):
    yy = 2.02 - k * 0.17
    ax.add_patch(Rectangle((x + 0.08, yy - 0.06), 0.12, 0.12, fc=c, ec="none"))
    ax.text(x + 0.27, yy, f"author_{k}  {name}", fontsize=6.6, color=INK, va="center")
card(x, 0.70, cw, 0.66, "Splits per author",
     ["train  [0:n], n = 25 / 50 / 100", "eval   [150:250], held out"], fs=6.2)

# 2. Fine-tuning
panel(1, "Fine-tuning")
x = col_x(1) + 0.08
card(x, 2.62, cw, 0.56, "GPT-2 base (124M)", ["causal LM, mlm=False"])
arrow((x + cw / 2, 2.62), (x + cw / 2, 2.48))
card(x, 1.68, cw, 0.80, "Training",
     ["2 epochs, batch 8", "lr 5e-5, max len 64", "pad token = EOS"])
arrow((x + cw / 2, 1.68), (x + cw / 2, 1.56))
ax.text(x + 0.07, 1.50, "12 models", fontsize=7.2, fontweight="bold", color=INK, va="top")
ax.text(x + 0.07, 1.34, "4 authors × 3 sizes", fontsize=6.6, color=MUTED, va="top")
gx, gy, cell, ch = x + 0.36, 0.78, 0.16, 0.085
for r, c in enumerate(AUTHOR_COLORS):
    for j in range(3):
        ax.add_patch(Rectangle((gx + j * (cell + 0.04), gy + (3 - r) * (ch + 0.025)),
                               cell, ch, fc=c, ec="none", alpha=0.45 + 0.25 * j))
for j, n in enumerate([25, 50, 100]):
    ax.text(gx + j * (cell + 0.04) + cell / 2, gy - 0.03, str(n), fontsize=5.8,
            color=MUTED, ha="center", va="top")
ax.text(gx - 0.06, gy - 0.03, "n:", fontsize=5.8, color=MUTED, ha="right", va="top")

# 3. Generation
panel(2, "Generation")
x = col_x(2) + 0.08
card(x, 2.22, cw, 0.96, "Sampling",
     ["start from EOS token", "top-k 50, top-p 0.95", "temperature 0.9", "50 new tokens"])
arrow((x + cw / 2, 2.22), (x + cw / 2, 2.10))
ax.text(x + 0.07, 2.04, "20 texts per model", fontsize=7.2, fontweight="bold",
        color=INK, va="top")
ax.text(x + 0.07, 1.88, "240 texts in total", fontsize=6.6, color=MUTED, va="top")
for k in range(4):  # stacked page icons
    px, py = x + 0.36 + k * 0.1, 0.84 + k * 0.08
    ax.add_patch(FancyBboxPatch((px, py), 0.42, 0.5, boxstyle="round,pad=0,rounding_size=0.02",
                                fc="white", ec=AUTHOR_COLORS[k], lw=0.9))
    if k == 3:
        for t in range(4):
            ax.plot([px + 0.07, px + 0.35 - (0.1 if t == 3 else 0)],
                    [py + 0.38 - t * 0.09] * 2, color=LINE, lw=0.8)

# 4. Scoring
panel(3, "Scoring")
x = col_x(3) + 0.08
card(x, 2.38, cw, 0.8, "Sentence encoder",
     ["all-MiniLM-L6-v2", "embeds generated and", "held-out real texts"])
arrow((x + cw / 2, 2.38), (x + cw / 2, 2.26))
card(x, 1.30, cw, 0.96, "Style similarity",
     ["mean pairwise cosine", "similarity between", "generated and real", "embeddings"])
ax.text(x + cw / 2, 0.98, r"$s = \frac{1}{|G||R|}\sum_{g,r} \cos(e_g, e_r)$",
        fontsize=7.5, color=INK, ha="center", va="center")

# 5. Analyses
panel(4, "Analyses")
x = col_x(4) + 0.08
card(x, 2.53, cw, 0.65, "Capacity curve",
     ["same-author similarity", "vs. training size n"], fs=6.4)
card(x, 1.70, cw, 0.65, "Confusion matrix",
     ["4×4: n=100 models", "vs. every author"], fs=6.4)
card(x, 0.74, cw, 0.80, "Style drift",
     ["author_0, n=100", "5 chained steps, each", "prompted by last output"], fs=6.4)

# Arrows between stages
mid = 1.95
for i in range(3):
    arrow((col_x(i) + COL_W + 0.02, mid), (col_x(i + 1) - 0.02, mid), color=ACCENT, lw=1.3)
src = (col_x(3) + COL_W + 0.02, mid)
for ty in (2.855, 2.025, 1.14):
    arrow(src, (col_x(4) + 0.06, ty), color=ACCENT, lw=1.1)

# Held-out eval texts route under stages 2 and 3 into scoring
ex = col_x(0) + COL_W / 2
yb = 0.42
ax.plot([ex, ex], [BOTTOM, yb], color=MUTED, lw=1.0, ls=(0, (3, 2)))
ax.plot([ex, col_x(3) + COL_W / 2], [yb, yb], color=MUTED, lw=1.0, ls=(0, (3, 2)))
arrow((col_x(3) + COL_W / 2, yb), (col_x(3) + COL_W / 2, BOTTOM), color=MUTED, ls=(0, (3, 2)))
ax.text((ex + col_x(3) + COL_W / 2) / 2, yb - 0.08,
        "held-out real texts [150:250] (never used for training)",
        fontsize=6.6, color=MUTED, ha="center", va="top", style="italic")

for ext in ("png", "pdf"):
    fig.savefig(ROOT / "figures" / f"architecture.{ext}", dpi=300, facecolor="white")
print("Wrote figures/architecture.png and figures/architecture.pdf")
