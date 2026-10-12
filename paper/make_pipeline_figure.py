r"""Draw Fig. 2, the pipeline, from source.

The figure had been a PNG with no script behind it, and paper/README.md said it was TikZ
inside main.tex, which it never was. A review then asked for its wording to change
("the only trained part" -> trainable), which a PNG without a source cannot take.

Layout follows the original: input, the two frozen models side by side, the selector
they both feed, the fixed sharpening that also takes the depth directly, and the output.
Drawn a little wider than before, which keeps it two points shorter in the column; the
page it shares with the full-width Fig. 1 was 1.4pt over.

    python make_pipeline_figure.py      # writes figures/pipeline.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures" / "pipeline.png"
EDGE, TEXT, SUB = "#4a4f57", "#111111", "#555555"
HILITE_EDGE, HILITE_FILL = "#1f4e8c", "#eef1f6"

fig, ax = plt.subplots(figsize=(7.2, 3.35))
ax.set_xlim(0, 14.4)
ax.set_ylim(-0.08, 6.7)
ax.axis("off")


def box(x, y, w, h, title, sub=None, hilite=False, centre=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=2.2 if hilite else 1.2,
        edgecolor=HILITE_EDGE if hilite else EDGE,
        facecolor=HILITE_FILL if hilite else "white"))
    if centre:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=9.0, color=TEXT)
        return
    tx = x + 0.28
    if sub:
        ax.text(tx, y + h * 0.68, title, ha="left", va="center", fontsize=8.8,
                fontweight="bold", color=TEXT)
        ax.text(tx, y + h * 0.28, sub, ha="left", va="center", fontsize=7.8, color=SUB)
    else:
        ax.text(tx, y + h / 2, title, ha="left", va="center", fontsize=8.8,
                fontweight="bold", color=TEXT)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.2,
                                shrinkA=0, shrinkB=0, mutation_scale=11))


def line(xs, ys):
    ax.plot(xs, ys, color=EDGE, lw=1.2, solid_capstyle="butt")


# boxes
box(4.9, 5.85, 4.4, 0.62, "Input: one RGB image", centre=True)
box(0.9, 4.2, 5.8, 0.98, "Depth model (frozen)", "affine-invariant depth")
box(7.6, 4.2, 5.8, 0.98, "Segmentation (frozen)", "region borders → candidate contours")
box(3.0, 2.55, 8.2, 1.1, "Selector (U-Net, 1.9M, trainable)",
    "a probability per contour pixel; keeps the top 6%", hilite=True)
box(3.0, 1.05, 8.2, 0.9, "Sharpening (fixed, no parameters)",
    "discard ±2 px, refill each side from itself")
box(4.6, 0.0, 5.0, 0.62, "Output: sharpened depth", centre=True)

# input -> both models
line([7.1, 7.1], [5.85, 5.5])
line([3.8, 10.5], [5.5, 5.5])
arrow(3.8, 5.5, 3.8, 5.18)
arrow(10.5, 5.5, 10.5, 5.18)

# both models -> selector
arrow(3.8, 4.2, 3.8, 3.65)
arrow(10.5, 4.2, 10.5, 3.65)

# selector -> sharpening -> output
arrow(7.1, 2.55, 7.1, 1.95)
arrow(7.1, 1.05, 7.1, 0.62)

# depth model -> sharpening, around the left
line([0.9, 0.35], [4.69, 4.69])
line([0.35, 0.35], [4.69, 1.5])
arrow(0.35, 1.5, 3.0, 1.5)
ax.text(0.5, 3.1, "predicted\ndepth", ha="left", va="center", fontsize=9, color=SUB)

fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
print(f"wrote {OUT}")
