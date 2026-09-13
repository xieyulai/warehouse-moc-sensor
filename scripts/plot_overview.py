#!/usr/bin/env python3
"""Dataset overview figure for the public README (no photos, no paper crops).

python3 scripts/plot_overview.py
→ assets/overview.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_clip import load_clip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "overview.png"

BONES = [
    (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (6, 7), (7, 8), (2, 9), (9, 10),
    (10, 11), (11, 12), (4, 13), (13, 14), (14, 15), (15, 16), (4, 17),
    (17, 18), (18, 19), (19, 20),
]
# schematic 2D tree (x, y), y up, similar to paper Fig.6 joint panel
TREE = {
    1: (0.0, 8.0), 2: (0.0, 7.0), 3: (0.0, 5.6), 4: (0.0, 4.2),
    5: (-1.6, 6.2), 6: (-1.6, 4.8), 7: (-1.6, 3.4), 8: (-1.6, 2.2),
    9: (1.6, 6.2), 10: (1.6, 4.8), 11: (1.6, 3.4), 12: (1.6, 2.2),
    13: (-0.7, 3.2), 14: (-0.7, 1.8), 15: (-0.7, 0.6), 16: (-0.7, -0.4),
    17: (0.7, 3.2), 18: (0.7, 1.8), 19: (0.7, 0.6), 20: (0.7, -0.4),
}
LEFT = {5, 6, 7, 8, 13, 14, 15, 16}
RIGHT = {9, 10, 11, 12, 17, 18, 19, 20}
WATCH = 11
AXIS_C = ["#d62728", "#2ca02c", "#1f77b4"]
WATCH_C = "#ff7f00"

PREVIEWS = [
    ("A0101_S0101_T008", "A01 walking"),
    ("A0201_S0101_T002", "A02 picking"),
    ("A0301_S0101_T006", "A03 shelf"),
    ("A0401_S0101_T010", "A04 throwing"),
    ("A0501_S0101_T009", "A05 pointing"),
    ("A0601_S0101_T010", "A06 driving"),
    ("A0703_S0101_T005", "A07 scanning"),
    ("A0801_S0101_T007", "A08 packing"),
]


def _norm_xyz(x: np.ndarray) -> np.ndarray:
    y = x.astype(float)
    for c in range(y.shape[1]):
        lo, hi = y[:, c].min(), y[:, c].max()
        if hi - lo < 1e-9:
            y[:, c] = 0.0
        else:
            y[:, c] = 2 * (y[:, c] - lo) / (hi - lo) - 1
    return y


def _draw_skel_3d(ax, pos_c, t, title):
    pts = pos_c[t]
    for a, b in BONES:
        pa, pb = pts[a - 1], pts[b - 1]
        col = WATCH_C if {a, b} & {10, 11, 12} else "#4c78a8"
        lw = 2.2 if {a, b} & {10, 11, 12} else 1.4
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color=col, lw=lw)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="#c44e52", s=8, depthshade=False)
    w = pts[WATCH - 1]
    ax.scatter([w[0]], [w[1]], [w[2]], s=70, c=WATCH_C, edgecolors="k", linewidths=0.5, depthshade=False)
    lo, hi = pos_c.min((0, 1)), pos_c.max((0, 1))
    pad = 8
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(lo[2] - pad, hi[2] + pad)
    ax.view_init(elev=16, azim=-58)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    try:
        ax.set_box_aspect([1, 1, 1.15])
    except Exception:
        pass
    ax.set_title(title, fontsize=9, pad=2)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)


def main():
    walk = load_clip("A0101_S0101_T008")
    fig = plt.figure(figsize=(13.2, 8.6), facecolor="white")
    gs = GridSpec(
        3, 8, figure=fig, height_ratios=[1.35, 1.05, 1.15],
        hspace=0.48, wspace=0.32, left=0.03, right=0.99, top=0.86, bottom=0.04,
    )

    # --- joint schematic ---
    axj = fig.add_subplot(gs[0, 0:2])
    for a, b in BONES:
        pa, pb = TREE[a], TREE[b]
        col = WATCH_C if {a, b} & {10, 11, 12} else "#888"
        axj.plot([pa[0], pb[0]], [pa[1], pb[1]], color=col, lw=2.0, zorder=1)
    for j, (x, y) in TREE.items():
        if j in LEFT:
            c = "#4c78a8"
        elif j in RIGHT:
            c = "#59a14f"
        else:
            c = "#e15759"
        s = 160 if j == WATCH else 46
        axj.scatter([x], [y], s=s, c=WATCH_C if j == WATCH else c, zorder=3,
                    edgecolors="k" if j == WATCH else "none", linewidths=0.7)
        axj.text(x + (0.38 if j in RIGHT else -0.38), y, f"J{j:02d}",
                 fontsize=6.5, ha="left" if j in RIGHT else "right", va="center",
                 color=WATCH_C if j == WATCH else "0.2",
                 fontweight="bold" if j == WATCH else "normal")
    axj.set_xlim(-2.6, 2.8)
    axj.set_ylim(-1.0, 8.7)
    axj.set_aspect("equal")
    axj.axis("off")
    axj.set_title("20 joints  (orange = J11 watch)", fontsize=10)

    # --- three signals ---
    wacc = _norm_xyz(walk["watch_acc"])
    macc = _norm_xyz(walk["mocap_acc"][:, 10, :])
    mgyr = _norm_xyz(walk["mocap_gyr"][:, 10, :])
    panels = [
        (gs[0, 2:4], wacc, "D01 watch ACC (student)"),
        (gs[0, 4:6], macc, "D03 J11 ACC (teacher, estimated)"),
        (gs[0, 6:8], mgyr, "D03 J11 GYR (teacher, estimated)"),
    ]
    for spec, data, title in panels:
        ax = fig.add_subplot(spec)
        for k in range(3):
            ax.plot(data[:, k], color=AXIS_C[k], lw=1.0)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xlim(0, len(data) - 1)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("frame", fontsize=8)

    # --- 8 action stick figures ---
    for i, (cid, title) in enumerate(PREVIEWS):
        ax = fig.add_subplot(gs[1 if i < 4 else 2, (i % 4) * 2:(i % 4) * 2 + 2], projection="3d")
        d = load_clip(cid)
        pos = d["mocap_pos"]
        pos_c = pos - pos[:, 3:4, :]
        t = pos_c.shape[0] // 2
        _draw_skel_3d(ax, pos_c, t, title)

    fig.legend(
        handles=[
            Line2D([0], [0], color=AXIS_C[0], lw=2, label="x"),
            Line2D([0], [0], color=AXIS_C[1], lw=2, label="y"),
            Line2D([0], [0], color=AXIS_C[2], lw=2, label="z"),
            Line2D([0], [0], color=WATCH_C, marker="o", lw=0, markersize=8,
                   markeredgecolor="k", label="J11 right-wrist watch"),
        ],
        loc="upper center", ncol=4, fontsize=8, frameon=False,
        bbox_to_anchor=(0.62, 0.925),
    )
    fig.suptitle(
        "Warehouse MoC+Sensor    20 MoCap joints (D03) + right-wrist watch ACC (D01)\n"
        "No subject photos.  Cite: Xie et al., Expert Syst. Appl. 319 (2026) 132094.",
        fontsize=12, y=0.98,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    plt.close()
    print(OUT, "MB", round(OUT.stat().st_size / 1e6, 2))


if __name__ == "__main__":
    main()
