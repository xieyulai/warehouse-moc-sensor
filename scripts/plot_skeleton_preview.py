#!/usr/bin/env python3
"""Preferred viz: POS skeleton (left) + twin wrist ACC panels (right) + per-joint ACC/GYR.

Display-only (does NOT change stored npy):
  - ACC demean (per-axis mean / gravity DC removed)
  - linear resample D01 → D03 length
  - optional xcorr time-shift of D01 vs J11 |ACC| so peaks line up for preview

Right of skeleton (same size + shared y-lim):
  - D01 watch ACC @ right wrist (J11)
  - D03 J11 MoCap ACC @ same joint

All on-figure text is English.

Usage:
  python3 plot_skeleton_preview.py [KEY]
  python3 plot_skeleton_preview.py --typical
  python3 plot_skeleton_preview.py --typical --no-xcorr
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_clip import load_clip as load_raw_clip

OUT = ROOT / "assets" / "previews"
XCORR_MAXLAG = 30
# Walking / weak matches often peak near the lag boundary with r≈0.3 — require a clearer peak.
XCORR_MIN_CORR = 0.50
XCORR_REJECT_NEAR_BOUND = 2  # |lag| >= maxlag-this → treat as unreliable

BONES = [
    (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (6, 7), (7, 8), (2, 9), (9, 10),
    (10, 11), (11, 12), (4, 13), (13, 14), (14, 15), (15, 16), (4, 17),
    (17, 18), (18, 19), (19, 20),
]
NAMES = [
    "head", "neck", "chest", "hip", "l_sho", "l_elb", "l_wri", "l_hand",
    "r_sho", "r_elb", "r_wri", "r_hand", "l_hip", "l_kne", "l_ank", "l_toe",
    "r_hip", "r_kne", "r_ank", "r_toe",
]
AXIS_C = ["#d62728", "#2ca02c", "#1f77b4"]  # x y z
HIP_IDX = 3  # J04
R_WRIST_IDX = 10  # J11 r_wri = right-wrist watch (D01)
WATCH_COLOR = "#ff7f00"
MOCAP_COLOR = "#1f77b4"

TYPICAL = [
    ("A0101_S0101_T008", "A0101 walk", "A01_walk_A0101_S0101_T008.gif"),
    ("A0201_S0101_T002", "A0201 put-down (both hands)", "A02_picking_A0201_S0101_T002.gif"),
    ("A0301_S0101_T006", "A0301 place on shelf", "A03_shelf_A0301_S0101_T006.gif"),
    ("A0401_S0101_T010", "A0401 light throw", "A04_throwing_A0401_S0101_T010.gif"),
    ("A0501_S0101_T009", "A0501 standing finger check", "A05_pointing_A0501_S0101_T009.gif"),
    ("A0601_S0101_T010", "A0601 forklift drive", "A06_driving_A0601_S0101_T010.gif"),
    ("A0703_S0101_T005", "A0703 standing scan", "A07_scanning_A0703_S0101_T005.gif"),
    ("A0801_S0101_T007", "A0801 packing", "A08_packing_A0801_S0101_T007.gif"),
]


def resample(a: np.ndarray, n: int) -> np.ndarray:
    a = a.astype(float)
    if len(a) == n:
        return a
    idx = np.linspace(0, len(a) - 1, n)
    out = np.zeros((n, a.shape[1]))
    for c in range(a.shape[1]):
        out[:, c] = np.interp(idx, np.arange(len(a)), a[:, c])
    return out


def demean_display(a: np.ndarray) -> np.ndarray:
    """Display only: remove per-axis mean (gravity / DC)."""
    return a.astype(float) - a.astype(float).mean(axis=0, keepdims=True)


def xcorr_lag_1d(a: np.ndarray, b: np.ndarray, maxlag: int = XCORR_MAXLAG) -> tuple[int, float]:
    """Lag of a relative to b. +lag => a is late (peaks later than b)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    best_lag, best_c = 0, -2.0
    for lag in range(-maxlag, maxlag + 1):
        if lag > 0:
            aa, bb = a[lag:], b[:-lag]
        elif lag < 0:
            aa, bb = a[:lag], b[-lag:]
        else:
            aa, bb = a, b
        if len(aa) < 10:
            continue
        c = float(np.dot(aa, bb) / len(aa))
        if c > best_c:
            best_c, best_lag = c, lag
    return best_lag, best_c


def shift_rows(a: np.ndarray, lag: int) -> np.ndarray:
    """Align late signal: a_out[t] = a[t+lag]. Edges filled with NaN (matplotlib skips)."""
    out = np.full_like(a, np.nan, dtype=float)
    if lag > 0:
        out[: len(a) - lag] = a[lag:]
    elif lag < 0:
        out[-lag:] = a[: len(a) + lag]
    else:
        out[:] = a
    return out


def align_watch_display(
    d01: np.ndarray, j11: np.ndarray, *, enable: bool = True
) -> tuple[np.ndarray, int, float, bool]:
    """Display-only xcorr shift of watch vs MoCap J11 |ACC|.

    Returns (d01_aligned, lag, corr, applied).
    Skips shift when correlation is weak or lag sits on the search boundary
    (common for periodic walking after heavy resample).
    """
    if not enable or len(d01) < 20:
        return d01, 0, float("nan"), False
    lag, corr = xcorr_lag_1d(
        np.linalg.norm(d01, axis=1),
        np.linalg.norm(j11, axis=1),
    )
    near_bound = abs(lag) >= XCORR_MAXLAG - XCORR_REJECT_NEAR_BOUND
    if corr < XCORR_MIN_CORR or near_bound:
        return d01, lag, corr, False
    return shift_rows(d01, lag), lag, corr, True


def load_clip(key: str):
    d = load_raw_clip(key)
    acc, gyr, pos = d["mocap_acc"], d["mocap_gyr"], d["mocap_pos"]
    d01 = resample(d["watch_acc"], acc.shape[0])
    return acc, gyr, pos, d01


def _style_twin_panel(ax, title: str, edge: str, T: int, ylim: tuple[float, float]):
    ax.set_title(title, fontsize=10, pad=4, color=edge, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(edge)
        spine.set_linewidth(2.0)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(*ylim)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("frame", fontsize=8)


def make_gif(
    key: str,
    label: str | None = None,
    out_name: str | None = None,
    *,
    xcorr_align: bool = True,
) -> Path:
    acc_raw, gyr, pos, d01_raw = load_clip(key)
    T = acc_raw.shape[0]
    title_label = label or key

    # Display-only demean for ACC (joint panels + twin right panels)
    acc = demean_display(acc_raw)
    d01 = demean_display(d01_raw)
    j11 = acc[:, R_WRIST_IDX, :]
    d01, lag, corr, applied = align_watch_display(d01, j11, enable=xcorr_align)

    # Shared y-lim for the two wrist ACC panels (same scale / proportion)
    twin_peak = float(
        max(
            np.nanmax(np.abs(d01)),
            np.abs(j11).max(),
            1.0,
        )
    )
    twin_ylim = (-twin_peak * 1.05, twin_peak * 1.05)

    pos_c = pos - pos[:, HIP_IDX : HIP_IDX + 1, :]
    pad = 12
    mins, maxs = pos_c.min((0, 1)), pos_c.max((0, 1))
    lims = [(mins[i] - pad, maxs[i] + pad) for i in range(3)]

    # Top: skeleton LEFT | two stacked ACC compare panels RIGHT (fill the blank)
    # Bottom: 4x5 per-joint ACC/GYR
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3.5, 4.0], hspace=0.28)
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.18)
    gs_right = gs_top[0, 1].subgridspec(2, 1, hspace=0.32)
    gs_bot = gs[1].subgridspec(4, 5, hspace=0.55, wspace=0.32)

    ax3d = fig.add_subplot(gs_top[0, 0], projection="3d")
    ax_watch = fig.add_subplot(gs_right[0, 0])
    ax_mocap = fig.add_subplot(gs_right[1, 0])

    skel_lines = []
    for bi, _ in enumerate(BONES):
        a, b = BONES[bi]
        if {a, b} & {10, 11, 12}:
            (ln,) = ax3d.plot([], [], [], color=WATCH_COLOR, lw=4.5, alpha=0.95)
        else:
            (ln,) = ax3d.plot([], [], [], color="#1f77b4", lw=3.2)
        skel_lines.append(ln)
    skel_pts = ax3d.scatter([], [], [], c="#d62728", s=55, depthshade=True)
    watch_halo = ax3d.scatter(
        [], [], [], s=420, facecolors="none", edgecolors=WATCH_COLOR, linewidths=2.8, zorder=5
    )
    watch_dot = ax3d.scatter(
        [], [], [], c=WATCH_COLOR, s=160, marker="o", depthshade=False, zorder=6,
        edgecolors="k", linewidths=1.2,
    )
    txts = [ax3d.text(0, 0, 0, f"J{j+1}", fontsize=9, color="0.2") for j in range(20)]
    ax3d.set_xlim(lims[0])
    ax3d.set_ylim(lims[1])
    ax3d.set_zlim(lims[2])
    ax3d.set_xlabel("X", fontsize=11)
    ax3d.set_ylabel("Y", fontsize=11)
    ax3d.set_zlabel("Z up", fontsize=11)
    ax3d.tick_params(labelsize=8)
    ax3d.view_init(elev=18, azim=-60)
    try:
        ax3d.set_box_aspect([1, 1, 1.15])
    except Exception:
        pass

    if applied:
        watch_title = (
            f"D01 watch ACC @ J11  (demeaned + xcorr shift {lag:+d} fr, r={corr:.2f})"
        )
        align_note = f"display xcorr shift {lag:+d} (watch vs J11 |ACC|; npy unchanged)"
    elif xcorr_align and not np.isnan(corr):
        why = f"r={corr:.2f}"
        if abs(lag) >= XCORR_MAXLAG - XCORR_REJECT_NEAR_BOUND:
            why += ", lag@bound"
        watch_title = (
            f"D01 watch ACC @ J11  (demeaned; xcorr skipped, {why})"
        )
        align_note = f"xcorr skipped ({why}; npy unchanged)"
    else:
        watch_title = "D01 watch ACC @ J11  (display only: demeaned)"
        align_note = "demeaned only (npy unchanged)"

    for k in range(3):
        ax_watch.plot(d01[:, k], color=AXIS_C[k], lw=1.4)
        ax_mocap.plot(j11[:, k], color=AXIS_C[k], lw=1.4)
    _style_twin_panel(ax_watch, watch_title, WATCH_COLOR, T, twin_ylim)
    _style_twin_panel(
        ax_mocap,
        "D03 MoCap ACC @ J11  (display only: demeaned)",
        MOCAP_COLOR,
        T,
        twin_ylim,
    )

    vlines = []
    vlines.append(ax_watch.axvline(0, color="k", lw=1.0, ls=":"))
    vlines.append(ax_mocap.axvline(0, color="k", lw=1.0, ls=":"))

    for j in range(20):
        r, c = divmod(j, 5)
        ax = fig.add_subplot(gs_bot[r, c])
        if j == R_WRIST_IDX:
            # Same stream as top-right D03 panel: MoCap ACC only, same RGB
            for k in range(3):
                ax.plot(j11[:, k], color=AXIS_C[k], lw=1.4, ls="-")
            ax.set_title(
                f"J{j+1:02d} {NAMES[j]} = D03 ACC",
                fontsize=8, pad=1, color=MOCAP_COLOR, fontweight="bold",
            )
            for spine in ax.spines.values():
                spine.set_color(MOCAP_COLOR)
                spine.set_linewidth(2.0)
            ax.set_ylim(*twin_ylim)
        else:
            for k in range(3):
                ax.plot(acc[:, j, k], color=AXIS_C[k], lw=0.9, ls="-")
                ax.plot(gyr[:, j, k], color=AXIS_C[k], lw=0.9, ls="--", alpha=0.85)
            ax.set_title(f"J{j+1:02d} {NAMES[j]}", fontsize=8, pad=1)
        ax.axhline(0, color="0.45", lw=0.7)
        ax.set_xlim(0, T - 1)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.25)
        vlines.append(ax.axvline(0, color="k", lw=0.9, ls=":"))

    handles = [
        Line2D([0], [0], color=AXIS_C[0], lw=2, ls="-", label="ACC x"),
        Line2D([0], [0], color=AXIS_C[1], lw=2, ls="-", label="ACC y"),
        Line2D([0], [0], color=AXIS_C[2], lw=2, ls="-", label="ACC z"),
        Line2D([0], [0], color=AXIS_C[0], lw=2, ls="--", label="GYR x"),
        Line2D([0], [0], color=AXIS_C[1], lw=2, ls="--", label="GYR y"),
        Line2D([0], [0], color=AXIS_C[2], lw=2, ls="--", label="GYR z"),
        Line2D(
            [0], [0], color=WATCH_COLOR, marker="o", markersize=10, lw=0,
            markeredgecolor="k", label="R-wrist site J11",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=7,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.995),
        frameon=True,
    )

    fig.suptitle(
        f"{title_label}  |  {key}  |  ACC demeaned; {align_note}",
        fontsize=12,
        y=1.02,
    )

    def update(t):
        pts = pos_c[t]
        for ln, (a, b) in zip(skel_lines, BONES):
            pa, pb = pts[a - 1], pts[b - 1]
            ln.set_data([pa[0], pb[0]], [pa[1], pb[1]])
            ln.set_3d_properties([pa[2], pb[2]])
        skel_pts._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        w = pts[R_WRIST_IDX]
        watch_halo._offsets3d = ([w[0]], [w[1]], [w[2]])
        watch_dot._offsets3d = ([w[0]], [w[1]], [w[2]])
        for j, tx in enumerate(txts):
            tx.set_position((pts[j, 0], pts[j, 1]))
            tx.set_3d_properties(pts[j, 2] + 2.5, "z")
        ax3d.set_title(f"POS skeleton   frame {t}/{T - 1}", fontsize=14, pad=10, color="0.15")
        for v in vlines:
            v.set_xdata([t, t])
        return []

    frames = list(range(0, T, 2))
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
    OUT.mkdir(parents=True, exist_ok=True)
    out_gif = OUT / (out_name or f"skeleton_BIG_per_joint_{key}.gif")
    anim.save(out_gif, writer="pillow", fps=12, dpi=90)
    plt.close()
    print(
        f"  twin ylim=±{twin_peak:.0f}  xcorr lag={lag:+d} r={corr:.2f} "
        f"applied={applied}  "
        f"D01 raw mean={d01_raw.mean(0).round(1)}  "
        f"J11 raw mean={acc_raw[:, R_WRIST_IDX, :].mean(0).round(1)}",
        flush=True,
    )
    return out_gif


if __name__ == "__main__":
    args = sys.argv[1:]
    xcorr_align = True
    if "--no-xcorr" in args:
        xcorr_align = False
        args = [a for a in args if a != "--no-xcorr"]

    if args and args[0] == "--typical":
        jobs = [(k, lab, name) for k, lab, name in TYPICAL]
    elif args:
        key = args[0]
        lab = args[1] if len(args) > 1 else None
        jobs = [(key, lab, None)]
    else:
        jobs = [("A0101_S0101_T005", "A0101 walk", None)]

    for key, label, out_name in jobs:
        print("making", key, label or "", flush=True)
        path = make_gif(key, label, out_name, xcorr_align=xcorr_align)
        print("wrote", path, flush=True)
