# Raw data vs demo preview (GIF)

> Purpose: do **not** treat the pretty curves in `assets/previews/*.gif` as the numeric / temporal ground truth of the released npy zips.  
> Canonical loader: `scripts/load_clip.py` (returns arrays as stored). Demo plotter: `scripts/plot_skeleton_preview.py`.

## Bottom line

| | Released data (npy / zip) | Demo GIF (`assets/previews/`) |
|--|---------------------------|--------------------------------|
| Meaning | Paired action windows from manual segmentation | Visualization **for readability only** |
| Writes npy? | — | **No** — transforms exist only in plotting memory |
| Use as training input? | Yes (paper used GAF separately; GAF not shipped here) | **No** — not “already aligned / gravity-free” truth |

## 1. What the released data actually is

### 1.1 Pairing (window-level, not frame-level)

- Same `clip_id` ⇒ same session, same annotated action window (beat cues + synced video; paper §5.1.3).
- **D01** (Garmin real watch, right-wrist ACC) and **D03** (Rokoko→UE4, 20 joints × ACC/GYR/POS) share that **window**, but:
  - Almost every clip has `len(D01) ≠ len(D03)` (under a ~10 Hz view, median length gap ≈ **26** frames; watch usually shorter).
  - There is **no** shared sample clock with 1:1 frames, and **no** per-frame alignment table in the release.
- Cutting scripts (`MoC_raw/scripts/`) apply a per-subject whole-window **`time_offset` (seconds)** to map watch CSV / UE4 timelines onto the same action log. That is **window** alignment, not per-frame cross-correlation.

### 1.2 What `load_clip(clip_id)` returns

- `watch_acc`: `(T1, 3)`, **includes gravity / DC**, same units as capture (**not** demeaned).
- `mocap_*`: `(T3, 20, …)` ACC/GYR/POS; POS is UE4 world coordinates (Z up); watch wear site ↔ **J11**.
- D03 was stored as `float * 100` then integerized (see README / channels). Watch and MoCap **coordinate frames differ** — do **not** treat xyz as the same axes when overlaying.

### 1.3 How the paper trained

- Each 3-axis stream: min-max → **[-1, 1]**, then fixed-size **GAF** (224×224).
- Training therefore did **not** need per-frame D01↔D03 time alignment.
- **This repo ships raw npy only — no GAF.**

## 2. What the demo GIF adds (display-only)

Implemented in `scripts/plot_skeleton_preview.py`. These steps affect **figures only**; they do **not** modify `data/*.zip`.

| Step | Applied to | What | Why for demo | In released npy? |
|------|------------|------|--------------|------------------|
| A. Linear resample | D01 → length = D03 `T` | Uniform `np.interp` | Same horizontal axis as skeleton / J11 | **No**; raw `T1≠T3` |
| B. ACC demean (zeroing) | Joint ACC + watch ACC | Subtract per-axis mean (remove gravity / DC) | Curves sit near **0**, easier to compare peaks | **No**; npy still has DC |
| C. Cross-corr time shift (default on) | Resampled+demeaned watch vs J11 `‖ACC‖` | Search ±30 frames; shift watch **only if** **r≥0.50** and lag is **not** on the search boundary; edges NaN. Periodic walking often has weak r + boundary lag → **intentionally no shift** | Reduce apparent lead/lag when the peak is trustworthy | **No**; npy has no such shift; weak / boundary peaks skipped (e.g. A01 walk, some A07) |
| D. Twin panels (right of skeleton) | D01 vs D03@J11 | Equal size, shared y-lim, ACC only | Highlight real watch vs MoCap wrist | Layout is demo; values after A–C |
| E. J11 in the 20-panel grid | **D03 MoCap ACC only** | Same RGB + ylim as top-right D03; **no GYR, no watch** | Avoid confusing “watch orange” with MoCap | Raw J11 npy still has ACC+GYR+POS; watch lives in D01 |
| F. Skeleton POS | Hip (J04) centered | Subtract hip translation | Subject stands in frame | Raw POS is world coordinates, not centered |

Disable time shift (still demean + resample):

```bash
python3 scripts/plot_skeleton_preview.py --typical --no-xcorr
```

## 3. Common misconceptions

| GIF looks like… | Reality… |
|-----------------|----------|
| Watch and MoCap are frame-synced | Same **window** only; demo may still xcorr-shift the watch |
| ACC naturally oscillates around 0 | Usually **demean**; at rest a watch axis is often near ±g |
| Bottom J11 orange box = watch curve | Current J11 cell = **MoCap ACC** (blue border, `= D03 ACC`); watch is only the top-right orange panel |
| RGB axes are pointwise comparable | Different frames; at best energy / magnitude rhythm — and demo may have shifted time |
| Paper trained with this alignment | Paper used GAF; it does not depend on this GIF’s resample / shift |

## 4. If you train a time-series model on the real data

1. Load both streams with `load_clip` as stored.  
2. Choose **your own** temporal alignment (resample, crop `min(T)`, xcorr, DTW, …) — this repo does not prescribe one.  
3. If you need gravity removal, do it in **your** preprocessing and log it; do **not** assume the GIF format is the release format.  
4. To reproduce the website GIFs: run `plot_skeleton_preview.py` (xcorr on by default, with the gates above).

## 5. Related paths

| Path | Role |
|------|------|
| `data/D01_watch_acc.zip` / `data/D03_mocap_joints.zip` | Release |
| `scripts/load_clip.py` | Raw load |
| `scripts/plot_skeleton_preview.py` | Demo transforms |
| `assets/previews/A0*.gif` | Demo outputs |
| `README.md` / `README.zh-CN.md` (alignment section) | Window pairing + paper multi-level alignment summary |
| `MoC_raw/scripts/Readme.txt` | Per-subject `time_offset` notes |

---

## Abbreviations

- **ACC** = Accelerometer  
- **GYR** = Gyroscope  
- **POS** = Position (UE4 joint coordinates in this repo)  
- **D01** = Real watch ACC channel id in this release  
- **D03** = MoCap full-joint channel id in this release  
- **J11** = Joint 11, right wrist (watch wear site)  
- **MoCap** = Motion Capture  
- **GAF** = Gramian Angular Field (time series → image)  
- **xcorr** = Cross-correlation (demo lag estimate)  
- **demean** = Subtract mean (demo gravity / DC removal → curves near 0)  
- **npy** = NumPy array file  
- **UE4** = Unreal Engine 4  
- **DTW** = Dynamic Time Warping  
- **LSCA** = Layered Structural Contrastive Alignment (paper training; not a GIF step)  
- **KD** = Knowledge Distillation  
- **CE** = Cross-Entropy loss  
