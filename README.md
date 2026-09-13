# Warehouse MoC+Sensor

Industrial warehouse / logistics action dataset with **time-aligned** full-body MoCap streams and a real wrist accelerometer.

**Cite:** Xie, Yulai; Fu, Yijia; Xu, Qing; Ren, Fang. (2026). Sensor-to-Sensor procedural co-learning for sensor-limited human action recognition. *Expert Systems with Applications, 319*, 132094. https://doi.org/10.1016/j.eswa.2026.132094

## What makes this dataset special

This is not “skeleton poses only”, and not “one IMU only”. Each clip ships **both**:

1. **Full-body MoCap (D03)** — **20 joints**, and **every joint** has **POS + ACC + GYR** (9 channels per joint: xyz position, acceleration, angular velocity).
2. **Real right-wrist watch ACC (D01)** — a **physical** Garmin accelerometer on the right wrist (site **J11**), recorded in the same sessions as the suit — not a simulated wrist stream.

That pairing is the point of the dataset: rich teacher signals on the whole body, plus a real student sensor you can actually deploy.

![A05 pointing preview: full-body POS skeleton + per-joint ACC/GYR + real watch ACC](assets/previews/A05_pointing_A0501_S0101_T009.gif)

Example clip `A0501_S0101_T009` (**pointing** check): POS skeleton left (orange = **J11**); right = twin equal panels **D01 watch ACC** vs **D03 J11 MoCap ACC** (same size & shared y-lim; **display only: demeaned + optional xcorr time-shift**). Joint grid below: demeaned ACC + GYR.

![A03 shelf preview: full-body POS skeleton + per-joint ACC/GYR + real watch ACC](assets/previews/A03_shelf_A0301_S0101_T006.gif)

Example clip `A0301_S0101_T006` (**shelf** / place on shelf): same right-side twin wrist panels + demeaned ACC display.

![Capture scene: Rokoko suit + live avatar; green circle marks right-wrist Garmin watch with ACC inset](assets/scene.jpg)

Capture scene (author photo): MoCap suit + live skeleton (left bend/box / center **pointing** with watch / right standing). Green circle = **real right-wrist Garmin Venu**; inset shows watch ACC (Az≈−1031 ≈ gravity when still).

This repository ships **two zip archives of `.npy` time series** (no GAF images). Unzipping is optional; `scripts/load_clip.py` reads npy **from inside the zips**.

**Git repository name:** `warehouse-moc-sensor`

---

## Alignment (how D01 and D03 are paired)

Two different “alignment” notions appear in the paper; only the first is about this **dataset release**.

### 1. Dataset pairing (what this repo gives you)

- **Same session, same action window.** Watch (D01) and MoCap (D03) were recorded together. Subjects followed a **synchronized beat instruction**; each sub-action was **manually segmented** using synchronized video + beats (paper §5.1.3). Same `clip_id` (e.g. `A0201_S0101_T002`) means the same annotated take.
- **Not sample-by-sample equal length.** For almost every clip, `len(D01) ≠ len(D03)` (median gap ~**26** frames at ~10 Hz). Streams share the **action interval by annotation**, not a shared sample index.
- **Paper training did not resample to a common `T`.** Each 3-axis stream was min-max normalized to **[-1, 1]** and converted to a fixed-size **GAF** image (**224×224** RGB). Different lengths / rates become the same image size, so the co-learning pipeline never needed per-frame D01↔D03 warp (paper §5.2 / Fig. 4). **This release ships raw npy only** — no GAF.
- **If you build a time-series model** on the npy: choose your own temporal alignment (e.g. linear resample one stream to the other’s `T`, or crop to `min(T1,T3)`). `scripts/load_clip.py` returns both arrays as stored.
- **Preview GIFs ≠ raw npy:** demean / resample / optional xcorr shift are display-only — see [`docs/20260914_0020_note_raw_vs_demo.md`](docs/20260914_0020_note_raw_vs_demo.md).

### 2. Model multi-level alignment (paper method, not files here)

The co-learning framework’s five-level alignment (paper §4 / Fig. 3) is **network training**, not a second time-stamp file:

| Level | What it aligns |
|------:|----------------|
| 1 | **Unified input** — GAF so heterogeneous lengths become same-size images |
| 2 | **Input** — pre-adaptive Conv-BN-ReLU (+ skip) maps student channels toward teacher |
| 3 | **Architecture** — identical backbone (e.g. ResNet18) for teacher and student |
| 4 | **Features** — MSE + **LSCA** (layered structural contrastive alignment) |
| 5 | **Output** — classification CE (+ optional KD) |

Use this section when reading the paper; it does **not** change the raw zip layout.

More channel notes: [`metadata/channels.md`](metadata/channels.md).

---

## Actions (paper: 8 warehouse classes)

Defined from real logistics / warehouse operation standards (paper §5.1.3 / Fig. 6). Recognition uses the **8-way** label `int(clip_id[1:3]) - 1` (A01…A08). Filenames also encode **25 subclasses**; full table: [`metadata/actions.csv`](metadata/actions.csv).

| ID | Code | Paper / English name | What subjects do | Example subclasses |
|---:|------|----------------------|------------------|--------------------|
| 0 | **A01** | **Walking** | Walk in the work area; may push or pull a cart | walk; walk + push cart; walk + pull cart |
| 1 | **A02** | **Picking** | Pick up / put down goods (hands and boxes) | both-hands put-down; right palm-up / palm-down put-down |
| 2 | **A03** | **Shelf** (place on shelf) | Place an item onto a shelf with the right hand | single subclass `A0301` (smallest class) |
| 3 | **A04** | **Throwing** | Throw an item (light / far / two-handed) | light R-hand; farther R-hand; both-hands |
| 4 | **A05** | **Pointing check** | Finger safety / direction check while standing or sitting | standing L/R/front; sitting L/R/front |
| 5 | **A06** | **Lift-car driving** | Forklift / lift-car style driving gestures (wheel + lever) | wheel+lever variants; extra drive/turn clips for one subject |
| 6 | **A07** | **Idle / scanning** | Stand or sit idle, or barcode-scan (stand / squat) | standing/sitting idle; standing/squatting scan |
| 7 | **A08** | **Packing** | Open carton, fill, close, and tape | packing sequence split into `A0801` / `A0802` |

Nine subjects × about **10 repetitions** per subclass under synchronized beat instructions. After removing invalid takes: **1768** clips, mean length ~**9 s**.

Homepage GIFs above: **A05 pointing** and **A03 shelf**. One official clip per class:

| Class | Clip | GIF |
|-------|------|-----|
| A01 walking | `A0101_S0101_T008` | [gif](assets/previews/A01_walk_A0101_S0101_T008.gif) |
| A02 picking | `A0201_S0101_T002` | [gif](assets/previews/A02_picking_A0201_S0101_T002.gif) |
| A03 shelf | `A0301_S0101_T006` | [gif](assets/previews/A03_shelf_A0301_S0101_T006.gif) |
| A04 throwing | `A0401_S0101_T010` | [gif](assets/previews/A04_throwing_A0401_S0101_T010.gif) |
| A05 pointing | `A0501_S0101_T009` | [gif](assets/previews/A05_pointing_A0501_S0101_T009.gif) |
| A06 driving | `A0601_S0101_T010` | [gif](assets/previews/A06_driving_A0601_S0101_T010.gif) |
| A07 scanning | `A0703_S0101_T005` | [gif](assets/previews/A07_scanning_A0703_S0101_T005.gif) |
| A08 packing | `A0801_S0101_T007` | [gif](assets/previews/A08_packing_A0801_S0101_T007.gif) |

Regenerate GIFs: `python3 scripts/plot_skeleton_preview.py --typical` (add `--no-xcorr` to skip display-only time-shift).

---

## Capture equipment

| Role | Device (as in the paper) | Wear / use | What this release stores | Rate |
|------|--------------------------|------------|--------------------------|------|
| Student / real sensor (**D01**) | **Garmin Venu** smartwatch | **Right wrist** | Accelerometer **xyz only** (`(T, 3)`) — no watch gyro, no watch pose | ~**10 Hz** |
| Teacher / MoCap (**D03**) | **Rokoko** professional motion-capture suit | Full body | After Rokoko → **Unreal Engine 4 (UE4)** virtual-sensor export: **20 joints × (POS, ACC, GYR)** = `(T, 9)` per joint | ~**10 Hz** |

### Watch — Garmin Venu (D01)

- Paper wording: *“a single Garmin VENU smartwatch on the right wrist”*.
- Placement matches MoCap joint **J11** (right wrist).
- Released modality: **accelerometer only**. The watch may have other sensors in hardware; **this dataset does not ship** watch gyroscope / orientation / GPS.
- Values are integer-scaled (not guaranteed SI); see [`metadata/channels.md`](metadata/channels.md).

### MoCap — Rokoko suit → UE4 (D03)

- Paper wording: *“a professional ROKOKO motion capture suit”* capturing full-body motion and estimating **position (POS), acceleration (ACC), and angular velocity (GYR)** at **20** joints.
- Pipeline used in the original collection (internal notes): **Rokoko take → import to UE4 → virtual sensors → CSV → `.npy`**. The public files are the final npy (inside `D03_mocap_joints.zip`), not the raw Rokoko takes or UE4 project.
- Unlike Kinect/RGB skeletons, the suit is **occlusion-free** inertial MoCap; the 20-joint tree is the same anatomy used in the paper Fig. 6 / this repo’s joint map (**J11 = right wrist**).
- Exact Rokoko SKU (e.g. Smartsuit Pro vs Pro II) is **not named** in the paper or the public scripts; cite it as a **Rokoko professional MoCap suit** unless you have a stronger internal inventory record.

### Sync and scene

- Subjects performed warehouse / logistics actions under beat instructions; clips were segmented with annotation tools after recording.
- D01 and D03 share the **same action window by annotation**, but frame counts `T` usually differ (median gap ~26 frames). Resample for shared-axis time-series models; the paper used GAF (not shipped here) so training did not need per-frame alignment.

More channel detail: [`metadata/channels.md`](metadata/channels.md). Joint names: [`metadata/joints.csv`](metadata/joints.csv).

---

## Signal overview

![Dataset overview: 20 MoCap joints, watch vs teacher signals, and eight stick-figure actions](assets/overview.png)

Orange = right-wrist watch site **J11**. Top: joint map + student watch ACC vs teacher J11 ACC/GYR. Bottom: one mid-clip pose per main class (schematic stick figures).

Regenerate: `python3 scripts/plot_overview.py`

---

## Snapshot

| | |
|--|--|
| Scene | Warehouse / logistics operations |
| Subjects | 9 anonymized ids `S0101`–`S1101` |
| Labels | **8** main classes; 25 subclass codes in the filename |
| MoCap (D03) | **Rokoko** suit → **UE4** virtual sensors: **20 joints × (POS, ACC, GYR)**, ~10 Hz |
| Watch (D01) | **Garmin Venu**, right wrist, **real ACC only**, ~10 Hz (paired with D03) |
| Raw paired clips | **1870** |
| Paper valid set | **1768** |
| Paper split | **cross-subject** train **1241** / test **527** |
| Invalid list | **102** clips in `splits/abnormal_clips.txt` |

---

## Layout

```
warehouse-moc-sensor/
  data/D01_watch_acc.zip       1870 npy  (~1.3 MB)
  data/D03_mocap_joints.zip    37400 npy (~58 MB)
  splits/train_clips.txt       1241
  splits/test_clips.txt         527
  splits/valid_clips.txt       1768
  splits/abnormal_clips.txt     102
  splits/abnormal_reasons.tsv
  metadata/
  assets/scene.jpg
  assets/overview.png
  assets/previews/
  scripts/load_clip.py
  scripts/plot_overview.py
```

---

## Files and channels

`A{class:02d}{subclass:02d}_S{subject:04d}_T{trial:03d}_D{01|03}_H01[_J{joint:02d}].npy`

| Zip | Inner files | Shape | Content |
|-----|-------------|-------|---------|
| `data/D01_watch_acc.zip` | `output/{clip}_D01_H01.npy` | `(T, 3)` int64 | watch accelerometer xyz |
| `data/D03_mocap_joints.zip` | `{clip}_D03_H01_Jxx.npy` | `(T, 9)` int64 | `0:3` ACC, `3:6` GYR, `6:9` POS |

D03 values were stored as `float * 100`. POS is UE4 world coordinates (Z up); hip = **J04**; watch site = **J11**.

Details: [`metadata/channels.md`](metadata/channels.md), [`metadata/joints.csv`](metadata/joints.csv), [`metadata/actions.csv`](metadata/actions.csv).

**D01 `T` ≠ D03 `T`** on almost every clip (median gap ~26 frames). Resample for time-series models.

```python
from scripts.load_clip import load_clip
d = load_clip("A0101_S0101_T008")
# d["watch_acc"] (T1, 3); d["mocap"] (T3, 20, 9); d["label8"] in 0..7
```

---

## Official split

Collection order, **first 6 subjects train / last 3 test**:

| Split | Subjects | N |
|-------|----------|--:|
| train | S0101, S0201, S0501, S0601, S0701, S0801 | 1241 |
| test | S0901, S1001, S1101 | 527 |

8-way counts on the **1768** valid set (`label = int(clip_id[1:3]) - 1`):

| Class | Name | Train | Test | Valid |
|------:|------|------:|-----:|------:|
| A01 | walking | 172 | 67 | 239 |
| A02 | picking | 208 | 86 | 294 |
| A03 | shelf | 56 | 30 | 86 |
| A04 | throwing | 180 | 88 | 268 |
| A05 | pointing | 119 | 56 | 175 |
| A06 | driving | 159 | 54 | 213 |
| A07 | idle / scan | 234 | 91 | 325 |
| A08 | packing | 113 | 55 | 168 |
| | **sum** | **1241** | **527** | **1768** |

---

## Invalid list (102 clips)

These ids are **inside the zips** (raw dump) but **must not** enter the paper protocol. They are exactly `1870 − 1768`. Full ids: [`splits/abnormal_clips.txt`](splits/abnormal_clips.txt). Per-clip reason: [`splits/abnormal_reasons.tsv`](splits/abnormal_reasons.tsv).

| Reason code | N | Meaning |
|-------------|--:|---------|
| `D01_empty` | 49 | watch array length 0 |
| `D01_short` | 35 | watch `0 < T < 40` |
| `signal_degenerate` | 10 | lengths look OK, QC waveforms empty or spike-only |
| `D03_short` | 6 | mocap `T < 40` |
| `D03_empty` | 1 | mocap length 0 |
| `both_short` | 1 | both streams `T < 40` |
| **sum** | **102** | |

S0901 accounts for 63 of 102, which is why the test set is smaller (137 vs ~200 raw).

Do **not** replace this list by a simple `T >= 40` filter (that would keep the 10 `signal_degenerate` clips).

---

## Citation

```bibtex
@article{Xie2026SensorToSensor,
  title   = {Sensor-to-Sensor procedural co-learning for sensor-limited human action recognition},
  author  = {Xie, Yulai and Fu, Yijia and Xu, Qing and Ren, Fang},
  journal = {Expert Systems with Applications},
  volume  = {319},
  pages   = {132094},
  year    = {2026},
  issn    = {0957-4174},
  doi     = {10.1016/j.eswa.2026.132094}
}
```

---

## License and ethics

Default file [`LICENSE`](LICENSE): **CC BY-NC 4.0** (confirm before public upload). Subjects are listed only as `Sxxxx`. This dump has **no video**.

---

## Acknowledgments

We thank **Mr. Zhang Yanfei (张燕飞)** and **Mr. Wang Xiaohui (王晓辉)** for their contributions to data collection and curation.

---

## Abbreviations

- **HAR** = Human Action Recognition
- **MoC / MoCap** = Motion Capture
- **IMU** = Inertial Measurement Unit
- **ACC / GYR / POS** = acceleration / angular velocity / position
- **GAF** = Gramian Angular Field (used in the paper, **not** shipped here)
- **D01 / D03** = watch / UE4 virtual-sensor device codes
