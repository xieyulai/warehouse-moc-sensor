# Channel and file format

## Capture equipment (paper)

| Code | Hardware | Body site | Modalities in this release | Rate |
|------|----------|-----------|----------------------------|------|
| **D01** | **Garmin Venu** smartwatch | right wrist (= **J11**) | ACC xyz only | ~10 Hz |
| **D03** | **Rokoko** professional MoCap suit → **UE4** virtual sensors | 20 full-body joints | POS + ACC + GYR per joint | ~10 Hz |

Pipeline for D03 (collection notes): Rokoko take → Unreal Engine 4 → virtual-sensor CSV → `.npy`. Exact Rokoko SKU is not named in the paper.

## Filenames

```
A{class:02d}{subclass:02d}_S{subject:04d}_T{trial:03d}_D{device}_H01[_J{joint:02d}].npy
```

Examples:

- Watch (inside `data/D01_watch_acc.zip`): `output/A0101_S0101_T008_D01_H01.npy`
- MoCap joint 11 (inside `data/D03_mocap_joints.zip`): `A0101_S0101_T008_D03_H01_J11.npy`

| Field | Meaning |
|-------|---------|
| `Axxxx` | action: first two digits = 8-way class (01–08), last two = subclass |
| `Sxxxx` | subject id (anonymized) |
| `Txxx` | repetition / trial |
| `D01` | real **Garmin Venu** watch on the **right wrist** |
| `D03` | virtual IMU/pose exported from **Rokoko** → **UE4** at **20 joints** |
| `H01` | wearing slot (only `H01` in this release) |
| `J01`–`J20` | joint index (D03 only). **J11 = right wrist** (same body site as the watch) |

## Arrays

### D01 watch — shape `(T, 3)`, dtype `int64`

Only accelerometer xyz. No gyroscope, no position.

Units are **not SI in the file**: values are integer-scaled (plots in the original QC pipeline often divide by 1000). Treat them as **arbitrary-unit acc** unless you recalibrate.

Paper sampling rate: **10 Hz**. Median `T` on the official 1768 set is 76 (~7.6 s if exactly 10 Hz). Paper mean duration is ~9 s; D01 and D03 lengths **do not match** clip-by-clip.

### D03 MoCap virtual sensors — shape `(T, 9)`, dtype `int64`

Saved as `raw_float * 100` then cast to int64. Columns after the UE4 export remap:

| columns | name | notes |
|---------|------|--------|
| `0:3` | ACC xyz | export swapped xy and negated x/z relative to the UE4 CSV |
| `3:6` | GYR xyz | angular velocity |
| `6:9` | POS xyz | world frame; **Z up**; origin is the UE4 world origin (not the body). Hip = J04 |

Paper sampling rate: **10 Hz**. Median `T` on the official 1768 set is 102.

Official training in the paper converted each 3-axis stream to a GAF image. **This release ships raw npy only** (no GAF).

## Alignment

### Dataset pairing (this release)

- Same `clip_id` ⇒ same annotated take (beat instructions + manual cut on synchronized video; paper §5.1.3).
- `T` of D01 and D03 almost never match (median gap ~26 frames). Paired by **action window**, not by sample index.
- Paper training used min-max to [-1, 1] then **GAF → 224×224** per stream (no per-frame D01↔D03 resample). **GAF not shipped here.**
- Time-series users should resample / crop themselves.

### Model multi-level alignment (paper §4; training only)

GAF input unification → pre-adaptive input alignment → identical backbone → MSE+LSCA feature alignment → CE (+ optional KD) output alignment. Not additional files in this repo.

## Bones (1-based joint ids)

```
(1-2)(2-3)(3-4)(2-5)(5-6)(6-7)(7-8)(2-9)(9-10)(10-11)(11-12)
(4-13)(13-14)(14-15)(15-16)(4-17)(17-18)(18-19)(19-20)
```
