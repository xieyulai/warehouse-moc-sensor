#!/usr/bin/env python3
"""Load one Warehouse MoC+Sensor clip from the two release zips."""
from __future__ import annotations

import zipfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D01_ZIP = ROOT / "data" / "D01_watch_acc.zip"
D03_ZIP = ROOT / "data" / "D03_mocap_joints.zip"


@lru_cache(maxsize=2)
def _zip(path: Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def _npy(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    if name in zf.namelist():
        inner = name
    elif f"output/{name}" in zf.namelist():
        inner = f"output/{name}"
    else:
        raise FileNotFoundError(f"{name} not in {zf.filename}")
    return np.load(BytesIO(zf.read(inner)))


def load_clip(clip_id: str):
    """Return watch acc (T1,3) and mocap (T3,20,9) = ACC|GYR|POS.

    clip_id example: A0101_S0101_T008
    """
    z1, z3 = _zip(D01_ZIP), _zip(D03_ZIP)
    watch = _npy(z1, f"{clip_id}_D01_H01.npy")
    joints = np.stack(
        [_npy(z3, f"{clip_id}_D03_H01_J{j:02d}.npy") for j in range(1, 21)],
        axis=1,
    )
    return {
        "clip_id": clip_id,
        "label8": int(clip_id[1:3]) - 1,
        "subclass": clip_id[:5],
        "subject": clip_id.split("_")[1],
        "watch_acc": watch.astype(np.float64),
        "mocap": joints.astype(np.float64),
        "mocap_acc": joints[:, :, 0:3].astype(np.float64),
        "mocap_gyr": joints[:, :, 3:6].astype(np.float64),
        "mocap_pos": joints[:, :, 6:9].astype(np.float64),
    }


def read_split(name: str) -> list[str]:
    path = ROOT / "splits" / name
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("clip_id", nargs="?", default="A0101_S0101_T008")
    args = p.parse_args()
    d = load_clip(args.clip_id)
    print(
        f"{d['clip_id']}  label8={d['label8']}  "
        f"watch{d['watch_acc'].shape}  mocap{d['mocap'].shape}"
    )
