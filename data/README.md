# data/

Two zip archives of raw NumPy clips (no GAF). Load with `scripts/load_clip.py` without unzipping.

| file | compressed | inner npy | meaning |
|------|-----------:|----------:|---------|
| `D01_watch_acc.zip` | ~1.3 MB | 1870 | Garmin watch acc; paths `output/*.npy` |
| `D03_mocap_joints.zip` | ~58 MB | 37400 | MoCap 20 joints; paths `{clip}_D03_H01_Jxx.npy` |

Pairing key: `Axxxx_Sxxxx_Txxx`. Extracted on disk these would be ~300 MB; the zips are the release format.

See `../metadata/channels.md` and `../splits/` for the official 1768-clip protocol.
