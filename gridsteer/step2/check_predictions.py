#!/usr/bin/env python3
"""
Check predicted well-centering positions against the recorded frames.

Reads well_centering_positions.json (produced by map_wells.py) and,
for each well's predicted motor position, finds the frame whose recorded
stage coordinates (x, y, z) are closest. A well predicted correctly
should appear near the center of its closest frame -- if that frame was
taken near the predicted position at all (see the distance in the title).

Usage:
    python scripts/check_predictions.py \
        output_tracks/mapper15/well_centering_positions.json data/mapper15

Outputs (next to the input JSON by default):
    closest_frames.png    montage: each well's closest frame, with a
                          center crosshair and the motor distance

Dependencies: numpy, matplotlib.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def numeric_key(path: Path):
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def load_frame_positions(data_dir: Path, key: str = "sample"):
    """Load each frame's image and recorded motor position (x, y, z)."""
    files = sorted(data_dir.glob("*.npz"), key=numeric_key)
    if not files:
        raise FileNotFoundError(f"no .npz files in {data_dir}")
    frames, positions = [], []
    for f in files:
        d = np.load(f)
        frames.append(np.asarray(d[key]))
        positions.append([float(np.atleast_1d(d[k])[0]) for k in ("x", "y", "z")])
    return frames, np.array(positions), [f.name for f in files]


def closest_frames(wells, positions):
    """For each well, the index of and distance to the nearest frame.

    Distance is euclidean over the (x, y, z) motor coordinates between
    the well's predicted centering position and each frame's recorded
    stage position.
    """
    out = {}
    for label, w in wells.items():
        p = w["motor_position"]
        target = np.array([p["x"], p["y"], p["z"]])
        dists = np.linalg.norm(positions - target, axis=1)
        i = int(np.argmin(dists))
        out[label] = (i, float(dists[i]))
    return out


def save_montage(wells, matches, frames, names, path):
    """One panel per well: its closest frame with a center crosshair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(wells, key=lambda k: (wells[k]["row"], wells[k]["column"]))
    ncol = min(5, len(order))
    nrow = int(np.ceil(len(order) / ncol))
    h, w = frames[0].shape[:2]
    # 0.8in per row of headroom for the two-line panel titles
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(3.2 * ncol, (3.2 * h / w + 0.8) * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, order):
        well = wells[key]
        i, dist = matches[key]
        ax.imshow(frames[i], cmap="gray")
        ax.axhline(h / 2, color="cyan", lw=0.8, alpha=0.7)
        ax.axvline(w / 2, color="cyan", lw=0.8, alpha=0.7)
        tag = " (Predicted)" if well["predicted"] else ""
        ax.set_title(f"{well['label']}{tag}\n{names[i]}  d={dist:.3f}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(order):]:
        ax.axis("off")
    fig.suptitle("Closest Recorded Frame to Each Well's Predicted Centering Position",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path", type=Path,
                    help="well_centering_positions.json from map_wells.py")
    ap.add_argument("data_dir", type=Path, help="directory of .npz frames")
    ap.add_argument("--key", default="sample", help="npz key holding the image")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: alongside the JSON)")
    args = ap.parse_args()

    out = args.out or args.json_path.parent
    out.mkdir(parents=True, exist_ok=True)

    with open(args.json_path) as f:
        wells = json.load(f)["well_centering_positions"]
    frames, positions, names = load_frame_positions(args.data_dir, args.key)
    print(f"{len(wells)} Wells vs {len(frames)} Recorded Frames")

    matches = closest_frames(wells, positions)

    for key in sorted(wells, key=lambda k: (wells[k]["row"], wells[k]["column"])):
        well = wells[key]
        i, dist = matches[key]
        tag = "Predicted" if well["predicted"] else "Detected "
        print(f"  {well['label']:>6} ({tag})  ->  {names[i]:<12} d={dist:.3f}")

    save_montage(wells, matches, frames, names, out / "closest_frames.png")
    print(f"Wrote {out / 'closest_frames.png'}")


if __name__ == "__main__":
    main()
