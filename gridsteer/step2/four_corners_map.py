"""Build a full well map from the four refined corners.

Usage (called by fourCorners.tcl):
    python -m gridsteer.step2.four_corners_map <output_json_2 dir>

Reads four_corners.json (corners (2,1), (1,1), (1,9), (2,10), refined and
focused by fourCorners) and writes mapping_from_4corners.json in the
mapping.json format. Each row's wells are evenly spaced between that row's
two corners, linearly in x, y and z -- the same equal-pitch model as the
fitted lattice and as DCSS add_4PointGrid -- so the corners' focus carries
across the tray (tilt included).
"""

import json
import os
import sys

# row -> (first column, last column); R1 is the 9-well row, R2 the 10-well row
ROWS = {1: (1, 9), 2: (1, 10)}

out_dir = sys.argv[1]
corners = json.load(open(os.path.join(out_dir, "four_corners.json")))["four_corners"]


def corner(row, col):
    key = f"({row},{col})"
    if key not in corners:
        sys.exit(f"four_corners.json has no corner {key}")
    return corners[key]["motor_position"]


positions = {}
for row, (c0, c1) in ROWS.items():
    a, b = corner(row, c0), corner(row, c1)
    for col in range(c0, c1 + 1):
        f = (col - c0) / (c1 - c0)
        mp = {k: a[k] + f * (b[k] - a[k]) for k in ("x", "y", "z")}
        mp["phi"] = a["phi"] + f * (b["phi"] - a["phi"])
        positions[f"({row},{col})"] = {
            "label": f"R{row}C{col}",
            "row": row,
            "column": col,
            "from_corner": col in (c0, c1),
            "motor_position": {k: round(v, 6) for k, v in mp.items()},
        }

out = {
    "positions": "interpolated per row between the four_corners.json corners",
    "source": "four_corners.json",
    "well_centering_positions": dict(sorted(
        positions.items(), key=lambda kv: (kv[1]["row"], kv[1]["column"]))),
}
path = os.path.join(out_dir, "mapping_from_4corners.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(path)
