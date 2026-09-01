"""Write four_corners.json from command-line arguments.

Usage (called by fourCorners.tcl):
    python -m gridsteer.step2.write_four_corners <outfile> \
        "A 2 1 x y z phi" "B 1 1 x y z phi" "C 1 9 x y z phi" "D 1 10 x y z phi"
"""

import json
import sys
import os

outfile = sys.argv[1]
entries = sys.argv[2:]

corners = {}
for entry in entries:
    parts = entry.split()
    name = parts[0]
    row, col = int(parts[1]), int(parts[2])
    x, y, z, phi = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
    corners[f"({row},{col})"] = {
        "name": name,
        "row": row,
        "column": col,
        "motor_position": {"x": x, "y": y, "z": z, "phi": phi},
    }

os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, "w") as f:
    json.dump({"four_corners": corners}, f, indent=2)
