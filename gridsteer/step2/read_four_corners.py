"""Print the corners stored in four_corners.json, one per line, in corner order.

Usage (called by fourCorners.tcl --from-json):
    python -m gridsteer.step2.read_four_corners <four_corners.json>

Output lines: "<name> <row> <col> <x> <y> <z> <phi>", sorted by name (A, B, C, D).
"""

import json
import sys

json_f = sys.argv[1]
corners = json.load(open(json_f, "r"))["four_corners"]
if len(corners) != 4:
    sys.exit(f"expected 4 corners in {json_f}, found {len(corners)}")

for c in sorted(corners.values(), key=lambda c: c["name"]):
    m = c["motor_position"]
    print(c["name"], c["row"], c["column"], m["x"], m["y"], m["z"], m["phi"])
