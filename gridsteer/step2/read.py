
import sys
import os
import json
from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument("dirname", type=str)
ap.add_argument("wellid", nargs=2, type=int)
ap.add_argument("--lattice", action="store_true",
                help="read mappings_on_lattice.json (all wells on the fitted lattice)")
ap.add_argument("--from-4corners", action="store_true",
                help="read mapping_from_4corners.json (interpolated from fourCorners' refined corners)")
args = ap.parse_args()

if args.from_4corners:
    json_name = "mapping_from_4corners.json"
elif args.lattice:
    json_name = "mappings_on_lattice.json"
else:
    json_name = "mapping.json"
json_f=os.path.join(args.dirname, "output_json_2", json_name)
a,b = args.wellid
wellKey = f"({a},{b})"

json_data = json.load(open(json_f, 'r'))
d = json_data["well_centering_positions"][wellKey]["motor_position"]
print(d['x'], d['y'], d['z'], d['phi'])

