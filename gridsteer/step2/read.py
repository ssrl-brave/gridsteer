
import sys
import os
import json
from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument("dirname", type=str)
ap.add_argument("wellid", nargs=2, type=int)
ap.add_argument("--lattice", action="store_true",
                help="read mappings_on_lattice.json (all wells on the fitted lattice)")
args = ap.parse_args()

json_name = "mappings_on_lattice.json" if args.lattice else "mapping.json"
json_f=os.path.join(args.dirname, "output_json_2", json_name)
a,b = args.wellid
wellKey = f"({a},{b})"

json_data = json.load(open(json_f, 'r'))
d = json_data["well_centering_positions"][wellKey]["motor_position"]
print(d['x'], d['y'], d['z'], d['phi'])

