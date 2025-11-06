
import sys
import os
import json
from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument("dirname", type=str)
ap.add_argument("wellid", nargs=2, type=int)
args = ap.parse_args()

json_f=os.path.join(args.dirname, "output_json", "mapping.json")
a,b = args.wellid
wellKey = f"({a},{b})"

json_data = json.load(open(json_f, 'r'))
d = json_data["well_centering_positions"][wellKey]["motor_position"]
print(d['x'], d['y'], d['z'], d['phi'])

