
import argparse

parser = argparse.ArgumentParser(description='record motor values and hutch camera')
parser.add_argument('blname', 
    type=str, help="beamline name: 122, 71, 121, 92, or 141", choices=["122","121","71","92","141"])
parser.add_argument('dirname', type=str, help='subfolder name within /data/blctl/grid_scans)')
parser.add_argument('x', type=float, help='Value for x')
parser.add_argument('y', type=float, help='Value for y')
parser.add_argument('z', type=float, help='Value for z')
parser.add_argument('phi', type=float, help='Value for phi')
parser.add_argument('dy', type=float, help='Value for dy')
parser.add_argument('count', type=float, help='Value for count')
parser.add_argument('-o', '--offaxis', type=str, help="camera url for the off-axis snapshot", default=None)
parser.add_argument('-i', '--inline', type=str, help="camera url for the inline-axis snapshot", default=None)

args = parser.parse_args()
x = args.x
y = args.y
z = args.z
phi = args.phi
dy = args.dy
count = args.count


import numpy as np
from PIL import Image
from io import BytesIO
import requests
import time
import os

URLS={
  'video.snapshotDirectUrl': f'http://bl{args.blname}campcp/image?mac=CAM1',
  'video.snapshotOrigDirectUrl': f'http://bl{args.blname}campcp/rawImage?mac=CAM1',
  'video.snapshotDirectInlineUrl': f'http://bl{args.blname}campcp/image?mac=CAM2',
  'video.snapshotOrigInlineDirectUrl': f'http://bl{args.blname}campcp/rawImage?mac=CAM2'
}

def img_from_url(url):
    t = time.time()
    rq = requests.get(url)
    turl = time.time()-t
    img = Image.open(BytesIO(rq.content)).convert("L")
    img = np.asarray(img)
    timg = time.time()-t
    print(f"Took {turl:.3f} sec for url and {timg-turl:.3f} sec for bytes to {img.shape} img conversion...")
    return img

if args.offaxis is not None:
    cam1_url = args.offaxis
else:
    cam1_url=URLS['video.snapshotDirectUrl']
img1 = img_from_url(cam1_url)

if args.inline is not None:
    cam2_url =args.inline
else:
    cam2_url=URLS['video.snapshotDirectInlineUrl']
img2 = img_from_url(cam2_url)

print("scan %d:"%count, x,y,z,phi, dy)

tnp = time.time()
os.makedirs(args.dirname, exist_ok=True)
np.savez('%s/test%d'%(args.dirname,count),
    sample=img1, inline=img2, x=x,y=y,z=z,phi=phi, dy=dy, count=count)
tnp = time.time()-tnp
print(f"numpy save in {tnp*1000:.2f} msec")

