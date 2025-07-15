from PIL import Image
from io import BytesIO
import numpy as np
import requests


def downsamp_PIL(img, scale):
    new_shape = img.size[0]//scale, img.size[1]//scale
    img = img.resize(new_shape, Image.LANCZOS)
    return img

def downsamp_numpy(img, scale):
    ydim,xdim = img.shape
    new_shape = ydim//scale, xdim//scale
    I = Image.fromarray(img)
    I = I.resize((new_shape[1], new_shape[0]), Image.LANCZOS)
    img = np.asarray(I)
    return img

def img_from_url(url, downsamp_scale=None, as_numpy=True):
    rq = requests.get(url)
    img = Image.open(BytesIO(rq.content)).convert("L")
    if downsamp_scale is not None:
        img = downsamp_PIL(img, downsamp_scale)
    if as_numpy:
        img = np.array(img.convert("RGB"))
    return img


