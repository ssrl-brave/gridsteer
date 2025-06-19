
import numpy as np
from scipy.signal import argrelmax
from PIL import Image
from scipy.ndimage import gaussian_filter

import pylab as plt

from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny

def find_edge_pix(grad, pk_width=100, order=20):
    nbin=50
    pk = np.argmax(grad.mean(axis=0))
    grads = np.array_split(grad,nbin)
    n = grad.shape[0]
    grads_pos = [vals.mean() for vals in np.array_split(np.arange(n),nbin)]

    x,y = [],[]
    slc = slice(pk-pk_width//2,pk+pk_width//2,1)
    for i_g, g in enumerate(grads):
        sig = np.mean(g[:,slc], axis=0)
        mx_pos = argrelmax(sig, order=order)[0]
        if mx_pos.size==0:
            continue
        mx = 0
        best_pos = 0
        for mxp in mx_pos:
            if sig[mxp] > mx:
                mx = sig[mxp]
                best_pos = mxp
        x.append(best_pos+pk-pk_width//2)
        y.append(grads_pos[i_g])
    return x, y

def downsamp(img, scale):
    ydim,xdim = img.shape
    new_shape = ydim//scale, xdim//scale
    I = Image.fromarray(img)
    I = I.resize((new_shape[1], new_shape[0]), Image.LANCZOS)
    img = np.asarray(I)
    return img

def proc_img(img, scale=6 ):
    img = downsamp(img)
    grad0, grad1 = np.gradient( gaussian_filter(img, sigma=6))
    y,x = find_edge_pix(grad0.T)
    x2,y2 = find_edge_pix(grad1)
    return img, (x,y), (x2,y2)

def find_lines(img, scale=6, sigma=5, low=20, high=40, ax=None):
    img = downsamp(img, scale)
    edge=canny(img, sigma=sigma, low_threshold=low, high_threshold=high)
    angs = np.linspace(-np.pi/2, np.pi/2, 500)
    h,theta, d = hough_line(edge, angs)
    ph,pang,pdist=hough_line_peaks(h, theta,d,threshold=np.percentile(h,99.9), min_distance=20, min_angle=80,num_peaks=4)
    lines = []
    xline = np.arange(edge.shape[1]) 
    for ang, dist in zip(pang, pdist):
        x0,y0 = dist*np.cos(ang), dist*np.sin(ang)
        m = y0/x0
        m2 = -1/m
        yline = m2* (xline-x0) + y0
        sel = np.logical_and(yline > 0 , yline < edge.shape[0])
        if ax is not None:
            ax.plot( xline[sel], yline[sel], color='r--')
        lines.append( (xline[sel], yline[sel]))
    return img, ph, lines


def find_circles( img, scale=6, target_rad=107, ax=None, view=3, sigma=5, low=20, high=40 ):
    img = downsamp(img, scale)
    edge = canny(img, sigma=sigma, low_threshold=low, high_threshold=high)
    rads = np.arange(target_rad-view,target_rad+view+1)
    out = hough_circle(edge,rads)
    accum, cx, cy, rad = hough_circle_peaks( out, rads, min_xdistance=90, min_ydistance=90)
    for a,x,y,r in zip(accum/accum.max(), cx,cy,rad):
        if ax is not None:
            ax.add_patch(Circle(xy=(x,y), radius=r, ec='r', fc='none', ls='--', alpha=a) )
    circles = accum, cx, cy, rad
    return img, circles 


def main():
    all_circ_acc, all_line_acc=[],[]
    for i in range(0,40):
        img = np.load(f"test_images/img{i}.h5.npz")["img"]
        ds_img, circs = find_circles(img, view=4)
        _, line_acc, lines = find_lines(img)
        all_line_acc += list(line_acc)
        if not plt.gca().images:
            plt.imshow(ds_img, cmap="gray")
            xl=plt.gca().get_xlim()
            yl = plt.gca().get_ylim()
        else:
            plt.gca().images[0].set_data(ds_img)
        while plt.gca().lines:
            plt.gca().lines[0].remove()
        if lines:
          for a,(x,y) in zip( line_acc/max(all_line_acc), lines):
            plt.plot(x,y,'b', alpha=a, lw=5, ls='--')
        while plt.gca().patches:
            plt.gca().patches[0].remove()
        accum, cx,cy,rad = circs
        all_circ_acc += list(accum)
        for a,x,y,r in zip(accum/max(all_circ_acc), cx,cy,rad):
            plt.gca().add_patch(plt.Circle(xy=(x,y), radius=r, ec='tomato', fc='none', ls='--', alpha=a, lw=4) )
        plt.title(str(i))
        plt.gca().set_xlim(xl)
        plt.gca().set_ylim(yl)
        plt.draw()
        plt.pause(.75)
        plt.savefig(f"test_images/circs_and_lines{i}.png", dpi=150)
        

if __name__=="__main__":
    main()
