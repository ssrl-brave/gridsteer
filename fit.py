
import numpy as np
from scipy.signal import argrelmax
from PIL import Image
from scipy.ndimage import gaussian_filter

import pylab as plt

from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny

import matplotlib.animation as animation


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
    # img = downsamp(img)
    grad0, grad1 = np.gradient( gaussian_filter(img, sigma=6))
    y,x = find_edge_pix(grad0.T)
    x2,y2 = find_edge_pix(grad1)
    return img, (x,y), (x2,y2)


def find_lines(img, sigma=9.5, low=0.1, high=0.95, ax=None, scale=6):
    # img = downsamp(img, scale)
    edge = canny(img, sigma=sigma, low_threshold=low, high_threshold=high, use_quantiles=True)
    angs = np.linspace(-np.pi/2, np.pi/2, 360, endpoint=False)
    h, theta, d = hough_line(edge, angs)
    ph, pang, pdist=hough_line_peaks(h, theta, d, threshold=70, min_distance=20, min_angle=80, num_peaks=4)
    # print(ph, pang, pdist)
    lines = []
    xline = np.arange(edge.shape[1]) 
    for ang, dist in zip(pang, pdist):
        x0,y0 = dist*np.cos(ang), dist*np.sin(ang)
        if x0 == 0 or y0 == 0:
            continue
        m = y0/x0
        m2 = -1/m
        yline = m2 * (xline-x0) + y0
        sel = np.logical_and(yline > 0 , yline < edge.shape[0])
        if ax is not None:
            ax.plot( xline[sel], yline[sel], color='r--')
        lines.append( (xline[sel], yline[sel]))
    return img, ph, lines


def find_circles(img, target_rad=86, ax=None, view=3, sigma=4.5, low=0.1, high=0.95, scale=6, threshold=0.2):
    # img = downsamp(img, scale)
    edge = canny(img, sigma=sigma, low_threshold=low, high_threshold=high, use_quantiles=True)
    rads = np.arange(target_rad-view,target_rad+view+1)
    out = hough_circle(edge,rads)
    accum, cx, cy, radii = hough_circle_peaks(out, rads, min_xdistance=150, min_ydistance=150, num_peaks=4, threshold=0.2)
    if ax is not None:
        for a,x,y,r in zip(accum, cx, cy, radii):
            ax.add_patch(Circle(xy=(x,y), radius=r, ec='r', fc='none', ls='--', alpha=a) )
    circles = accum, cx, cy, radii
    return img, circles 


def main():
    for i in range(0, 1000):
        img = np.load(f"../grid_center_db.2/test{i}.npz")["sample"]
        ds_img, circs = find_circles(img)
        ds_img, calibrationPoints = find_circles(img, target_rad=18, view=3, threshold=0.3)
        _, line_acc, lines = find_lines(img)
        if not plt.gca().images:
            plt.imshow(ds_img, cmap="gray")
            xl=plt.gca().get_xlim()
            yl = plt.gca().get_ylim()
        else:
            plt.gca().images[0].set_data(ds_img)
        while plt.gca().lines:
            plt.gca().lines[0].remove()
        if lines:
          for a,(x,y) in zip(line_acc/max(line_acc), lines):
            plt.plot(x, y, 'b', alpha=a, lw=5, ls='--')
        while plt.gca().patches:
            plt.gca().patches[0].remove()
        accum, cx, cy, radii = circs
        for a,x,y,r in zip(accum, cx, cy, radii):
            plt.gca().add_patch(plt.Circle(xy=(x,y), radius=r, ec='tomato', fc='none', ls='--', alpha=0.8, lw=4))
        # accum, cx, cy, radii = calibrationPoints
        # for a,x,y,r in zip(accum, cx, cy, radii):
        #     plt.gca().add_patch(plt.Circle(xy=(x,y), radius=r, ec='green', fc='none', ls='--', alpha=0.8, lw=4) )
        plt.title(str(i))
        plt.gca().set_xlim(xl)
        plt.gca().set_ylim(yl)
        plt.draw()
        plt.pause(0.05)
        plt.savefig(f"test_images/circs_and_lines{i}.png", dpi=150)



def make_movie():
    sample = []
    inline = []
    # xyzphi = []
    for i in range(0, 1000):
        sample.append(np.load(f"../grid_center_db.2/test{i}.npz")["sample"])
        inline.append(np.load(f"../grid_center_db.2/test{i}.npz")["inline"])
        # xyzphi.append(np.load(f"../grid_center_db.2/test{i}.npz")["xyzphi"])

    plt.close("all")
    fps = 30
    fig, (ax1, ax2) = plt.subplots(1, 2)

    im1 = ax1.imshow(sample[0], cmap='gray', animated=True)
    im2 = ax2.imshow(inline[0], cmap='gray', animated=True)
    def animate_func(i):
        if i % fps == 0:
            print( '.', end='' )

        im1.set_array(255*canny(sample[i], sigma=4.5, low_threshold=0.1, high_threshold=0.95, use_quantiles=True).astype("uint8"))
        im2.set_array(255*canny(sample[i], sigma=9.5, low_threshold=0.1, high_threshold=0.95, use_quantiles=True).astype("uint8"))
        # x = xyzphi[i, 0]
        # y = xyzphi[i, 1]
        # z = xyzphi[i, 2]
        # phi = xyzphi[i, 3]
        # fig.suptitle(f'x={x:6.2f} y={y:6.2f} z={z:6.2f} phi={phi:6.2f}', fontsize=12, color='blue')
        return im1, im2

    anim = animation.FuncAnimation(fig, 
                               animate_func, 
                               frames=1000,
                               interval=1000 / fps, # in ms
    )
    writer = animation.FFMpegWriter(fps=fps, codec="h264")
    anim.save('test_anim.mp4', writer=writer)

if __name__=="__main__":
    make_movie()
