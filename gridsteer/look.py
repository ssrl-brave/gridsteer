import glob
import os
import numpy as np # Explicitly import numpy
import matplotlib.pyplot as plt # Use plt instead of pylab for clarity

# Global variable to track the current image index
current_index = 0
fnames = []
names = 'x', 'y', 'z', 'phi'
fig = None
ax1 = None
ax2 = None

def update_plot(index):
    global fig, ax1, ax2

    if not fnames or index < 0 or index >= len(fnames):
        return

    f = fnames[index]
    title = os.path.basename(f)
    d = np.load(f)

    for val in names:
        title += f"; {val}={d[val][()]:.2f}"

    img_samp = d['sample'][()]
    img_inline = d['inline'][()]
    
    if not ax1.images:
        ax1.imshow(img_samp, cmap="cividis")
        ax2.imshow(img_inline, cmap='cividis')
    else:
        ax1.images[0].set_data(img_samp)
        ax1.images[0].set_clim(img_samp.min(), img_samp.max())
        ax2.images[0].set_data(img_inline)
        ax2.images[0].set_clim(img_inline.min(), img_inline.max())

    fig.suptitle(title)
    fig.canvas.draw_idle()


def key_press(event):
    global current_index

    if event.key == 'right':
        # Cycle forward
        new_index = (current_index + 1) % len(fnames)
        if new_index != current_index:
            current_index = new_index
            update_plot(current_index)

    elif event.key == 'left':
        # Cycle backward
        new_index = (current_index - 1 + len(fnames)) % len(fnames)
        if new_index != current_index:
            current_index = new_index
            update_plot(current_index)


def main():
    global fnames, fig, ax1, ax2, current_index
    
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("dirname", type=str, help="Directory containing test*npz files.")
    args = ap.parse_args()

    # --- File Loading and Sorting ---
    fnames = glob.glob(f"{args.dirname}/test*npz")
    try:
        fnames = sorted(fnames, key=lambda x: int(os.path.basename(x).split("test")[1].split(".")[0]))
    except IndexError:
        print("Error: Ensure files follow the pattern 'test<number>.npz'.")
        return
        
    if not fnames:
        print(f"No files found matching '{args.dirname}/test*npz'")
        return

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=.93, wspace=0)
    fig.set_size_inches(10.7,3.9)
    
    ax1.axis('off')
    ax2.axis('off')

    current_index = 0
    update_plot(current_index)

    fig.canvas.mpl_connect('key_press_event', key_press)

    print(f"Loaded {len(fnames)} images. Use the LEFT and RIGHT arrow keys to navigate.")
    plt.show() # Use plt.show() to start the interactive loop

if __name__=="__main__":
    main()
