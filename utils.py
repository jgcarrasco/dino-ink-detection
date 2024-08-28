from glob import glob

import numpy as np

import torch
import torchvision.transforms as T
from torch.amp import autocast

from sklearn.decomposition import PCA

import cv2
from tiffile import imread

import matplotlib.pyplot as plt
from matplotlib import patches
from IPython.display import display, clear_output
import ipywidgets as widgets
from ipywidgets import interact

def load_segment(segment_id):
    tif = imread(sorted(glob(f"segments/{segment_id}/layers/*.tif")))
    # convert to uint8
    if tif.dtype == "uint16":
        tif = (tif * 255.0/65535.0).astype("uint8")
    inklabels = cv2.imread(f"segments/{segment_id}/{segment_id}_inklabels.png")
    inklabels = cv2.cvtColor(inklabels,cv2.COLOR_RGB2GRAY)
    return tif, inklabels

def visualize_cropper(tif, inklabels, crop_size):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    tif_display = ax1.imshow(tif[0], cmap="gray")
    ax2.imshow(inklabels, cmap="gray")
    rect = patches.Rectangle((0, 0), crop_size, crop_size, linewidth=1, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((0, 0), crop_size, crop_size, linewidth=1, edgecolor='r', facecolor='none')

    ax1.add_patch(rect); ax2.add_patch(rect2)
    ax1.axis("off"); ax2.axis("off")
    plt.tight_layout()

    def on_press(event):
        if event.inaxes != ax1:
            return
        x1, y1 = event.xdata, event.ydata
        rect.set_xy((x1 - crop_size / 2, y1 - crop_size / 2))
        rect2.set_xy((x1 - crop_size / 2, y1 - crop_size / 2))
        fig.canvas.draw()

    def on_key(event):
        if event.key == 'enter':
            x0, y0 = rect.get_xy()
            x0, y0 = int(x0), int(y0)
            cropped_tif = tif[:, y0:y0+crop_size, x0:x0+crop_size]
            cropped_inklabels = inklabels[y0:y0+crop_size, x0:x0+crop_size]
            plt.close()
            clear_output()
            print("Subimage saved!")
            result["cropped_tif"] = cropped_tif
            result["cropped_inklabels"] = cropped_inklabels

    def update_image(slice_idx):
        tif_display.set_data(tif[slice_idx])
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('key_press_event', on_key);

    # Create a slider to navigate through the images
    slider = widgets.IntSlider(min=0, max=64, step=1, description='Slice')
    # Link the slider to the update_image function
    interact(update_image, slice_idx=slider);

    result = {"cropped_tif": None, "cropped_inklabels": None}

    plt.show()

    return result

def prepare_tif(tif):
    tif_grayscale = []
    for i in range(tif.shape[0]):
        tif_grayscale.append(cv2.cvtColor(tif[i],cv2.COLOR_GRAY2RGB))
    tif_grayscale = torch.tensor(np.stack(tif_grayscale, axis=0) / 255, dtype=torch.float32)
    tif_grayscale = tif_grayscale.permute(0, 3, 1, 2)
    tif_grayscale = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(tif_grayscale)
    return tif_grayscale

def compute_PCA(tif, model, pca_depth, n_components=3):  
    tif_grayscale = prepare_tif(tif)

    if torch.cuda.is_available():
        with autocast("cuda"):
            output = torch.cat([model.get_intermediate_layers(tif_slice[None, ...].cuda(), 1, reshape=True)[0].detach().cpu() for tif_slice in tif_grayscale])
    else:
        output = torch.cat([model.get_intermediate_layers(tif_slice[None, ...], 1, reshape=True)[0].detach() for tif_slice in tif_grayscale])

    y = output.permute(0, 2, 3, 1)

    if pca_depth:
        pca = PCA(n_components=n_components)
        y_pca = pca.fit_transform(y.reshape(-1, y.shape[-1])).reshape(*y.shape[:-1], -1)
    else:
        y_pca = []
        for yi in y:
            pca = PCA(n_components=n_components)
            yi_pca = pca.fit_transform(yi.reshape(-1, y.shape[-1])).reshape(*yi.shape[:-1], -1)
            y_pca.append(yi_pca)
        y_pca = np.stack(y_pca)
    return y_pca

def visualize_PCA(y_pca, cropped_tif, cropped_inklabels):
    n_components = y_pca.shape[-1]
    def plot_slices(slice_index):
        fig, axes = plt.subplots(1, n_components+2, figsize=(20, 4))
        for i in range(n_components):
            ax = axes[i]
            ax.imshow(y_pca[slice_index, :, :, i], cmap='gray')
            ax.set_title(f'Slice {slice_index}, PCA component {i+1}')
            ax.axis('off')
        axes[-2].imshow(cropped_tif[slice_index], cmap="gray")
        axes[-1].imshow(cropped_inklabels, cmap="gray")
        axes[-2].axis("off"); axes[-1].axis("off")
        axes[-2].set_title("tif"); axes[-1].set_title("GP Prediction")
        plt.tight_layout()
        plt.show()

    # Create an interactive slider
    slider = widgets.IntSlider(value=36, min=0, max=y_pca.shape[0] - 1, step=1, description='Slice Index')

    # Use the interact function to create an interactive plot
    interact(plot_slices, slice_index=slider)