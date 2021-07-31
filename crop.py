import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from copy import deepcopy


def main(project, fname, nx, ny):
    path = f"images/{project}/{fname}"
    outdir = f"images/{project}/cropped"
    img_in = mpimg.imread(path)
    img_in = deepcopy(img_in.astype(float)) / 255.0
    (x_size, y_size, color_channels) = img_in.shape
    x_chunk_size = x_size / nx
    y_chunk_size = y_size / ny
    for i in nx:
        for j in ny:
            image_chunk = img_in[
                int(round(i * x_chunk_size)) : int(round((i + 1) * x_chunk_size)),
                int(round(j * y_chunk_size)) : int(round((j + 1) * y_chunk_size)),
            ]
            out_name = f"{outdir}/{i}_{j}.png"
            plt.imsave(out_name, image_chunk)


if __name__ == "__main__":
    project = "birch_warm"
    fname = "out_[3, 3, 1]5_final.png"
    nx = 13
    ny = 8
    main(project, fname, nx, ny)
