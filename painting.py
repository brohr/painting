import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from copy import deepcopy
from sklearn.cluster import KMeans
import numpy as np
import torch
import colorsys
from custom_kmeans import MultiChannelKMeans, ImageKMeans, get_one_hot
import json


def blur(input_matrix, conv_object):
    input_matrix = torch.from_numpy(input_matrix).type(torch.FloatTensor).unsqueeze(0).unsqueeze(1)
    out = conv_object(input_matrix)
    out = out.squeeze(1).squeeze(0).detach().cpu().numpy()
    return out


def blur_rgb(rgb_matrix, conv_object):
    R = blur(rgb_matrix[:,:,0], conv_object)
    G = blur(rgb_matrix[:,:,1], conv_object)
    B = blur(rgb_matrix[:,:,2], conv_object)

    out = np.zeros((R.shape[0], R.shape[1], 3))
    out[:,:,0] = R
    out[:,:,1] = G
    out[:,:,2] = B

    return out

def json_to_np(json_path):
    list_of_lists = json.load(open(json_path,'r'))
    arr = np.array([np.array(item) for item in list_of_lists])
    arr = arr.astype(np.float64)
    if np.max(arr) > 1.:
        arr = arr / 255.
    return arr

def change_color_levels(cluster_centers, magnitude=0.):
    brightness_cutoff = np.median(np.sum(cluster_centers, axis=1))
    for i in range(len(cluster_centers)):
        if np.sum(cluster_centers[i]) > brightness_cutoff:
            brightest_color_magnitude = max(cluster_centers[i])
            max_movement = 1 - brightest_color_magnitude
            for j in range(3):
                cluster_centers[i][j] += max_movement * levels
        else:
            darkest_color_magnitude = min(cluster_centers[i])
            max_movement = darkest_color_magnitude
            for j in range(3):
                cluster_centers[i][j] -= max_movement * levels

        # if np.random.rand() > 0.5:
        #     cluster_centers[i] = np.array([0.,0.,0.])
        # else:
        #     cluster_centers[i] = np.array([1.,1.,1.])

    return cluster_centers


def change_saturation_lightness(cluster_centers, saturation=0., lightness=0.):
    for i in range(cluster_centers.shape[0]):
        r = cluster_centers[i][0]
        g = cluster_centers[i][1]
        b = cluster_centers[i][2]
        (h,l,s) = colorsys.rgb_to_hls(r, g, b)
        s += (1-s) * saturation
        l += (1-l) * lightness
        (r,g,b) = colorsys.hls_to_rgb(h, l, s)
        cluster_centers[i] = np.array([r, g, b])
    return cluster_centers


def manipulate_colors(cluster_centers, levels=0., saturation=0., lightness=0.):
    cluster_centers = change_color_levels(cluster_centers, levels)
    cluster_centers = change_saturation_lightness(cluster_centers, saturation, lightness)
    return cluster_centers


def kmeans_compress(img_in, k, levels=0., saturation=0., lightness=0., custom_colors=None):
    img_in_vector = img_in.reshape((img_in.shape[0]*img_in.shape[1], img_in.shape[2]))

    ### Using sklearn KMeans
    # km = KMeans(n_clusters=k)
    # km.fit(img_in_vector)
    # one_hots = get_one_hot(km.labels_)
    # label_map = km.labels_.reshape(img_in.shape[0], img_in.shape[1])
    #
    # km.cluster_centers_ = manipulate_colors(km.cluster_centers_, levels, saturation, lightness)
    #
    # out_img_vector = np.dot(one_hots, km.cluster_centers_)
    # out_img = out_img_vector.reshape(img_in.shape[0], img_in.shape[1], img_in.shape[2])
    # ###

    ### Using Custom KMeans
    if type(k) == type([]):
        km = MultiChannelKMeans(img_in_vector, k, convergence=0.001, max_iter=20, max_init=100)
    else:
        km = ImageKMeans(img_in_vector, k, convergence=0.001, max_iter=20, max_init=100)

    if custom_colors is not None:
        custom_colors_array = json_to_np(custom_colors)
        km.set_custom_colors(custom_colors_array)

    km.fit()
    out_img = km.get_image(img_in.shape[0], img_in.shape[1])
    label_map = None
    ###


    return out_img, label_map, km


def load_image(fname):
    img_in = mpimg.imread(fname)
    img_in = deepcopy(img_in.astype(float))/255.
    return img_in


def main(img_in, k, conv_object, n_repeats, show_plot=False, levels=0., saturation=0., lightness=0., custom_colors=None):
    img_out, label_map, km = kmeans_compress(img_in, k, levels, saturation, lightness, custom_colors)
    for i in range(n_repeats):
        img_out = blur_rgb(img_out, conv_object)
        km.image_vector = img_out
        km.fit()
        img_out = km.get_image(img_in.shape[0], img_in.shape[1])
        # img_out, label_map = kmeans_compress(img_out, k, levels, saturation, lightness, custom_colors)
        if show_plot:
            plt.figure()
            plt.imshow(img_out)
            plt.show()

    return img_out


def resize(img_in, aspect_ratio, centering):
    height = img_in.shape[0]
    width = img_in.shape[1]
    if width/height > aspect_ratio:
        new_width = int(round(height * aspect_ratio))
        crop_amount = width - new_width
        left_crop = int(round(centering * crop_amount))
        right_crop = int(round((1-centering) * crop_amount))
        img_out = img_in[:, left_crop:width-right_crop, :]
    elif width/height < aspect_ratio:
        crop_amount = width - new_width
        bottom_crop = centering * crop_amount
        top_crop = (1-centering) * crop_amount
        new_height = int(round(width / aspect_ratio))
        img_out = img_in[top_crop:height-bottom_crop, :, :]
    else:
        img_out = img_in
    return img_out

if __name__ == '__main__':
    n_repeats = 0
    k = [3,3,1]
    levels = 0.
    saturation = 0.
    lightness = 0.
    show_plot = False
    project_name = 'birch_warm'
    extension = 'jpg'
    fname = f'images/{project_name}/input.{extension}'
    custom_colors = None
    # custom_colors = f'images/{project_name}/c2.json'
    save = True

    img_in = load_image(fname)
    # img_in = resize(img_in, 140/98, 0.75)

    conv_object = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2, bias=False, padding_mode='replicate')
    conv_object.weight.requires_grad = False
    conv_object.weight[:,:,:,:] = 1./np.product(conv_object.weight.shape)

    img_out = main(img_in, k, conv_object, n_repeats, show_plot, levels, saturation, lightness, custom_colors)

    if save:
        plt.imsave(f'images/{project_name}/img_in.png', img_in)
        if custom_colors is not None:
            cc_fname = custom_colors.split('/')[-1].split('.')[0]
            fname = f'images/{project_name}/out_{k}_{cc_fname}.png'
        else:
            fname = f'images/{project_name}/out_{k}.png'
        plt.imsave(fname, img_out)
    else:
        plt.imshow(img_in)
        plt.figure()
        plt.imshow(img_out)
        plt.show()
