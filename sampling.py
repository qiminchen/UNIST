import os
import h5py
import numpy as np
from PIL import Image, ImageFilter
from scipy.spatial import distance

dim = 256
threshold = 10
ranges = np.arange(0, dim, 1, np.uint8)
aux_x = np.ones([dim, dim], np.uint8) * np.expand_dims(ranges, axis=1)
aux_y = np.ones([dim, dim], np.uint8) * np.expand_dims(ranges, axis=0)
coord = np.hstack((aux_x.reshape(-1, 1), aux_y.reshape(-1, 1)))

# Please change accordingly
all_images = []
num_samples = len(all_images)
path_to_img = ''
path_to_save_hdf5 = ''

# We sample 128*128=16384 points in total
# you could sample more or less, please change accordingly
pixels = np.zeros((num_samples, dim, dim))
points_128 = np.zeros((num_samples, 128*128, 2))
values_128 = np.zeros((num_samples, 128*128, 1))

for i, img in enumerate(all_images):
    im = Image.open(os.path.join(path_to_img, img))
    edges = im.filter(ImageFilter.FIND_EDGES)
    x, y = np.where(np.array(edges) == 255)
    boundary_points = np.array(list(zip(x, y)))
    cdist = distance.cdist(coord, boundary_points, 'euclidean')
    cdist_min = np.min(cdist, axis=1)
    idx_cdist_min_lt_threshold = np.where(cdist_min <= threshold)

    # points near the boundary
    idx_cdist_min_lt_threshold_128 = np.random.choice(idx_cdist_min_lt_threshold[0],
                                                      min(int(16384 * 3 / 6), len(idx_cdist_min_lt_threshold[0])), replace=False)
    selected_points_near_128 = coord[idx_cdist_min_lt_threshold_128]
    selected_values_near_128 = np.array(im).reshape(-1, 1)[idx_cdist_min_lt_threshold_128]

    # points inside the boundary
    idx_inside_points = np.where(np.array(im).reshape(-1, ) == 255)
    idx_inside_points_not_near_boundary = np.setdiff1d(idx_inside_points, idx_cdist_min_lt_threshold_128)
    idx_inside_128 = np.random.choice(idx_inside_points_not_near_boundary,
                                      min(int(16384 * 2 / 6), len(idx_inside_points_not_near_boundary)), replace=False)
    selected_inside_points_128 = coord[idx_inside_128]
    selected_inside_values_128 = np.array(im).reshape(-1, 1)[idx_inside_128]

    # points outside the boundary
    idx_outside_points = np.setdiff1d(np.arange(len(coord)), np.hstack((idx_cdist_min_lt_threshold_128, idx_inside_128)))
    idx_outside_128 = np.random.choice(idx_outside_points, min(16384 - len(idx_cdist_min_lt_threshold_128) - len(idx_inside_128),
                                                               len(idx_outside_points)), replace=False)
    selected_outside_points_128 = coord[idx_outside_128]
    selected_outside_values_128 = np.array(im).reshape(-1, 1)[idx_outside_128]

    selected_points_128 = np.vstack((selected_points_near_128,
                                     selected_inside_points_128,
                                     selected_outside_points_128))
    selected_values_128 = np.vstack((selected_values_near_128,
                                     selected_inside_values_128,
                                     selected_outside_values_128))

    # shuffle points and sdf values
    np.random.seed(2022)
    permuted_idx = np.random.permutation(len(selected_points_128))
    selected_points_128 = selected_points_128[permuted_idx]
    selected_values_128 = selected_values_128[permuted_idx]

    pixels[i] = (np.array(im) / 255).astype(np.uint8)
    points_128[i] = selected_points_128
    values_128[i] = selected_values_128 / 255

hdf5_file = h5py.File(path_to_save_hdf5, mode='w')
hdf5_file.create_dataset("pixels", [num_samples, dim, dim], np.uint8, data=pixels, compression=9)
hdf5_file.create_dataset("points_128", [num_samples, 128*128, 2], np.uint8, data=points_128, compression=9)
hdf5_file.create_dataset("values_128", [num_samples, 128*128, 1], np.uint8, data=values_128, compression=9)
hdf5_file.create_dataset("file_names", [num_samples, 1], data=all_images, compression=9)
hdf5_file.close()
