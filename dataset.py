import os
import h5py
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class Config:
    def __init__(self):
        self.train = True
        self.domain = ''
        # self.sample_im_size = 64
        self.sample_vox_size = 64
        self.data_dir = './data'
        self.dataset = 'chair-table'


# dataset for AE training
class DomainABDatasetFor3D(Dataset):
    def __init__(self, config, domain, mode):
        self.mode = mode
        self.domain = domain
        self.sample_vox_size = config.sample_vox_size
        self.load_point_batch_size = 16*16*16*4 \
            if self.sample_vox_size == 64 else 16*16*16
        self.point_batch_size = 16*16*16

        self.data_dir = config.data_dir
        self.dataset_name = config.dataset
        self.dataset_load = '{}_{}.hdf5'.format(self.domain, self.mode)
        data_hdf5_name = os.path.join(self.data_dir, self.dataset_name, self.dataset_load)
        if os.path.exists(data_hdf5_name):
            data_dict = h5py.File(data_hdf5_name, 'r')
            self.data_points = (data_dict['points_' + str(self.sample_vox_size)][:].astype(np.float32) + 0.5) / 256 - 0.5
            self.data_values = data_dict['values_' + str(self.sample_vox_size)][:].astype(np.float32)
            self.data_voxels = data_dict['voxels'][:]
            self.data_file_names = data_dict['file_names'][:].astype(str).ravel()
        else:
            raise FileNotFoundError(data_hdf5_name)

    def __getitem__(self, idx):
        # if number of point of a sample is greater than 4096, randomly choose 4096 points
        point_batch_num = int(self.load_point_batch_size / self.point_batch_size)
        if point_batch_num == 1:
            point_coord = self.data_points[idx]
            point_value = self.data_values[idx]
        else:
            which_batch = np.random.randint(point_batch_num)
            point_coord = self.data_points[idx, which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size]
            point_value = self.data_values[idx, which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size]

        batch_voxel = self.data_voxels[idx].astype(np.float32)
        batch_voxel = torch.from_numpy(batch_voxel)
        point_coord = torch.from_numpy(point_coord)
        point_value = torch.from_numpy(point_value)
        batch_file_names = self.data_file_names[idx]

        # permute voxel to channel first 64x64x64x1 -> 1x64x64x64
        batch_voxel = batch_voxel.permute(3, 0, 1, 2)

        return batch_voxel, point_coord, point_value, batch_file_names

    def __len__(self):
        return len(self.data_points)


class DomainABDatasetFor2D(Dataset):
    def __init__(self, config, domain, mode):
        self.mode = mode
        self.domain = domain
        if config.sample_im_size == 64:
            self.load_point_batch_size = 64 * 64
            self.point_batch_size = 64 * 64
        else:
            assert config.sample_im_size == 128
            self.load_point_batch_size = 64 * 64 * 4
            self.point_batch_size = 64 * 64

        self.data_dir = config.data_dir
        self.dataset_name = config.dataset
        self.dataset_load = '{}_{}.hdf5'.format(self.domain, self.mode)  #
        data_hdf5_name = os.path.join(self.data_dir, self.dataset_name, self.dataset_load)
        if os.path.exists(data_hdf5_name):
            data_dict = h5py.File(data_hdf5_name, 'r')
            self.data_points = (data_dict['points_' + str(config.sample_im_size)][:].astype(np.float32) + 0.5) / 256 - 0.5
            self.data_values = data_dict['values_' + str(config.sample_im_size)][:].astype(np.float32)
            self.data_pixels = data_dict['pixels'][:].astype(np.float32)
            self.data_file_names = data_dict['file_names'][:].astype(str).ravel()
            try:
                self.data_weights = data_dict['weights_' + str(config.sample_im_size)][:].astype(np.float32)
            except KeyError:
                self.data_weights = np.ones(self.data_values.shape)
            print("Unique value of weights: ", np.unique(self.data_weights))
        else:
            raise FileNotFoundError(data_hdf5_name)

    def __getitem__(self, idx):
        point_batch_num = int(self.load_point_batch_size / self.point_batch_size)
        if point_batch_num == 1:
            batch_coords = self.data_points[idx]
            batch_values = self.data_values[idx]
            batch_weights = self.data_weights[idx]
        else:
            which_batch = np.random.randint(point_batch_num)
            batch_coords = self.data_points[idx, which_batch * self.point_batch_size:(which_batch + 1) * self.point_batch_size]
            batch_values = self.data_values[idx, which_batch * self.point_batch_size:(which_batch + 1) * self.point_batch_size]
            batch_weights = self.data_weights[idx, which_batch * self.point_batch_size:(which_batch + 1) * self.point_batch_size]

        batch_pixels = self.data_pixels[idx:idx + 1]
        batch_pixels = torch.from_numpy(batch_pixels)
        batch_coords = torch.from_numpy(batch_coords)
        batch_values = torch.from_numpy(batch_values)
        batch_file_names = self.data_file_names[idx]
        batch_weights = torch.from_numpy(batch_weights)

        return batch_pixels, batch_coords, batch_values, batch_file_names, batch_weights

    def __len__(self):
        return len(self.data_points)


# dataset for GAN training
class LatentCodeDataset:
    def __init__(self, hdf5_path, init_shuffle=True):
        data_dict = h5py.File(hdf5_path, 'r')
        self.zs_vector = data_dict["zs"][:]
        self.file_names = data_dict["file_names"][:].astype(str).ravel()
        self.input_data = data_dict["input_data"][:]  # could be pixels or voxels depending on 2d or 3d data
        self.num_examples = self.zs_vector.shape[0]
        self._index_in_epoch = 0
        if init_shuffle:
            self.shuffle_data()

    def shuffle_data(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        permutation = np.arange(self.num_examples)
        np.random.shuffle(permutation)
        self.zs_vector = self.zs_vector[permutation]
        self.file_names = self.file_names[permutation]
        self.input_data = self.input_data[permutation]

        return self

    def next_batch(self, batch_size, seed=None):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            self.shuffle_data(seed)
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        zs = torch.from_numpy(self.zs_vector[start:end])  # (B, 256) / (B, 64, 2, 2) / (B, 32, 2, 2, 2)
        file_names = self.file_names[start:end]
        input_data = self.input_data[start:end]

        return zs, file_names, input_data

    def get_one_example(self, idx=0):
        z = torch.from_numpy(self.zs_vector[idx:idx+1])  # (B, 256) / (B, 64, 2, 2) / (B, 32, 2, 2, 2)
        file_name = self.file_names[idx:idx+1]
        input_data = self.input_data[idx:idx+1]

        return z, file_name, input_data


if __name__ == '__main__':
    config = Config()
