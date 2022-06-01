import os
import math
import mcubes
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class IMNETDecoder(nn.Module):
    def __init__(self, z_dim, df_dim):
        super(IMNETDecoder, self).__init__()
        self.linear_1 = nn.Linear(z_dim, df_dim * 8, bias=True)
        self.linear_2 = nn.Linear(df_dim * 8, df_dim * 8, bias=True)
        self.linear_3 = nn.Linear(df_dim * 8, df_dim * 8, bias=True)
        self.linear_4 = nn.Linear(df_dim * 8, df_dim * 4, bias=True)
        self.linear_5 = nn.Linear(df_dim * 4, df_dim * 2, bias=True)
        self.linear_6 = nn.Linear(df_dim * 2, df_dim * 1, bias=True)
        self.linear_7 = nn.Linear(df_dim * 1, 1, bias=True)
        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_3.bias, 0)
        nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_4.bias, 0)
        nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_5.bias, 0)
        nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_6.bias, 0)
        nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
        nn.init.constant_(self.linear_7.bias, 0)

    def forward(self, z):
        # input z - B x 4096 x z_dim
        l1 = self.linear_1(z)
        l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

        l4 = self.linear_4(l3)
        l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

        l5 = self.linear_5(l4)
        l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

        l6 = self.linear_6(l5)
        l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

        l7 = self.linear_7(l6)
        l7 = torch.max(torch.min(l7, l7 * 0.01 + 0.99), l7 * 0.01)

        return l7


class GridEncoder2D(nn.Module):
    def __init__(self, ef_dim=16, z_dim=64):
        super(GridEncoder2D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, ef_dim, 3, stride=2, padding=1), nn.ReLU(),  # (B, 16, 128, 128)
            nn.Conv2d(ef_dim * 1, ef_dim * 1, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(ef_dim * 1),
            nn.Conv2d(ef_dim * 1, ef_dim * 2, 3, stride=2, padding=1), nn.ReLU(),  # (B, 32, 64, 64)
            nn.Conv2d(ef_dim * 2, ef_dim * 2, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(ef_dim * 2),
            nn.Conv2d(ef_dim * 2, ef_dim * 4, 3, stride=2, padding=1), nn.ReLU(),  # (B, 64, 32, 32)
            nn.Conv2d(ef_dim * 4, ef_dim * 4, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(ef_dim * 4),
            nn.Conv2d(ef_dim * 4, ef_dim * 8, 3, stride=2, padding=1), nn.ReLU(),  # (B, 128, 16, 16)
            nn.Conv2d(ef_dim * 8, ef_dim * 8, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(ef_dim * 8),
            nn.Conv2d(ef_dim * 8, ef_dim * 16, 3, stride=2, padding=1), nn.ReLU(),  # (B, 256, 8, 8)
            nn.Conv2d(ef_dim * 16, ef_dim * 16, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(ef_dim * 16),
            nn.Conv2d(ef_dim * 16, ef_dim * 32, 3, stride=2, padding=1), nn.ReLU(),  # (B, 512, 4, 4)
            nn.Conv2d(ef_dim * 32, ef_dim * 32, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(ef_dim * 32),
            nn.Conv2d(ef_dim * 32, ef_dim * 4, 3, stride=2, padding=1), nn.ReLU(),  # (B, 64, 2, 2)
            nn.Conv2d(ef_dim * 4, ef_dim * 4, 1, stride=1, padding=0), nn.ReLU(),
        )

    def forward(self, inputs):
        out = self.layers(inputs)

        return out


class GridDecoder2D(nn.Module):
    def __init__(self, z_dim=64, df_dim=128):
        super(GridDecoder2D, self).__init__()
        self.decoder = IMNETDecoder(z_dim=z_dim, df_dim=df_dim)

    def forward(self, zs, points):
        points = (2 * points).unsqueeze(1)  # (B, 1, 4096, 2)
        zs_sampled = F.grid_sample(zs, points, padding_mode='border', align_corners=True)  # (B, 64, 1, 4096)
        zs_sampled = zs_sampled.squeeze(2).permute(0, 2, 1)  # (B, 4096, 64)
        sdf = self.decoder(zs_sampled)

        return sdf


class GridAE2D(nn.Module):
    def __init__(self, ef_dim=16, z_dim=64, df_dim=128):
        super(GridAE2D, self).__init__()
        self.encoder = GridEncoder2D(ef_dim=ef_dim, z_dim=z_dim)
        self.decoder = GridDecoder2D(z_dim=z_dim, df_dim=df_dim)

    def forward(self, inputs, points):
        z = self.encoder(inputs)
        out = self.decoder(z, points)
        return out


class GridEncoder3D(nn.Module):
    def __init__(self, ef_dim=16, z_dim=32):
        super(GridEncoder3D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(1, ef_dim, 3, stride=1, padding=1), nn.ReLU(),  # (B, 16, 64, 64, 64)
            nn.Conv3d(ef_dim * 1, ef_dim * 1, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm3d(ef_dim * 1),
            nn.Conv3d(ef_dim * 1, ef_dim * 2, 3, stride=2, padding=1), nn.ReLU(),  # (B, 32, 32, 32, 32)
            nn.Conv3d(ef_dim * 2, ef_dim * 2, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm3d(ef_dim * 2),
            nn.Conv3d(ef_dim * 2, ef_dim * 4, 3, stride=2, padding=1), nn.ReLU(),  # (B, 64, 16, 16, 16)
            nn.Conv3d(ef_dim * 4, ef_dim * 4, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm3d(ef_dim * 4),
            nn.Conv3d(ef_dim * 4, ef_dim * 8, 3, stride=2, padding=1), nn.ReLU(),  # (B, 128, 8, 8, 8)
            nn.Conv3d(ef_dim * 8, ef_dim * 8, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm3d(ef_dim * 8),
            nn.Conv3d(ef_dim * 8, ef_dim * 16, 3, stride=2, padding=1), nn.ReLU(),  # (B, 256, 4, 4, 4)
            nn.Conv3d(ef_dim * 16, ef_dim * 16, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm3d(ef_dim * 16),
            nn.Conv3d(ef_dim * 16, ef_dim * 32, 3, stride=2, padding=1), nn.ReLU(),  # (B, 512, 2, 2, 2)
            nn.Conv3d(ef_dim * 32, ef_dim * 32, 3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm3d(ef_dim * 32),
            nn.Conv3d(ef_dim * 32, z_dim, 3, stride=1, padding=1), nn.ReLU(),  # (B, 32, 2, 2, 2)
            nn.Conv3d(z_dim, z_dim, 3, stride=1, padding=1), nn.ReLU(),
        )

    def forward(self, inputs):
        out = self.layers(inputs)

        return out


class GridDecoder3D(nn.Module):
    def __init__(self, z_dim=32, df_dim=128):
        super(GridDecoder3D, self).__init__()
        self.decoder = IMNETDecoder(z_dim=z_dim, df_dim=df_dim)

    def forward(self, zs, points):
        points = (2 * points).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 4096, 3)
        zs_sampled = F.grid_sample(zs, points, padding_mode='border', align_corners=True)  # (B, 32, 1, 1, 4096)
        zs_sampled = zs_sampled.squeeze(2).squeeze(2).permute(0, 2, 1)  # (B, 4096, 32)
        sdf = self.decoder(zs_sampled)

        return sdf


class GridAE3D(nn.Module):
    def __init__(self, ef_dim=16, z_dim=32, df_dim=128):
        super(GridAE3D, self).__init__()
        self.encoder = GridEncoder3D(ef_dim=ef_dim, z_dim=z_dim)
        self.decoder = GridDecoder3D(z_dim=z_dim, df_dim=df_dim)

    def forward(self, inputs, points):
        z = self.encoder(inputs)
        out = self.decoder(z, points)
        return out


class _CustomDataParallel(nn.DataParallel):
    def __init__(self, model):
        super(_CustomDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Tester:
    def __init__(self, device, cell_grid_size=4, frame_grid_size=64):
        self.test_size = 32  # related to testing batch_size, adjust according to gpu memory size
        self.cell_grid_size = cell_grid_size
        self.frame_grid_size = frame_grid_size
        self.real_size = self.cell_grid_size * self.frame_grid_size  # =256, output point-value voxel grid size in testing
        self.test_point_batch_size = self.test_size * self.test_size * self.test_size  # 32 x 32 x 32, do not change
        self.sampling_threshold = 0.5
        self.device = device

        self.get_test_coord_for_training()  # initialize self.coords
        self.get_test_coord_for_testing()  # initialize self.frame_coords

    def get_test_coord_for_training(self):
        dima = self.test_size  # 32
        dim = self.frame_grid_size  # 64
        multiplier = int(dim / dima)  # 2
        multiplier2 = multiplier * multiplier
        multiplier3 = multiplier * multiplier * multiplier
        ranges = np.arange(0, dim, multiplier, np.uint8)
        self.aux_x = np.ones([dima, dima, dima], np.uint8) * np.expand_dims(ranges, axis=(1, 2))
        self.aux_y = np.ones([dima, dima, dima], np.uint8) * np.expand_dims(ranges, axis=(0, 2))
        self.aux_z = np.ones([dima, dima, dima], np.uint8) * np.expand_dims(ranges, axis=(0, 1))
        self.coords = np.zeros([multiplier ** 3, dima, dima, dima, 3], np.float32)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 0] = self.aux_x + i
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 1] = self.aux_y + j
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 2] = self.aux_z + k
        self.coords = (self.coords.astype(np.float32) + 0.5) / dim - 0.5
        self.coords = np.reshape(self.coords, [multiplier3, self.test_point_batch_size, 3])
        self.coords = torch.from_numpy(self.coords).to(self.device)

    def get_test_coord_for_testing(self):
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size
        self.cell_x = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_y = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_z = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_coords = np.zeros([dimf, dimf, dimf, dimc, dimc, dimc, 3], np.float32)
        self.frame_coords = np.zeros([dimf, dimf, dimf, 3], np.float32)
        self.frame_x = np.zeros([dimf, dimf, dimf], np.int32)
        self.frame_y = np.zeros([dimf, dimf, dimf], np.int32)
        self.frame_z = np.zeros([dimf, dimf, dimf], np.int32)
        for i in range(dimc):
            for j in range(dimc):
                for k in range(dimc):
                    self.cell_x[i, j, k] = i
                    self.cell_y[i, j, k] = j
                    self.cell_z[i, j, k] = k
        for i in range(dimf):
            for j in range(dimf):
                for k in range(dimf):
                    self.cell_coords[i, j, k, :, :, :, 0] = self.cell_x + i * dimc
                    self.cell_coords[i, j, k, :, :, :, 1] = self.cell_y + j * dimc
                    self.cell_coords[i, j, k, :, :, :, 2] = self.cell_z + k * dimc
                    self.frame_coords[i, j, k, 0] = i
                    self.frame_coords[i, j, k, 1] = j
                    self.frame_coords[i, j, k, 2] = k
                    self.frame_x[i, j, k] = i
                    self.frame_y[i, j, k] = j
                    self.frame_z[i, j, k] = k

        self.cell_coords = (self.cell_coords.astype(np.float32) + 0.5) / self.real_size - 0.5
        self.cell_coords = np.reshape(self.cell_coords, [dimf, dimf, dimf, dimc * dimc * dimc, 3])
        self.cell_x = np.reshape(self.cell_x, [dimc * dimc * dimc])
        self.cell_y = np.reshape(self.cell_y, [dimc * dimc * dimc])
        self.cell_z = np.reshape(self.cell_z, [dimc * dimc * dimc])
        self.frame_x = np.reshape(self.frame_x, [dimf * dimf * dimf])
        self.frame_y = np.reshape(self.frame_y, [dimf * dimf * dimf])
        self.frame_z = np.reshape(self.frame_z, [dimf * dimf * dimf])
        self.frame_coords = (self.frame_coords.astype(np.float32) + 0.5) / dimf - 0.5
        self.frame_coords = np.reshape(self.frame_coords, [dimf * dimf * dimf, 3])
        self.frame_coords = torch.from_numpy(self.frame_coords).to(self.device)

    def test_during_train(self, network, input_voxel, name):
        network.eval()
        batch_voxels = input_voxel.unsqueeze(0).to(self.device)  # batch_voxel - 1 x 1 x 64 x 64 x 64
        model_float = np.zeros([self.frame_grid_size + 2, self.frame_grid_size + 2, self.frame_grid_size + 2], np.float32)
        multiplier = int(self.frame_grid_size / self.test_size)
        multiplier2 = multiplier * multiplier
        with torch.no_grad():
            zs_vector = network.encoder(batch_voxels)

        for idx, z_vector in enumerate(zs_vector):
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        minib = i * multiplier2 + j * multiplier + k
                        point_coord = self.coords[minib:minib + 1]
                        with torch.no_grad():
                            net_out = network.decoder(z_vector.unsqueeze(0), point_coord)
                        model_float[self.aux_x + i + 1, self.aux_y + j + 1, self.aux_z + k + 1] = np.reshape(net_out.detach().cpu().numpy(),
                                                                                                             [self.test_size,
                                                                                                              self.test_size,
                                                                                                              self.test_size])
            vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
            vertices = (vertices.astype(np.float32) - 0.5) / self.frame_grid_size - 0.5
            write_ply_triangle(name + '-z{}.ply'.format(idx), vertices, triangles)
        print("[sample]")

    # output shape as ply and point cloud as ply if specified
    def test_mesh_point(self, network, input_data, name, input_type='voxel', save_mesh=True, save_point=False):
        # input_data could be [voxel] or [z], should be either 1x1x64x64x64 or 1x256
        zs_vector = network.encoder(input_data) if input_type == 'voxel' else input_data

        for idx, z_vector in enumerate(zs_vector):
            model_float = self.z2voxel(network, z_vector.unsqueeze(0))
            vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
            vertices = (vertices.astype(np.float32) - 0.5) / self.real_size - 0.5
            # optimized_vertices = self.optimize_mesh(network, vertices, z_vector)

            if save_mesh:
                write_ply_triangle(name + '_z{}_vox.ply'.format(idx), vertices, triangles)
            if save_point:
                sampled_points_normals = sample_points_triangle(vertices, triangles, num_of_points=2048)
                np.random.shuffle(sampled_points_normals)
                # write_ply_point_normal(name + '_z{}_pcd.ply'.format(idx), sampled_points_normals)
                write_ply_point(name + '_z{}_pcd.ply'.format(idx), sampled_points_normals)
            break
        print("[sample]")

    def test_point_as_image(self, network, input_data, input_type='z', scale=1):
        model_float = self.z2voxel(network, input_data) if input_type == 'z' else input_data
        vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
        vertices = (vertices.astype(np.float32) - 0.5) / self.real_size - 0.5
        sampled_points_normals = sample_points_triangle(vertices, triangles, num_of_points=2048)
        np.random.shuffle(sampled_points_normals)
        centroid = np.mean(sampled_points_normals[:, :3], axis=0)
        sampled_points_normals[:, :3] += -centroid
        image = plot_3d_point_to_image(sampled_points_normals[:, :3] * scale)  # change scale to zoom in or out
        return image

    def z2voxel(self, network, z):
        model_float = np.zeros([self.real_size + 2, self.real_size + 2, self.real_size + 2], np.float32)
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size

        frame_flag = np.zeros([dimf + 2, dimf + 2, dimf + 2], np.uint8)
        queue = []
        frame_batch_num = int(dimf ** 3 / self.test_point_batch_size)
        assert frame_batch_num > 0

        # get frame grid values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            point_coord = point_coord.unsqueeze(dim=0)
            with torch.no_grad():
                model_out_ = network.decoder(z, point_coord)
                model_out = model_out_.detach().cpu().numpy()[0]
            x_coords = self.frame_x[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            y_coords = self.frame_y[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            z_coords = self.frame_z[i * self.test_point_batch_size:(i + 1) * self.test_point_batch_size]
            frame_flag[x_coords + 1, y_coords + 1, z_coords + 1] = np.reshape((model_out > self.sampling_threshold).astype(np.uint8),
                                                                              [self.test_point_batch_size])

        # get queue and fill up ones
        for i in range(1, dimf + 1):
            for j in range(1, dimf + 1):
                for k in range(1, dimf + 1):
                    maxv = np.max(frame_flag[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                    minv = np.min(frame_flag[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2])
                    if maxv != minv:
                        queue.append((i, j, k))
                    elif maxv == 1:
                        x_coords = self.cell_x + (i - 1) * dimc
                        y_coords = self.cell_y + (j - 1) * dimc
                        z_coords = self.cell_z + (k - 1) * dimc
                        model_float[x_coords + 1, y_coords + 1, z_coords + 1] = 1.0

        # print("running queue:", len(queue))
        cell_batch_size = dimc ** 3
        cell_batch_num = int(self.test_point_batch_size / cell_batch_size)
        assert cell_batch_num > 0
        # run queue
        while len(queue) > 0:
            batch_num = min(len(queue), cell_batch_num)
            point_list = []
            cell_coords = []
            for i in range(batch_num):
                point = queue.pop(0)
                point_list.append(point)
                cell_coords.append(self.cell_coords[point[0] - 1, point[1] - 1, point[2] - 1])
            cell_coords = np.concatenate(cell_coords, axis=0)
            cell_coords = np.expand_dims(cell_coords, axis=0)
            cell_coords = torch.from_numpy(cell_coords)
            cell_coords = cell_coords.to(self.device)
            with torch.no_grad():
                model_out_batch_ = network.decoder(z, cell_coords)
                model_out_batch = model_out_batch_.detach().cpu().numpy()[0]
            for i in range(batch_num):
                point = point_list[i]
                model_out = model_out_batch[i * cell_batch_size:(i + 1) * cell_batch_size, 0]
                x_coords = self.cell_x + (point[0] - 1) * dimc
                y_coords = self.cell_y + (point[1] - 1) * dimc
                z_coords = self.cell_z + (point[2] - 1) * dimc
                model_float[x_coords + 1, y_coords + 1, z_coords + 1] = model_out

                if np.max(model_out) > self.sampling_threshold:
                    for i in range(-1, 2):
                        pi = point[0] + i
                        if pi <= 0 or pi > dimf:
                            continue
                        for j in range(-1, 2):
                            pj = point[1] + j
                            if pj <= 0 or pj > dimf:
                                continue
                            for k in range(-1, 2):
                                pk = point[2] + k
                                if pk <= 0 or pk > dimf:
                                    continue
                                if frame_flag[pi, pj, pk] == 0:
                                    frame_flag[pi, pj, pk] = 1
                                    queue.append((pi, pj, pk))
        return model_float

    def optimize_mesh(self, network, vertices, z, iterations=3):
        new_vertices = np.copy(vertices)

        new_vertices_ = np.expand_dims(new_vertices, axis=0)
        new_vertices_ = torch.from_numpy(new_vertices_)
        new_vertices_ = new_vertices_.to(self.device)
        new_v_out_ = network.decoder(z, new_vertices_)
        new_v_out = new_v_out_.detach().cpu().numpy()[0]

        for iteration in range(iterations):
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    for k in [-1, 0, 1]:
                        if i == 0 and j == 0 and k == 0: continue
                        offset = np.array([[i, j, k]], np.float32) / (self.real_size * 6 * 2 ** iteration)
                        current_vertices = vertices + offset
                        current_vertices_ = np.expand_dims(current_vertices, axis=0)
                        current_vertices_ = torch.from_numpy(current_vertices_)
                        current_vertices_ = current_vertices_.to(self.device)
                        current_v_out_ = network.decoder(z, current_vertices_)
                        current_v_out = current_v_out_.detach().cpu().numpy()[0]
                        keep_flag = abs(current_v_out - self.sampling_threshold) < abs(new_v_out - self.sampling_threshold)
                        keep_flag = keep_flag.astype(np.float32)
                        new_vertices = current_vertices * keep_flag + new_vertices * (1 - keep_flag)
                        new_v_out = current_v_out * keep_flag + new_v_out * (1 - keep_flag)
            vertices = new_vertices
        return vertices


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    test_input = torch.randn(16, 1, 256, 256).cuda()
    test_point = torch.randn(16, 4096, 2).cuda()
    AE = GridAE3D(ef_dim=16, z_dim=32, df_dim=128, multi_scale=False).cuda()
    num_parameters = count_parameters(AE)
    print(num_parameters)
