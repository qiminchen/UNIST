import os
import sys
import time
import h5py
import mcubes
import argparse
import numpy as np
from PIL import Image
from collections import OrderedDict

from modelAE import GridAE3D, _CustomDataParallel, Tester
from dataset import DomainABDatasetFor3D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from utils import write_ply_triangle

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Number of epoch to train [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00005, type=float, help="Learning rate [0.00005]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--lambda1", action="store", dest="lambda1", default=0.1, type=float, help="Scalar weight for sub feature vectors loss")
parser.add_argument("--dataset", action="store", dest="dataset", default="chair_table", help="The name of dataset domain A_B [chair_table]")
parser.add_argument("--batch_size", action="store", dest="batch_size", default=20, type=int, help="Batch size [8]")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="./checkpoint",
                    help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="./data", help="Root directory of dataset [data]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples",
                    help="Directory name to save the image samples [samples]")
parser.add_argument("--sample_vox_size", action="store", dest="sample_vox_size", default=64, type=int,
                    help="Voxel resolution for coarse-to-fine training [64]")
parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
parser.add_argument("--getz", action="store_true", dest="getz", default=False, help="True for getting latent codes [False]")
config = parser.parse_args()


# train  - python run_ae.py --train --epoch 300 --dataset chair-table --sample_vox_size 16
# train  - python run_ae.py --train --epoch 300 --dataset chair-table --sample_vox_size 32
# train  - python run_ae.py --train --epoch 600 --dataset chair-table --sample_vox_size 64

# test   - python run_ae.py --dataset chair-table --sample_dir outputs
# trainz - python run_ae.py --train --getz --dataset chair-table
# testz  - python run_ae.py --getz --dataset chair-table

mode = 'train' if config.train else 'test'

# ---------------------------- domain A and domain B ----------------------------
domain_a = config.dataset.split('-')[0]  # chair
domain_b = config.dataset.split('-')[1]  # table

# ------------- create sample folder for test samples if not exist --------------
sample_a_dir = os.path.join(config.sample_dir, config.dataset, domain_a + '_{}_{}'.format(config.network_type, config.sample_vox_size))
sample_b_dir = os.path.join(config.sample_dir, config.dataset, domain_b + '_{}_{}'.format(config.network_type, config.sample_vox_size))
if not os.path.exists(sample_a_dir):
    os.makedirs(sample_a_dir)
if not os.path.exists(sample_b_dir):
    os.makedirs(sample_b_dir)

# ------------------- Create checkpoint folder if not exist ---------------------
checkpoint_dir = os.path.join(config.checkpoint_dir, config.dataset, '3d_grid_ae_32x2x2x2')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def log(strs):
    fp.write('%s\n' % strs)
    fp.flush()
    print(strs)


fp = open(os.path.join(checkpoint_dir, 'logs_{}_{}.txt'.format(mode, config.sample_vox_size)), 'w')
s = 'python run_2dae.py'
for j in sys.argv:
    s += ' ' + j
log(s)

# Create network and optimizer
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

log("===> Loading Grid AE")
network = GridAE3D(ef_dim=16, z_dim=32, df_dim=128)
log("===> Number of parameters: {:,}".format(sum(p.numel() for p in network.parameters() if p.requires_grad)))
if torch.cuda.device_count() > 1:
    network = _CustomDataParallel(network)
network.to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
tester = Tester(device)

# --------------------------- Load previous checkpoint --------------------------
# need to load previous checkpoint because of progressive training
checkpoint_txt = os.path.join(checkpoint_dir, 'checkpoint')
if os.path.exists(checkpoint_txt):
    fin = open(checkpoint_txt)
    model_dir = fin.readline().strip()
    fin.close()
    state_dict = torch.load(model_dir)
    try:
        network.load_state_dict(state_dict)
    except RuntimeError:
        # for model trained using multiple GPUs
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        network.load_state_dict(new_state_dict)
    log("===> Model {} load SUCCESS".format(model_dir))
else:
    log("===> Model {} load failed...".format(checkpoint_txt))

# -------------------------------- Load dataset --------------------------------
dataset_A_train = DomainABDatasetFor3D(config, domain=domain_a, mode='train')
dataset_B_train = DomainABDatasetFor3D(config, domain=domain_b, mode='train')
dataset_A_test = DomainABDatasetFor3D(config, domain=domain_a, mode='test')
dataset_B_test = DomainABDatasetFor3D(config, domain=domain_b, mode='test')
dataset_train = ConcatDataset([dataset_A_train, dataset_B_train])
dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=4)
log('\n------------------------------------------------')
log('Number of training data from domain [{}]: {}'.format(domain_a, len(dataset_A_train)))
log('Number of training data from domain [{}]: {}'.format(domain_b, len(dataset_B_train)))
log('Number of training data from both domain: {}'.format(len(dataset_train)))
log('Number of testing data from domain [{}]: {}'.format(domain_a, len(dataset_A_test)))
log('Number of testing data from domain [{}]: {}'.format(domain_b, len(dataset_B_test)))
log('------------------------------------------------\n')

# ------------------------------- start training ------------------------------
if config.train and not config.getz:
    start_time = time.time()
    for epoch in range(0, config.epoch):

        average_loss = 0
        average_nums = 0

        network.train()
        for i, (batch_voxel, point_coord, point_value, _) in enumerate(dataloader_train):
            batch_voxel = batch_voxel.to(device)
            point_coord = point_coord.to(device)
            point_value = point_value.to(device)

            network.zero_grad()
            net_out = network(batch_voxel, point_coord)
            loss = torch.mean((net_out - point_value) ** 2)
            loss.backward()
            optimizer.step()

            average_loss += loss.item()
            average_nums += 1

        log("[%2d] Epoch: [%2d/%2d] Time: %4.4f, avg_loss: %.6f" %
            (config.sample_vox_size, epoch + 1, config.epoch, time.time() - start_time, average_loss / average_nums))

        # visualize training sample every 20 epochs
        if (epoch + 1) % 20 == 0:
            # domain A
            idx_a = np.random.randint(len(dataset_A_test))
            test_voxel, _, _, _ = dataset_A_test[idx_a]
            tester.test_during_train(network, test_voxel, os.path.join(sample_a_dir, 'epoch{}.ply'.format(epoch+1)))

            # domain B
            idx_b = np.random.randint(len(dataset_B_test))
            test_voxel, _, _, _ = dataset_B_test[idx_b]
            tester.test_during_train(network, test_voxel, os.path.join(sample_b_dir, 'epoch{}.ply'.format(epoch+1)))

        if (epoch + 1) % 100 == 0:
            # --------------- save model, write model name to checkpoint.txt --------------
            save_dir = os.path.join(checkpoint_dir, '3d_grid_model{}-{}.pth'.format(config.sample_vox_size, epoch))
            torch.save(network.state_dict(), save_dir)
            fout = open(checkpoint_txt, 'w')
            fout.write(save_dir + '\n')
            fout.close()

# test or getz
else:
    z_dim = 256
    batch_size = 1
    assert config.sample_vox_size == 64

    dataloader_A = DataLoader(dataset_A_train, batch_size=batch_size, shuffle=False, num_workers=4) if config.train else \
                   DataLoader(dataset_A_test, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_B = DataLoader(dataset_B_train, batch_size=batch_size, shuffle=False, num_workers=4) if config.train else \
                   DataLoader(dataset_B_test, batch_size=batch_size, shuffle=False, num_workers=4)

    network.eval()
    # extract feature vector
    if config.getz:
        print("===> save [{}] feature vector to hdf5 file for Domain [{}]".format(mode, domain_a))
        hdf5_path = os.path.join(checkpoint_dir, domain_a + '_{}_z.hdf5'.format(mode))
        shape_num = len(dataset_A_train) if config.train else len(dataset_A_test)
        dima = 64

        file_names = []
        zs = np.zeros((shape_num, 32, 2, 2, 2))
        voxels = np.zeros((shape_num, dima, dima, dima))
        for i, (batch_voxel, _, _, file_name) in enumerate(dataloader_A):
            batch_voxel = batch_voxel.to(device)
            with torch.no_grad():
                z_vector = network.encoder(batch_voxel)

            zs[i:i + 1] = z_vector.detach().cpu().numpy()
            voxels[i:i + 1] = batch_voxel.squeeze(0).detach().cpu().numpy()
            file_names.append(file_name[0])

        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset("zs", zs.shape, data=zs, dtype=np.float32)
        hdf5_file.create_dataset("file_names", [shape_num, 1], data=file_names, dtype=h5py.string_dtype(encoding='utf-8'), compression=9)
        hdf5_file.create_dataset("input_data", [shape_num, dima, dima, dima], data=voxels, dtype=np.uint8, compression=9)
        hdf5_file.close()

        print("===> save [{}] feature vector to hdf5 file for Domain [{}]".format(mode, domain_b))
        hdf5_path = os.path.join(checkpoint_dir, domain_b + '_{}_z.hdf5'.format(mode))
        shape_num = len(dataset_B_train) if config.train else len(dataset_B_test)

        file_names = []
        zs = np.zeros((shape_num, 32, 2, 2, 2))
        voxels = np.zeros((shape_num, dima, dima, dima))
        for i, (batch_voxel, _, _, file_name) in enumerate(dataloader_B):
            batch_voxel = batch_voxel.to(device)
            with torch.no_grad():
                z_vector = network.encoder(batch_voxel)

            zs[i:i + 1] = z_vector.detach().cpu().numpy()
            voxels[i:i + 1] = batch_voxel.squeeze(0).detach().cpu().numpy()
            file_names.append(file_name[0])

        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset("zs", zs.shape, data=zs, dtype=np.float32)
        hdf5_file.create_dataset("file_names", [shape_num, 1], data=file_names, dtype=h5py.string_dtype(encoding='utf-8'), compression=9)
        hdf5_file.create_dataset("input_data", [shape_num, dima, dima, dima], data=voxels, dtype=np.uint8, compression=9)
        hdf5_file.close()

    # reconstruction
    else:
        print("===> sample [{}] mesh and point clouds for Domain [{}] and [{}]".format(mode, domain_a, domain_b))
        for dataloader, sample_dir in [(dataloader_A, sample_a_dir), (dataloader_B, sample_b_dir)]:
            for i, (batch_voxel, _, _, file_name) in enumerate(dataloader):
                print([i + 1])
                batch_voxel = batch_voxel.to(device)
                tester.test_mesh_point(network, batch_voxel, os.path.join(sample_dir, file_name[0]))
                vertices, triangles = mcubes.marching_cubes(batch_voxel.squeeze(0).squeeze(0).detach().cpu().numpy(), 0.5)
                vertices = (vertices.astype(np.float32) - 0.5) / 64 - 0.5
                write_ply_triangle(os.path.join(sample_dir, file_name[0] + '.ply'), vertices, triangles)

                with torch.no_grad():
                    z = network.encoder(batch_voxel)
                    input_z = z[0] if config.multi_scale else z
                reconstructed_im = tester.test_point_as_image(network, input_z)
                ground_truth_im = tester.test_point_as_image(_, batch_voxel.squeeze(0).squeeze(0).detach().cpu().numpy(), 'voxel', scale=4)
                image = Image.fromarray(np.concatenate((ground_truth_im, reconstructed_im), axis=1)).convert('L')
                image.save(os.path.join(sample_dir, file_name[0] + '.png'))

