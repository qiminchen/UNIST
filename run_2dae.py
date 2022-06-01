import os
import sys
import time
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
from collections import OrderedDict

from modelAE import GridAE2D, _CustomDataParallel
from dataset import DomainABDatasetFor2D

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Number of epoch to train [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00005, type=float, help="Learning rate [0.00005]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--lambda1", action="store", dest="lambda1", default=0.1, type=float, help="Scalar weight for sub feature vectors loss")
parser.add_argument("--dataset", action="store", dest="dataset", default="chair_table", help="The name of dataset domain A_B [chair_table]")
parser.add_argument("--batch_size", action="store", dest="batch_size", default=56, type=int, help="Batch size [24]")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="./checkpoint",
                    help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="./data", help="Root directory of dataset [data]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples",
                    help="Directory name to save the image samples [samples]")
parser.add_argument("--sample_im_size", action="store", dest="sample_im_size", default=128, type=int,
                    help="Image size for coarse-to-fine training [128]")
parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
parser.add_argument("--getz", action="store_true", dest="getz", default=False, help="True for getting latent codes [False]")
parser.add_argument("--iou", action="store_true", dest="iou", default=False, help="True for computing IOU [False]")
config = parser.parse_args()


# train  - python run_2dae.py --train --epoch 800 --dataset A-H --sample_im_size 128
# test   - python run_2dae.py  --dataset A-H --sample_dir outputs --sample_im_size 128
# trainz - python run_2dae.py --train --getz --dataset A-H --sample_im_size 128
# testz  - python run_2dae.py --getz --dataset A-H --sample_im_size 128
# train iou - python run_2dae.py --train --iou --dataset A-H --sample_im_size 128
# test iou - python run_2dae.py --iou --dataset A-H --sample_im_size 128

mode = 'train' if config.train else 'test'

# ---------------------------- domain A and domain B ----------------------------
domain_a = config.dataset.split('-')[0]  # chair
domain_b = config.dataset.split('-')[1]  # table

# ------------- create sample folder for test samples if not exist --------------
sample_a_dir = os.path.join(config.sample_dir, config.dataset, domain_a + '_{}_{}'.format(config.network_type, config.sample_im_size))
sample_b_dir = os.path.join(config.sample_dir, config.dataset, domain_b + '_{}_{}'.format(config.network_type, config.sample_im_size))
if not os.path.exists(sample_a_dir):
    os.makedirs(sample_a_dir)
if not os.path.exists(sample_b_dir):
    os.makedirs(sample_b_dir)

# ------------------- Create checkpoint folder if not exist ---------------------
checkpoint_dir = os.path.join(config.checkpoint_dir, config.dataset, '2d_grid_ae_64x2x2')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def log(strs):
    fp.write('%s\n' % strs)
    fp.flush()
    print(strs)


fp = open(os.path.join(checkpoint_dir, 'logs_{}_{}.txt'.format(mode, config.sample_im_size)), 'w')
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


log("===> Loading 2D Grid AE")
network = GridAE2D(ef_dim=16, z_dim=64, df_dim=128)
log("===> Number of parameters: {:,}".format(sum(p.numel() for p in network.parameters() if p.requires_grad)))
if torch.cuda.device_count() > 1:
    network = _CustomDataParallel(network)
network.to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))

# --------------------------- Load previous checkpoint --------------------------
checkpoint_txt = os.path.join(checkpoint_dir, 'checkpoint')
if os.path.exists(checkpoint_txt):
    fin = open(checkpoint_txt)
    model_dir = fin.readline().strip()
    fin.close()
    state_dict = torch.load(model_dir)
    try:
        network.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        network.load_state_dict(new_state_dict)
    log("===> Model {} load SUCCESS".format(model_dir))
else:
    log("===> Model {} load failed...".format(checkpoint_txt))

# -------------------------------- Load dataset --------------------------------
dataset_A_train = DomainABDatasetFor2D(config, domain=domain_a, mode='train')
dataset_B_train = DomainABDatasetFor2D(config, domain=domain_b, mode='train')
dataset_A_test = DomainABDatasetFor2D(config, domain=domain_a, mode='test')
dataset_B_test = DomainABDatasetFor2D(config, domain=domain_b, mode='test')
dataset_train = ConcatDataset([dataset_A_train, dataset_B_train])
dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=4)
log('\n------------------------------------------------')
log('Number of training data from domain [{}]: {}'.format(domain_a, len(dataset_A_train)))
log('Number of training data from domain [{}]: {}'.format(domain_b, len(dataset_B_train)))
log('Number of training data from both domain: {}'.format(len(dataset_train)))
log('Number of testing data from domain [{}]: {}'.format(domain_a, len(dataset_A_test)))
log('Number of testing data from domain [{}]: {}'.format(domain_b, len(dataset_B_test)))
log('------------------------------------------------\n')


# ------------------------- coordinate for testing -----------------------------


def get_test_coords(dima=256):
    ranges = np.arange(0, dima, 1, np.uint8)
    aux_x = np.ones([dima, dima], np.uint8) * np.expand_dims(ranges, axis=1)
    aux_y = np.ones([dima, dima], np.uint8) * np.expand_dims(ranges, axis=0)
    coord = np.hstack((aux_x.reshape(-1, 1), aux_y.reshape(-1, 1)))
    coord = (coord.astype(np.float32) + 0.5) / dima - 0.5
    return torch.from_numpy(coord).unsqueeze(dim=0)  # 1 x 256^2 x 2


dima = 256
coords = get_test_coords(dima=dima)
coords = coords.to(device)


# ------------------------------- start training ------------------------------
if config.train and not config.getz and not config.iou:
    start_time = time.time()
    for epoch in range(0, config.epoch):

        average_loss = 0
        average_nums = 0

        network.train()
        for i, (batch_pixel, point_coord, point_value, _, point_weight) in enumerate(dataloader_train):
            batch_pixel = batch_pixel.to(device)
            point_coord = point_coord.to(device)
            point_value = point_value.to(device)
            point_weight = point_weight.to(device)

            network.zero_grad()
            net_out = network(batch_pixel, point_coord)
            loss = torch.mean(point_weight * (net_out - point_value) ** 2)
            loss.backward()
            optimizer.step()

            average_loss += loss.item()
            average_nums += 1

        log(
            "Epoch: [%2d/%2d] Time: %4.4f, avg_loss: %.6f" % (epoch + 1, config.epoch, time.time() - start_time, average_loss / average_nums))

        # visualize training sample every epoch
        if (epoch + 1) % 1 == 0:
            network.eval()
            # domain A
            idx_a = np.random.randint(len(dataset_A_train))
            test_pixels, _, _, _, _ = dataset_A_train[idx_a]  # 1 x 256 x 256
            test_pixels = test_pixels.unsqueeze(dim=0).to(device)  # 1 x 1 x 256 x 256

            with torch.no_grad():
                pred_image = network(test_pixels, coords)

            images = test_pixels.squeeze(0).squeeze(0).detach().cpu().numpy() * 255
            for im in pred_image:
                im = np.reshape(im.detach().cpu().numpy(), (dima, dima))
                im[im < 0.5] = 0
                im[im >= 0.5] = 1
                images = np.concatenate((images, im * 255), axis=1)
            images = ImageOps.invert(Image.fromarray(images).convert('L'))
            images.save(os.path.join(sample_a_dir, 'epoch_{}_gt_pred.png'.format(epoch + 1)))

            # domain B
            idx_b = np.random.randint(len(dataset_B_train))
            test_pixels, _, _, _, _ = dataset_B_train[idx_b]
            test_pixels = test_pixels.unsqueeze(dim=0).to(device)

            with torch.no_grad():
                pred_image = network(test_pixels, coords)

            images = test_pixels.squeeze(0).squeeze(0).detach().cpu().numpy() * 255
            for im in pred_image:
                im = np.reshape(im.detach().cpu().numpy(), (dima, dima))
                im[im < 0.5] = 0
                im[im >= 0.5] = 1
                images = np.concatenate((images, im * 255), axis=1)
            images = ImageOps.invert(Image.fromarray(images).convert('L'))
            images.save(os.path.join(sample_b_dir, 'epoch_{}_gt_pred.png'.format(epoch + 1)))

        if (epoch + 1) % 100 == 0:
            # --------------- save model, write model name to checkpoint.txt --------------
            save_dir = os.path.join(checkpoint_dir, '2d_grid_model{}-{}.pth'.format(config.sample_im_size, epoch))
            torch.save(network.state_dict(), save_dir)
            fout = open(checkpoint_txt, 'w')
            fout.write(save_dir + '\n')
            fout.close()

        if (epoch + 1) in (400, 800):
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.5

    fp.close()

# test or getz or iou
else:
    z_dim = 256
    batch_size = 1
    assert config.sample_im_size == 128
    #
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

        file_names = []
        zs = np.zeros((shape_num, 64, 2, 2))
        pixels = np.zeros((shape_num, dima, dima))
        for i, (batch_pixel, _, _, file_name, _) in enumerate(dataloader_A):
            batch_pixel = batch_pixel.to(device)
            with torch.no_grad():
                z_vector = network.encoder(batch_pixel)

            zs[i:i + 1] = z_vector.squeeze(0).detach().cpu().numpy()
            pixels[i:i + 1] = batch_pixel.squeeze(0).detach().cpu().numpy()
            file_names.append(file_name[0])

        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset("zs", zs.shape, data=zs, dtype=np.float32)
        hdf5_file.create_dataset("file_names", [shape_num, 1], data=file_names, dtype=h5py.string_dtype(encoding='utf-8'), compression=9)
        hdf5_file.create_dataset("input_data", [shape_num, dima, dima], data=pixels, dtype=np.uint8, compression=9)
        hdf5_file.close()

        print("===> save [{}] feature vector to hdf5 file for Domain [{}]".format(mode, domain_b))
        hdf5_path = os.path.join(checkpoint_dir, domain_b + '_{}_z.hdf5'.format(mode))
        shape_num = len(dataset_B_train) if config.train else len(dataset_B_test)

        file_names = []
        zs = np.zeros((shape_num, 64, 2, 2))
        pixels = np.zeros((shape_num, dima, dima))
        for i, (batch_pixel, _, _, file_name, _) in enumerate(dataloader_B):
            batch_pixel = batch_pixel.to(device)
            with torch.no_grad():
                z_vector = network.encoder(batch_pixel)

            zs[i:i + 1] = z_vector.squeeze(0).detach().cpu().numpy()
            pixels[i:i + 1] = batch_pixel.squeeze(0).detach().cpu().numpy()
            file_names.append(file_name[0])

        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset("zs", zs.shape, data=zs, dtype=np.float32)
        hdf5_file.create_dataset("file_names", [shape_num, 1], data=file_names, dtype=h5py.string_dtype(encoding='utf-8'), compression=9)
        hdf5_file.create_dataset("input_data", [shape_num, dima, dima], data=pixels, dtype=np.uint8, compression=9)
        hdf5_file.close()

    elif config.iou:
        for dataloader, domain in [(dataloader_A, domain_a), (dataloader_B, domain_b)]:
            print("===> evaluate [{}] IOU for Domain [{}]".format(mode, domain))
            IOU = []
            MSE = []
            file_names = []
            for i, (batch_pixel, _, _, file_name, _) in enumerate(tqdm(dataloader, position=0)):
                batch_pixel = batch_pixel.to(device)

                with torch.no_grad():
                    pred_image = network(batch_pixel, coords)
                pred_image = np.reshape(pred_image[0].detach().cpu().numpy(), (dima, dima))
                pred_image[pred_image < 0.5] = 0
                pred_image[pred_image >= 0.5] = 1
                gt_image = batch_pixel.squeeze(0).squeeze(0).detach().cpu().numpy()
                pred_image = pred_image.astype(np.uint8)
                gt_image = gt_image.astype(np.uint8)

                iou = np.sum(gt_image & pred_image) / float(np.sum(gt_image | pred_image))
                mse = np.mean((gt_image - pred_image) ** 2, axis=(0, 1))
                IOU.append(iou)
                MSE.append(mse)
                file_names.append(file_name[0])

            # write to txt
            fout = open(os.path.join(checkpoint_dir, "{}_iou_{}.txt".format(domain, mode)), 'w')
            fout.write("mean IOU: " + str(np.mean(IOU)) + "\n")
            fout.write("mean MSE: " + str(np.mean(MSE)) + "\n")
            for i in range(len(IOU)):
                fout.write(file_names[i] + ": " + str(IOU[i]) + " | " + str(MSE[i]) + "\n")
            fout.close()

    # reconstruction
    else:
        for dataloader, sample_dir in [(dataloader_A, sample_a_dir), (dataloader_B, sample_b_dir)]:
            for i, (batch_pixel, _, _, file_name, _) in enumerate(dataloader):

                batch_pixel = batch_pixel.to(device)

                with torch.no_grad():
                    pred_images = network(batch_pixel, coords)
                images = batch_pixel.squeeze(0).squeeze(0).detach().cpu().numpy() * 255
                for im in pred_images[:1]:
                    im = np.reshape(im.detach().cpu().numpy(), (dima, dima))
                    im[im < 0.5] = 0
                    im[im >= 0.5] = 1
                    images = np.concatenate((images, im * 255), axis=1)
                images = ImageOps.invert(Image.fromarray(images).convert('L'))
                images.save(os.path.join(sample_dir, file_name[0]))
