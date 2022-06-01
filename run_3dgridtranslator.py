import os
import sys
import time
import mcubes
import argparse
import numpy as np
from PIL import Image, ImageOps
from utils import write_ply_triangle, sample_points_triangle, write_ply_point
from collections import OrderedDict

from dataset import LatentCodeDataset
from modelAE import GridAE3D, Tester
from modelTranslator import Latent3DGridDiscriminator, Latent3DGridGenerator

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Number of epoch to train [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.002, type=float, help="Learning rate [0.002]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--dataset", action="store", dest="dataset", default="chair_table", help="The name of dataset domain A_B [chair_table]")
parser.add_argument("--batch_size", action="store", dest="batch_size", default=16, type=int, help="Batch size [16]")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="./checkpoint",
                    help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="./data", help="Root directory of dataset [data]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples",
                    help="Directory name to save the image samples [samples]")
parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
parser.add_argument("--cycle_loss_weight", action="store", dest="cycle_loss_weight", default=20, type=float, help="Cycle loss weight")
parser.add_argument("--feature_loss_weight", action="store", dest="feature_loss_weight", default=100, type=float, help="Feature loss weight")
parser.add_argument("--lambda_gp", action="store", dest="lambda_gp", default=10, type=float, help="Penalty of WGAN-GP")
config = parser.parse_args()


# train - python run_3dgridtranslator.py --train --epoch 4800 --dataset chair-table --batch_size 128
# test  - python run_3dgridtranslator.py --dataset chair-table --sample_dir outputs

mode = 'train' if config.train else 'test'

# ---------------------------- domain A and domain B ----------------------------
domain_a = config.dataset.split('-')[0]  # chair
domain_b = config.dataset.split('-')[1]  # table

# ------------- create sample folder for test samples if not exist --------------
sample_a_dir = os.path.join(config.sample_dir, config.dataset, '{}_gan'.format(domain_a))
sample_b_dir = os.path.join(config.sample_dir, config.dataset, '{}_gan'.format(domain_b))
if not os.path.exists(sample_a_dir):
    os.makedirs(sample_a_dir)
if not os.path.exists(sample_b_dir):
    os.makedirs(sample_b_dir)

# --------------- Create checkpoint folder for GAN if not exist -----------------
checkpoint_dir_gan = os.path.join(config.checkpoint_dir, config.dataset, '3d_grid_gan_32x2x2x2')
checkpoint_txt_gan = os.path.join(checkpoint_dir_gan, 'checkpoint')
if not os.path.exists(checkpoint_dir_gan):
    os.makedirs(checkpoint_dir_gan)


def log(strs):
    fp.write('%s\n' % strs)
    fp.flush()
    print(strs)


fp = open(os.path.join(checkpoint_dir_gan, 'logs_{}.txt'.format(mode)), 'w')
s = 'python run_2dae.py'
for j in sys.argv:
    s += ' ' + j
log(s)

# ------------------------------ prepare dataset ------------------------------
checkpoint_dir_ae = os.path.join(config.checkpoint_dir, config.dataset, '3d_grid_ae_32x2x2x2')
checkpoint_txt_ae = os.path.join(checkpoint_dir_ae, 'checkpoint')
domain_a_hdf5 = os.path.join(checkpoint_dir_ae, domain_a + '_{}_z.hdf5'.format(mode))
domain_b_hdf5 = os.path.join(checkpoint_dir_ae, domain_b + '_{}_z.hdf5'.format(mode))
data_A = LatentCodeDataset(hdf5_path=domain_a_hdf5, init_shuffle=config.train)
data_B = LatentCodeDataset(hdf5_path=domain_b_hdf5, init_shuffle=config.train)
log('\n------------------------------------------------')
log('Number of {} data from domain [{}]: {}'.format(mode, domain_a, data_A.num_examples))
log('Number of {} data from domain [{}]: {}'.format(mode, domain_b, data_B.num_examples))
log('------------------------------------------------\n')

# ------------------------------- Build network --------------------------------
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

generator_AB2B = Latent3DGridGenerator(input_dim=32).to(device)
generator_AB2A = Latent3DGridGenerator(input_dim=32).to(device)
discriminator_B = Latent3DGridDiscriminator(input_dim=32).to(device)  # AB -> B
discriminator_A = Latent3DGridDiscriminator(input_dim=32).to(device)  # AB -> A

optimizer_d_B = torch.optim.Adam(discriminator_B.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
optimizer_d_A = torch.optim.Adam(discriminator_A.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
optimizer_g = torch.optim.Adam(list(generator_AB2B.parameters())+list(generator_AB2A.parameters()), lr=config.learning_rate,
                               betas=(config.beta1, 0.999))

# load AE for decoding
ae = GridAE3D(ef_dim=16, z_dim=32, df_dim=128).to(device)
tester = Tester(device)
if os.path.exists(checkpoint_txt_ae):
    fin = open(checkpoint_txt_ae)
    model_dir = fin.readline().strip()
    fin.close()
    state_dict = torch.load(model_dir)
    try:
        ae.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        ae.load_state_dict(new_state_dict)
    log("===> Model {} load SUCCESS".format(model_dir))
    log('------------------------------------------------\n')
else:
    log("===> Model {} load failed...".format(checkpoint_txt_ae))
    log('------------------------------------------------\n')


# ----------------------- start training or testing ----------------------------
if config.train:
    start_time = time.time()
    discriminator_steps=5
    generator_steps=1
    log_lr = config.learning_rate

    for epoch in range(0, config.epoch):
        data_A.shuffle_data()
        data_B.shuffle_data()
        num_examples = min(data_A.num_examples, data_B.num_examples)
        batch_size = config.batch_size
        num_batches = int(num_examples / batch_size)
        iteration_per_epoch = int(num_batches / max(discriminator_steps, generator_steps))

        epoch_loss_d_A = 0.
        epoch_loss_d_B = 0.
        epoch_loss_g_A2B = 0.
        epoch_loss_g_B2A = 0.
        epoch_loss_feature_B2B = 0.
        epoch_loss_feature_A2A = 0.
        epoch_loss_c_A2B2A = 0.
        epoch_loss_c_B2A2B = 0.
        epoch_gradients_penalty_A = 0.
        epoch_gradients_penalty_B = 0.

        ae.eval()
        generator_AB2B.train()
        generator_AB2A.train()
        discriminator_B.train()
        discriminator_A.train()
        for _ in range(iteration_per_epoch):

            # train discriminator
            for _ in range(discriminator_steps):

                discriminator_A.zero_grad()
                discriminator_B.zero_grad()

                real_A_z, _, _ = data_A.next_batch(batch_size)
                real_B_z, _, _ = data_B.next_batch(batch_size)
                input_A_z, _, _ = data_A.next_batch(batch_size)
                input_B_z, _, _ = data_B.next_batch(batch_size)
                real_A_z = real_A_z.to(device)
                real_B_z = real_B_z.to(device)
                input_A_z = input_A_z.to(device)
                input_B_z = input_B_z.to(device)

                # A -> B and B -> A
                generator_A2B_z = generator_AB2B(input_A_z)
                generator_B2A_z = generator_AB2A(input_B_z)

                # ------------ AB -> B ------------
                _, real_logit_B_z = discriminator_B(real_B_z)
                _, synthetic_logit_A2B_z = discriminator_B(generator_A2B_z)
                loss_d_B_z = torch.mean(synthetic_logit_A2B_z) - torch.mean(real_logit_B_z)
                # Compute gradient penalty
                alpha = torch.rand(batch_size, 1, 1, 1, 1)
                alpha = alpha.expand(real_B_z.size()).to(device)
                interpolates = alpha * real_B_z + (1 - alpha) * generator_A2B_z
                interpolates = Variable(interpolates, requires_grad=True)
                _, d_interpolates = discriminator_B(interpolates)
                gradients = torch_grad(outputs=d_interpolates, inputs=interpolates,
                                       grad_outputs=torch.ones(d_interpolates.size()).to(device),
                                       create_graph=True, retain_graph=True)[0]
                # gradients - (B, 32, 2, 2, 2), norm over all but 1st dim
                gradients = gradients.view(gradients.size(0), -1)
                gradients_penalty_B = torch.mean((torch.norm(gradients, p=2, dim=1) - 1) ** 2) * config.lambda_gp

                loss_d_B = loss_d_B_z + gradients_penalty_B
                loss_d_B.backward()
                optimizer_d_B.step()

                # ------------ AB -> A ------------
                _, real_logit_A_z = discriminator_A(real_A_z)
                _, synthetic_logit_B2A_z = discriminator_A(generator_B2A_z)
                loss_d_A_z = torch.mean(synthetic_logit_B2A_z) - torch.mean(real_logit_A_z)
                # Compute gradient penalty
                alpha = torch.rand(batch_size, 1, 1, 1, 1)
                alpha = alpha.expand(real_A_z.size()).to(device)
                interpolates = alpha * real_A_z + (1 - alpha) * generator_B2A_z
                interpolates = Variable(interpolates, requires_grad=True)
                _, d_interpolates = discriminator_A(interpolates)
                gradients = torch_grad(outputs=d_interpolates, inputs=interpolates,
                                       grad_outputs=torch.ones(d_interpolates.size()).to(device),
                                       create_graph=True, retain_graph=True)[0]
                # gradients - (B, 32, 2, 2, 2), norm over all but 1st dim
                gradients = gradients.view(gradients.size(0), -1)
                gradients_penalty_A = torch.mean((torch.norm(gradients, p=2, dim=1) - 1) ** 2) * config.lambda_gp

                loss_d_A = loss_d_A_z + gradients_penalty_A
                loss_d_A.backward()
                optimizer_d_A.step()

            # train generator
            for _ in range(generator_steps):

                generator_AB2B.zero_grad()
                generator_AB2A.zero_grad()

                input_A_z, _, _ = data_A.next_batch(batch_size)
                input_B_z, _, _ = data_B.next_batch(batch_size)
                input_A_z = input_A_z.to(device)
                input_B_z = input_B_z.to(device)

                # A -> B and B -> B / B -> A and A -> A / A -> B -> A and B -> A -> B
                generator_A2B_z = generator_AB2B(input_A_z)
                generator_B2B_z = generator_AB2B(input_B_z)
                generator_B2A_z = generator_AB2A(input_B_z)
                generator_A2A_z = generator_AB2A(input_A_z)
                generator_A2B2A_z = generator_AB2A(generator_A2B_z)
                generator_B2A2B_z = generator_AB2B(generator_B2A_z)

                # ------------ generator loss --------------
                _, synthetic_logit_A2B_z = discriminator_B(generator_A2B_z)
                _, synthetic_logit_B2A_z = discriminator_A(generator_B2A_z)
                loss_g_A2B = -torch.mean(synthetic_logit_A2B_z)
                loss_g_B2A = -torch.mean(synthetic_logit_B2A_z)

                # ---------- feature preserving loss -------
                loss_feature_A2A = torch.mean(torch.abs(input_A_z - generator_A2A_z)) * config.feature_loss_weight
                loss_feature_B2B = torch.mean(torch.abs(input_B_z - generator_B2B_z)) * config.feature_loss_weight

                # ----------- cycle loss -------------
                loss_c_A2B2A = torch.mean(torch.abs(input_A_z - generator_A2B2A_z)) * config.cycle_loss_weight
                loss_c_B2A2B = torch.mean(torch.abs(input_B_z - generator_B2A2B_z)) * config.cycle_loss_weight

                loss_g = loss_g_A2B + loss_g_B2A + loss_feature_A2A + loss_feature_B2B + loss_c_A2B2A + loss_c_B2A2B
                loss_g.backward()
                optimizer_g.step()

            # accumulate loss for log
            epoch_loss_d_B += loss_d_B.item()
            epoch_loss_g_A2B += loss_g_A2B.item()
            epoch_loss_feature_B2B += loss_feature_B2B.item()

            epoch_loss_d_A += loss_d_A.item()
            epoch_loss_g_B2A += loss_g_B2A.item()
            epoch_loss_feature_A2A += loss_feature_A2A.item()

            epoch_loss_c_A2B2A += loss_c_A2B2A.item()
            epoch_loss_c_B2A2B += loss_c_B2A2B.item()

            epoch_gradients_penalty_A += gradients_penalty_A.item()
            epoch_gradients_penalty_B += gradients_penalty_B.item()

        log("[%.5f][%3d/%3d] Time: %07.2f, Loss - d_B:% .4f, g_A2B:% .4f, f_B2B:% .4f | d_A:% .4f, g_B2A:% .4f, f_A2A:% .4f | "
            "c_A2B2A:% .4f, c_B2A2B:% .4f, grad_A:% .4f, grad_B:% .4f"%
            (log_lr, epoch+1, config.epoch, time.time()-start_time,
             epoch_loss_d_B/iteration_per_epoch, epoch_loss_g_A2B/iteration_per_epoch, epoch_loss_feature_B2B/iteration_per_epoch,
             epoch_loss_d_A/iteration_per_epoch, epoch_loss_g_B2A/iteration_per_epoch, epoch_loss_feature_A2A/iteration_per_epoch,
             epoch_loss_c_A2B2A/iteration_per_epoch, epoch_loss_c_B2A2B/iteration_per_epoch,
             epoch_gradients_penalty_A/iteration_per_epoch, epoch_gradients_penalty_B/iteration_per_epoch))

        # adjust learning rate
        if (epoch + 1) % 100 == 0:
            for optimizer in [optimizer_d_B, optimizer_d_A, optimizer_g]:
                for g in optimizer.param_groups:
                    g['lr'] = max(g['lr'] * 0.5, 5e-4)
                    log_lr = g['lr']

        # visualize training sample
        if (epoch + 1) % 100 == 0:
            generator_AB2B.eval()
            generator_AB2A.eval()
            np.random.seed(7)
            A_idx = np.random.randint(data_A.num_examples)
            B_idx = np.random.randint(data_B.num_examples)
            input_A_z, A_name, A_voxel = data_A.get_one_example(idx=A_idx)
            input_B_z, B_name, B_voxel = data_B.get_one_example(idx=B_idx)
            input_A_z = input_A_z.to(device)
            input_B_z = input_B_z.to(device)

            # visualize original domain A and domain A -> B
            with torch.no_grad():
                A2B_z = generator_AB2B(input_A_z)
            name_A = os.path.join(sample_a_dir, 'epoch{}-recons'.format(epoch + 1))
            name_A2B = os.path.join(sample_a_dir, 'epoch{}-trans'.format(epoch + 1))
            tester.test_mesh_point(ae, input_A_z, name_A, input_type='z')
            tester.test_mesh_point(ae, A2B_z, name_A2B, input_type='z')
            ground_truth_im = tester.test_point_as_image(ae, A_voxel[0], input_type='voxel', scale=4)
            reconstructed_im = tester.test_point_as_image(ae, input_A_z)
            translated_im = tester.test_point_as_image(ae, A2B_z)
            image = Image.fromarray(np.concatenate((ground_truth_im, reconstructed_im, translated_im), axis=1)).convert('L')
            image.save(os.path.join(sample_a_dir, 'epoch{}-points.png'.format(epoch + 1)))

            # visualize original domain B and domain B -> A
            with torch.no_grad():
                B2A_z = generator_AB2A(input_B_z)
            name_B = os.path.join(sample_b_dir, 'epoch{}-recons'.format(epoch + 1))
            name_B2A = os.path.join(sample_b_dir, 'epoch{}-trans'.format(epoch + 1))
            tester.test_mesh_point(ae, input_B_z, name_B, input_type='z')
            tester.test_mesh_point(ae, B2A_z, name_B2A, input_type='z')
            ground_truth_im = tester.test_point_as_image(ae, B_voxel[0], input_type='voxel', scale=4)
            reconstructed_im = tester.test_point_as_image(ae, input_B_z)
            translated_im = tester.test_point_as_image(ae, B2A_z)
            image = Image.fromarray(np.concatenate((ground_truth_im, reconstructed_im, translated_im), axis=1)).convert('L')
            image.save(os.path.join(sample_b_dir, 'epoch{}-points.png'.format(epoch + 1)))

        # visualize training sample
        if (epoch + 1) % 600 == 0:
            save_dir = os.path.join(checkpoint_dir_gan, 'gan_model_{}.pth'.format(epoch))
            torch.save({'generator_AB2B': generator_AB2B.state_dict(),
                        'generator_AB2A': generator_AB2A.state_dict(),
                        'discriminator_B': discriminator_B.state_dict(),
                        'discriminator_A': discriminator_A.state_dict()}, save_dir)
            fout = open(checkpoint_txt_gan, 'w')
            fout.write(save_dir + '\n')
            fout.close()

else:
    sampling_threshold = 0.5
    tester_iou = Tester(device, cell_grid_size=1, frame_grid_size=64)
    if os.path.exists(checkpoint_txt_gan):
        fin = open(checkpoint_txt_gan)
        model_dir = fin.readline().strip()
        fin.close()
        state_dict = torch.load(model_dir)
        generator_AB2A.load_state_dict(state_dict['generator_AB2A'])
        generator_AB2B.load_state_dict(state_dict['generator_AB2B'])
        discriminator_B.load_state_dict(state_dict['discriminator_B'])
        print("===> Model {} load SUCCESS".format(model_dir))
        print('------------------------------------------------\n')
    else:
        print("===> Model {} load failed...".format(checkpoint_txt_gan))
        print('------------------------------------------------\n')

    # training images for retrieval
    data_A_train = LatentCodeDataset(hdf5_path=domain_a_hdf5.replace('test', 'train'), init_shuffle=False)
    data_B_train = LatentCodeDataset(hdf5_path=domain_b_hdf5.replace('test', 'train'), init_shuffle=False)
    A_voxels_train = data_A_train.input_data
    B_voxels_train = data_B_train.input_data
    A_zs_train = data_A_train.zs_vector
    B_zs_train = data_B_train.zs_vector
    A_num_example = data_A_train.num_examples
    B_num_example = data_B_train.num_examples

    generator_AB2A.eval()
    generator_AB2B.eval()

    for i in range(data_A.num_examples):
        input_A_z, A_name, A_voxel = data_A.get_one_example(idx=i)
        input_A_z = input_A_z.to(device)

        with torch.no_grad():
            A2B_z = generator_AB2B(input_A_z)
        name_A = os.path.join(sample_a_dir, '{}_recons'.format(A_name[0]))
        name_A2B = os.path.join(sample_a_dir, '{}_trans'.format(A_name[0]))

        # 1 save reconstructed and translated mesh
        tester.test_mesh_point(ae, input_A_z, name_A, input_type='z')
        tester.test_mesh_point(ae, A2B_z, name_A2B, input_type='z')
        # tester_iou.test_mesh_point(ae, A2B_z, name_A2B, input_type='z')  # for output sampled at 64^3

        # 2. save reconstructed and translated point clouds as png
        # ground_truth_im = tester.test_point_as_image(ae, A_voxel[0], input_type='voxel', scale=4)
        # reconstructed_im = tester.test_point_as_image(ae, input_A_z)
        # translated_im = tester.test_point_as_image(ae, A2B_z)
        # image = Image.fromarray(np.concatenate((ground_truth_im, reconstructed_im, translated_im), axis=1)).convert('L')
        # image.save(os.path.join(sample_a_dir, '{}.png'.format(A_name[0])))

        # 3. save input point cloud / mesh
        # vertices, triangles = mcubes.marching_cubes(A_voxel[0], sampling_threshold)
        # vertices = (vertices.astype(np.float32) - 0.5) / 64 - 0.5
        # sampled_points_normals = sample_points_triangle(vertices, triangles, num_of_points=2048)
        # np.random.shuffle(sampled_points_normals)
        # write_ply_point(os.path.join(sample_a_dir, A_name[0] + '_input_pcd.ply'), sampled_points_normals)
        # write_ply_triangle(os.path.join(sample_a_dir, A_name[0] + '_input_vox.ply'), vertices, triangles)

        # 4.1.1 retrieval from training set to [input]      based on IOU of voxel
        # A_voxels = np.repeat(A_voxel, B_num_example, axis=0).astype(np.uint8)
        # iou = np.sum(A_voxels & B_voxels_train, axis=(1, 2, 3)) / np.sum(A_voxels | B_voxels_train, axis=(1, 2, 3))
        # closest_to_input_iou = B_voxels_train[np.argmax(iou)]
        # vertices, triangles = mcubes.marching_cubes(closest_to_input_iou, sampling_threshold)
        # vertices = (vertices.astype(np.float32) - 0.5) / 64 - 0.5
        # write_ply_triangle(os.path.join(sample_a_dir, A_name[0] + '_closest_to_input_iou.ply'), vertices, triangles)

        # 4.1.2 retrieval from training set to [translated] based on IOU of voxel
        # A2B_voxel = tester_iou.z2voxel(ae, A2B_z)
        # A2B_voxels = np.repeat(A2B_voxel[np.newaxis, 1:-1, 1:-1, 1:-1], B_num_example, axis=0).astype(np.uint8)
        # iou = np.sum(A2B_voxels & B_voxels_train, axis=(1, 2, 3)) / np.sum(A2B_voxels | B_voxels_train, axis=(1, 2, 3))
        # closest_to_trans_iou = B_voxels_train[np.argmax(iou)]
        # vertices, triangles = mcubes.marching_cubes(closest_to_trans_iou, sampling_threshold)
        # vertices = (vertices.astype(np.float32) - 0.5) / 64 - 0.5
        # write_ply_triangle(os.path.join(sample_a_dir, A_name[0] + '_closest_to_trans_iou.ply'), vertices, triangles)

        # 4.2.1 retrieval from training set to [translated] based on MSE of z
        # A2B_zs = np.repeat(A2B_z.detach().cpu().numpy(), B_num_example, axis=0)
        # mse_z = np.mean((A2B_zs - B_zs_train) ** 2, axis=(1, 2, 3, 4))
        # closest_to_translated_mse_z = B_voxels_train[np.argmin(mse_z)]
        # vertices, triangles = mcubes.marching_cubes(closest_to_translated_mse_z, sampling_threshold)
        # vertices = (vertices.astype(np.float32) - 0.5) / 64 - 0.5
        # write_ply_triangle(os.path.join(sample_a_dir, A_name[0] + '_closest_to_trans_msez.ply'), vertices, triangles)
        # exit()

    for i in range(data_B.num_examples):
        input_B_z, B_name, B_voxel = data_B.get_one_example(idx=i)
        input_B_z = input_B_z.to(device)

        with torch.no_grad():
            B2A_z = generator_AB2A(input_B_z)
        name_B = os.path.join(sample_b_dir, '{}_recons'.format(B_name[0]))
        name_B2A = os.path.join(sample_b_dir, '{}_trans'.format(B_name[0]))

        # 1 save reconstructed and translated mesh
        tester.test_mesh_point(ae, input_B_z, name_B, input_type='z')
        tester.test_mesh_point(ae, B2A_z, name_B2A, input_type='z')
        # tester_iou.test_mesh_point(ae, B2A_z, name_B2A, input_type='z', save_mesh=False, save_point=True)

        # 2. save reconstructed and translated point clouds as png
        # ground_truth_im = tester.test_point_as_image(ae, B_voxel[0], input_type='voxel', scale=4)
        # reconstructed_im = tester.test_point_as_image(ae, input_B_z)
        # translated_im = tester.test_point_as_image(ae, B2A_z)
        # image = Image.fromarray(np.concatenate((ground_truth_im, reconstructed_im, translated_im), axis=1)).convert('L')
        # image.save(os.path.join(sample_b_dir, '{}.png'.format(B_name[0])))

        # 3. save input point cloud / mesh
        # vertices, triangles = mcubes.marching_cubes(B_voxel[0], sampling_threshold)
        # vertices = (vertices.astype(np.float32) - 0.5) / 64 - 0.5
        # sampled_points_normals = sample_points_triangle(vertices, triangles, num_of_points=2048)
        # np.random.shuffle(sampled_points_normals)
        # write_ply_point(os.path.join(sample_b_dir, B_name[0] + '_input_pcd.ply'), sampled_points_normals)
        # write_ply_triangle(os.path.join(sample_b_dir, B_name[0] + '_input_vox.ply'), vertices, triangles)

        # 4.1.1 retrieval from training set to [input]      based on IOU of voxel
        # B_voxels = np.repeat(B_voxel, A_num_example, axis=0).astype(np.uint8)
        # iou = np.sum(B_voxels & A_voxels_train, axis=(1, 2, 3)) / np.sum(B_voxels | A_voxels_train, axis=(1, 2, 3))
        # closest_to_input_iou = A_voxels_train[np.argmax(iou)]
        # vertices, triangles = mcubes.marching_cubes(closest_to_input_iou, sampling_threshold)
        # vertices = (vertices.astype(np.float32) - 0.5) / 64 - 0.5
        # write_ply_triangle(os.path.join(sample_b_dir, B_name[0] + '_closest_to_input_iou.ply'), vertices, triangles)

        # 4.1.2 retrieval from training set to [translated] based on IOU of voxel
        # B2A_voxel = tester_iou.z2voxel(ae, B2A_z)
        # B2A_voxels = np.repeat(B2A_voxel[np.newaxis, 1:-1, 1:-1, 1:-1], A_num_example, axis=0).astype(np.uint8)
        # iou = np.sum(B2A_voxels & A_voxels_train, axis=(1, 2, 3)) / np.sum(B2A_voxels | A_voxels_train, axis=(1, 2, 3))
        # closest_to_trans_iou = A_voxels_train[np.argmax(iou)]
        # vertices, triangles = mcubes.marching_cubes(closest_to_trans_iou, sampling_threshold)
        # vertices = (vertices.astype(np.float32) - 0.5) / 64 - 0.5
        # write_ply_triangle(os.path.join(sample_b_dir, B_name[0] + '_closest_to_trans_iou.ply'), vertices, triangles)

        # 4.2.1 retrieval from training set to [translated] based on MSE of z
        # B2A_zs = np.repeat(B2A_z.detach().cpu().numpy(), A_num_example, axis=0)
        # mse_z = np.mean((B2A_zs - A_zs_train) ** 2, axis=(1, 2, 3, 4))
        # closest_to_translated_mse_z = A_voxels_train[np.argmin(mse_z)]
        # vertices, triangles = mcubes.marching_cubes(closest_to_translated_mse_z, sampling_threshold)
        # vertices = (vertices.astype(np.float32) - 0.5) / 64 - 0.5
        # write_ply_triangle(os.path.join(sample_b_dir, B_name[0] + '_closest_to_trans_msez.ply'), vertices, triangles)
        # exit()
