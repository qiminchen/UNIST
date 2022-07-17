# UNIST
Pytorch Implementation of [UNIST: Unpaired Neural Implicit Shape Translation Network](https://qiminchen.github.io/unist/), [Qimin Chen](https://qiminchen.github.io/), [Johannes Merz](), [Aditya Sanghi](https://www.autodesk.com/research/people/aditya-sanghi), [Hooman Shayani](https://www.autodesk.com/research/people/hooman-shayani), [Ali Mahdavi-Amiri](https://www.sfu.ca/~amahdavi/Home.html) and [Hao (Richard) Zhang](https://www.cs.sfu.ca/~haoz/), CVPR 2022

### [Paper](https://arxiv.org/abs/2112.05381)  |   [Video](https://youtu.be/FOfMNhDYA84)  |   [Project page](https://qiminchen.github.io/unist/)

<img src='img/teaser.svg' />

## Dependencies
Install required dependencies, run

```
conda env create -f environment.yml
```
Our code has been tested with Python 3.6, Pytorch 1.6.0, CUDA 10.1 and cuDNN 7.0 on Ubuntu 18.04.

## Datasets

Datasets can be found [here](https://drive.google.com/drive/folders/1ARC5NBTS3fWoGNo4YxkbUSrOPssU23Ei?usp=sharing). Description of each key in `hdf5` file can be found below. For custom datasets, please follow the below steps for preparation.

### 2D dataset

```
[key] - [description]
file_names - file name of each image
pixels - image in gray scale
points_128 - query points, (num_image, 16384, 2)
values_128 - inside / outside values of query points, (num_image, 16384, 1)
# Note that we don't employ progressive training in 2D experiments so you can ignore points_64 and values_64
```
We provide code of sampling query points **near/inside/outside** boundary with a ratio of `3:2:1` from a `256x256` image in `sampling.py`, you can adjust accordingly.

## Usage
We provide instructions for training and testing 2D translation on `A-H` dataset and 3D translations on `chair-table` dataset, below instruction works for other domain pairs.

### 2D Experiments
**[Note]** You need to train autoencoding first, then translation.

**[First]** To train autoencoder, use the following command to train on images of resolution `128^2`.
```
python run_2dae.py --train --epoch 800 --dataset A-H --sample_im_size 128
```
To test reconstruction, use the following command
```
python run_2dae.py  --dataset A-H --sample_dir outputs --sample_im_size 128
```
**[Second]** To train translation, first use the following command to extract feature grid
```
python run_2dae.py --train --getz --dataset A-H --sample_im_size 128  # training data
python run_2dae.py --getz --dataset A-H --sample_im_size 128          # testing data
```
then use the following command to train translation
```
python run_2dgridtranslator.py --train --epoch 1200 --dataset A-H --batch_size 128
```
To test translation, use the following command
```
python run_2dgridtranslator.py --dataset A-H --sample_dir outputs
```
### 3D Experiments
**[First]** To train autoencoder, use the following commands for progressive training.
```
python run_ae.py --train --epoch 300 --dataset chair-table --sample_vox_size 16
python run_ae.py --train --epoch 300 --dataset chair-table --sample_vox_size 32
python run_ae.py --train --epoch 600 --dataset chair-table --sample_vox_size 64
```
To test reconstruction (default on voxel of resolution `256^3`), use the following command
```
python run_ae.py --dataset chair-table --sample_dir outputs
```
**[Second]** To train translation, first use the following command to extract feature grid
```
python run_ae.py --train --getz --dataset chair-table  # training data
python run_ae.py --getz --dataset chair-table          # testing data
```
then use the following command to train translation
```
python run_3dgridtranslator.py --train --epoch 4800 --dataset chair-table --batch_size 128
```
To test translation, use the following command
```
python run_3dgridtranslator.py --dataset chair-table --sample_dir outputs
```

## Citation
Please cite our paper if you find this code or research relevant:

      @inproceedings{chen2022unist,
        title={UNIST: Unpaired Neural Implicit Shape Translation Network},
        author={Chen, Qimin and Merz, Johannes and Sanghi, Aditya and Shayani, Hooman and Mahdavi-Amiri, Ali and Zhang, Hao},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        year={2022}
      }
