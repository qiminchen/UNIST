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

## Datasets and pre-trained weights

## Usage
We provide instructions for training and testing 2D translation on `A-H` dataset and 3D translations on `chair-table` dataset, below instruction works for other domain pairs.
### 2D Experiments
**[Note]** You need to train autoencoding first, then translation.

**[First]** To train autoencoder, use the following command to train on images of resolution `128x128`.
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
