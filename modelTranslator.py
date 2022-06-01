import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Latent2DGridGenerator(nn.Module):
    def __init__(self, input_dim):
        super(Latent2DGridGenerator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim * 1, input_dim * 1, 1, stride=1, padding=0), nn.ReLU(inplace=True), nn.BatchNorm2d(input_dim * 1),
            nn.Conv2d(input_dim * 1, input_dim * 1, 1, stride=1, padding=0), nn.ReLU(inplace=True), nn.BatchNorm2d(input_dim * 1),
            nn.Conv2d(input_dim * 1, input_dim * 1, 1, stride=1, padding=0), nn.ReLU(inplace=True), nn.BatchNorm2d(input_dim * 1),
            nn.Conv2d(input_dim * 1, input_dim * 1, 1, stride=1, padding=0), nn.ReLU(inplace=True), nn.BatchNorm2d(input_dim * 1),
            nn.Conv2d(input_dim * 1, input_dim * 1, 1, stride=1, padding=0), nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        out = self.layers(inputs)

        return out


class Latent2DGridDiscriminator(nn.Module):
    def __init__(self, input_dim, df_dim=64):
        super(Latent2DGridDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, df_dim * 1, 1, stride=1, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(df_dim * 1, df_dim * 2, 1, stride=1, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(df_dim * 2, df_dim * 4, 1, stride=1, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(df_dim * 4, 1, 1, stride=1, padding=0),
        )

    def forward(self, inputs):
        logits = self.layers(inputs)
        out = torch.sigmoid(logits)

        return out, logits


class Latent3DGridGenerator(nn.Module):
    def __init__(self, input_dim):
        super(Latent3DGridGenerator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(input_dim * 1, input_dim * 1, 3, stride=1, padding=1), nn.ReLU(inplace=True), nn.BatchNorm3d(input_dim * 1),
            nn.Conv3d(input_dim * 1, input_dim * 1, 3, stride=1, padding=1), nn.ReLU(inplace=True), nn.BatchNorm3d(input_dim * 1),
            nn.Conv3d(input_dim * 1, input_dim * 1, 3, stride=1, padding=1), nn.ReLU(inplace=True), nn.BatchNorm3d(input_dim * 1),
            nn.Conv3d(input_dim * 1, input_dim * 1, 3, stride=1, padding=1), nn.ReLU(inplace=True), nn.BatchNorm3d(input_dim * 1),
            nn.Conv3d(input_dim * 1, input_dim * 1, 3, stride=1, padding=1), nn.ReLU(inplace=True), nn.BatchNorm3d(input_dim * 1),
            nn.Conv3d(input_dim * 1, input_dim * 1, 3, stride=1, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        out = self.layers(inputs)

        return out


class Latent3DGridDiscriminator(nn.Module):
    def __init__(self, input_dim, df_dim=32):
        super(Latent3DGridDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(input_dim, df_dim * 1, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(df_dim * 1, df_dim * 2, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(df_dim * 2, df_dim * 4, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(df_dim * 4, df_dim * 8, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(df_dim * 8, 1, 3, stride=1, padding=1),
        )

    def forward(self, inputs):
        logits = self.layers(inputs)
        out = torch.sigmoid(logits)

        return out, logits


if __name__ == '__main__':
    inputs = torch.randn(1, 64, 4, 4).cuda()
    g = Latent2DGridGenerator(input_dim=64).cuda()
    out = g(inputs)
    print(out.shape)
