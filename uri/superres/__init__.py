import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from fastai import Iterator, MSELossFlat
from fastai.vision import ImageItemList, Image, pil2tensor
import PIL
import numpy as np

def conv(ni, nf, kernel_size=3, actn=True):
    layers = [nn.Conv2d(ni, nf, kernel_size, padding=kernel_size//2)]
    if actn: layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)

class ResSequential(nn.Module):
    def __init__(self, layers, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.m = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.m(x) * self.res_scale
        return x

def res_block(nf):
    return ResSequential(
        [conv(nf, nf), conv(nf, nf, actn=False)],
        0.1)

def upsample(ni, nf, scale):
    layers = []
    for i in range(int(math.log(scale,2))):
        layers += [conv(ni, nf*4), nn.PixelShuffle(2)]
    return nn.Sequential(*layers)

class SrResnet(nn.Module):
    def __init__(self, n_feats, n_res, n_colors, scale ):
        super().__init__()
        features = [conv(n_colors, n_feats)]
        for i in range(n_res): features.append(res_block(n_feats))
        features += [conv(n_feats,n_feats), upsample(n_feats, n_feats, scale),
                     nn.BatchNorm2d(n_feats),
                     conv(n_feats, n_colors, actn=False)]
        self.features = nn.Sequential(*features)

    def forward(self, x): return self.features(x)


def psnr(pred, targs):
    mse = F.mse_loss(pred, targs)
    return 20 * torch.log10(1./torch.sqrt(mse))

def psnr_loss(pred, targs):
    mse = F.mse_loss(pred, targs)
    return -20 * torch.log10(1./torch.sqrt(mse))

def open_grayscale(fn):
    x = PIL.Image.open(fn)
    return Image(pil2tensor(x,np.float32).div_(255)[0:1])

class SuperResLabelList(ImageItemList):
    def __init__(self, items:Iterator, **kwargs):
        super().__init__(items, **kwargs)
        self.loss_func,self.create_func = MSELossFlat(),open_grayscale

    def new(self, items, classes=None, **kwargs):
        return self.__class__(items, **kwargs)


class SuperResItemList(ImageItemList):
    def __post_init__(self):
        super().__post_init__()
        self._label_cls = SuperResLabelList
        self.loss_func = MSELossFlat()
        self.create_func = open_grayscale
