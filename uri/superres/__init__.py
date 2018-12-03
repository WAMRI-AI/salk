import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from fastai import Iterator, MSELossFlat, partial, Learner, tensor
from fastai.vision import ImageItemList, Image, pil2tensor, get_transforms
from fastai.layers import Lambda
from fastai.callbacks import SaveModelCallback
import PIL
import numpy as np
import pytorch_ssim as ssim


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
        self.label_cls = SuperResLabelList
        self.loss_func = MSELossFlat()
        self.create_func = open_grayscale


def icnr(x, scale, init=nn.init.kaiming_normal_):
    new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = init(subkernel)
    subkernel = subkernel.transpose(0, 1)
    subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                            subkernel.shape[1], -1)
    kernel = subkernel.repeat(1, 1, scale ** 2)
    transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)
    kernel = kernel.transpose(0, 1)
    return kernel


class Block(nn.Module):
    def __init__(self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res

class ICNRUpsample(nn.Module):
    def __init__(self,nf_in, nf_out, kernel_size, scale, wn):
        super().__init__()
        conv_shuffle = nn.Conv2d(nf_in, nf_out, kernel_size, padding=kernel_size//2)
        kernel = icnr(conv_shuffle.weight, scale=scale)
        conv_shuffle.weight.data.copy_(kernel);
        self.features = nn.Sequential(*[wn(conv_shuffle), nn.PixelShuffle(scale)])

    def forward(self, x):
        return self.features(x)

class WDSR(nn.Module):
    def __init__(self, scale, n_resblocks, n_feats, res_scale, n_colors_in=3, n_colors_out=1):
        super().__init__()

        # hyper-params
        kernel_size = 3
        act = nn.ReLU(True)
        wn = lambda x: torch.nn.utils.weight_norm(x)

        #mean, std = [.0020], [0.0060]] # imagenet_stats
        #self.rgb_mean = torch.FloatTensor(mean).view([1, n_colors_in, 1, 1])
        ##self.rgb_std = torch.FloatTensor(std).view([1, n_colors_in, 1, 1])

        # define head module
        head = []
        head.append(wn(nn.Conv2d(n_colors_in, n_feats,kernel_size,padding=kernel_size//2)))

        # define body module
        body = []
        for i in range(n_resblocks):
            body.append(
                Block(n_feats, kernel_size, act=act, res_scale=res_scale, wn=wn))

        # define tail module
        tail = []
        # convert from n_color_in to n_color_out
        #tail.append(wn(nn.Conv2d(n_feats, n_colors_out, kernel_size, padding=kernel_size//2)))


        out_feats = scale*scale*n_colors_out
        tail.append(ICNRUpsample(n_feats, out_feats, kernel_size, scale, wn))

        skip = []
        skip.append(wn(nn.Conv2d(n_colors_in, n_colors_out, kernel_size, padding=kernel_size//2)))
        skip.append(ICNRUpsample(n_colors_out, out_feats, 5, scale, wn))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        #mean = self.rgb_mean.to(x)
        #std = self.rgb_std.to(x)

        #x = (x - mean) / std
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        # x = x*std + mean

        return x

def edge_loss(input, target):
    k = tensor([
        [0.  ,-5/3,1],
        [-5/3,-5/3,1],
        [1.  ,1   ,1],
    ]).cuda().expand(1,1,3,3)/6
    return 100*F.mse_loss(F.conv2d(input, k), F.conv2d(target, k))

def fft_loss(pred, targs):
    bs = pred.shape[0]
    pred_fft = torch.rfft(pred, 2)
    targs_fft = torch.rfft(targs,2)
    return F.mse_loss(pred_fft.view(bs,-1), targs_fft.view(bs,-1))

ssim_loss = ssim.SSIM(mult=-1.)

def combo_edge_mse(input, target):
    return F.mse_loss(input, target) + edge_loss(input, target)

superres_metrics = [F.mse_loss, edge_loss,ssim.ssim, psnr]

class GrayImageItemList(ImageItemList):
    def open(self, fn): return open_grayscale(fn)

def get_data(src, bs, sz_lr, sz_hr, num_workers=12, test_folder=None, **kwargs):
    tfms = get_transforms(flip_vert=True)
    src = src.transform(tfms, size=sz_lr).transform_y(size=sz_hr)
    if test_folder:
        src = src.add_test_folder(test_folder, label=test_label)
    data = src.databunch(bs=bs, num_workers=num_workers, **kwargs)
    return data


def build_learner(model, bs, lr_sz, sz_hr, src, tfms=None, loss=F.mse_loss,
                  save=None, callback_fns=None, test_folder=None, **kwargs):
    data = get_data(src, bs, lr_sz, sz_hr, test_folder=test_folder, **kwargs)
    if callback_fns is None: callback_fns = []
    if save: callback_fns.append(partial(SaveModelCallback, name=f'{save}_best'))
    learn = Learner(data, model, loss_func=loss, metrics=superres_metrics, callback_fns=callback_fns)
    return learn


def batch_learn(model, bs, lr_sz, hr_sz, lr, epochs, src,
                tfms=None, load=None, save=None, plot=True,
                loss=F.mse_loss, callback_fns=None, test_folder=None, **kwargs):
    learn = build_learner(model, bs, lr_sz, hr_sz, src, tfms=tfms, loss=loss,
                          save=save, callback_fns=callback_fns,
                          test_folder=test_folder, **kwargs)
    if load:
        print(f'load: {load}')
        learn = learn.load(load)
    learn.fit_one_cycle(epochs, lr)
    if save:
        print(f'save: {save}')
        learn.save(save)
    if plot: learn.recorder.plot_losses()
    return learn