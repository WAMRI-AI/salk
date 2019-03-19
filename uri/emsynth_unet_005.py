#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import vgg16_bn
import PIL
import imageio
from superres import *
from scipy.ndimage.interpolation import zoom as npzoom
from skimage.util import img_as_float32, img_as_ubyte
from skimage.measure import compare_ssim, compare_psnr

fastprogress.MAX_COLS = 80





# In[3]:


img_data = Path('/scratch/bpho/datasets/emsynth_003/')
model_path = Path('/scratch/bpho/models')


def get_src():
    hr_tifs = img_data/f'hr'
    lr_tifs = img_data/f'lr_up'

    def map_to_hr(x):
        hr_name = x.relative_to(lr_tifs)
        return hr_tifs/hr_name
    print(lr_tifs)
    src = (ImageImageList
            .from_folder(lr_tifs)
            .split_by_rand_pct()
            .label_from_func(map_to_hr))
    return src


def get_data(bs, size, noise=None, max_zoom=1.1):
    src = get_src()
    tfms = get_transforms(flip_vert=True, max_zoom=max_zoom)
    data = (src
            .transform(tfms, size=size)
            .transform_y(tfms, size=size)
            .databunch(bs=bs).normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data



@call_parse
def main(
        lr: Param("Learning rate", float)=1e-3,
        size: Param("Size", int)=128,
        wd: Param("weight decay", float)=1e-3,
        epochs: Param("Number of epochs", int)=5,
        bs: Param("Batch size", int)=64,
        max_zoom: Param("Max Zoom", float)=1.1,
        load_name: Param("load learner name", str)=None,
        save_name: Param("save learner name", str)="em_save",
        gpu:Param("GPU to run on", str)=None,
        ):
    "Distributed training of emdata."
    gpu = setup_distrib(gpu)
    if gpu is None: bs *= torch.cuda.device_count()

    print(f'lr: {lr}; size: {size};')
    data = get_data(bs, size, max_zoom=max_zoom)

    arch = models.resnet34
    wd = 1e-3
    learn = (unet_learner(data, arch, wd=wd,
                        loss_func=F.mse_loss,
                        metrics=superres_metrics,
                        blur=True, norm_type=NormType.Weight,
                        model_dir=model_path)
             .to_fp16(dynamic=True)
    )
    if not load_name is None: learn = learn.load(load_name)

    if gpu is None:       learn.to_parallel()
    elif num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai.launch`

    learn.freeze()
    learn.fit_one_cycle(epochs, lr, div_factor=10, pct_start=0.9)
    learn.unfreeze()
    learn.fit_one_cycle(epochs, slice(1e-5,lr), div_factor=10, pct_start=0.9)
    learn.save(save_name)


