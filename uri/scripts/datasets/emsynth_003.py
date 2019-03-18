from pathlib import Path
import shutil
import random
import os
from fastprogress import progress_bar, master_bar
from scipy.ndimage.interpolation import zoom as npzoom
import PIL.Image
import numpy as np
import czifile
from pdb import set_trace
from superres import get_czi_shape_info, build_index
from skimage.util import random_noise, img_as_ubyte
from skimage import filters
import PIL

datasetname = 'emsynth_003'
data_path = Path('/scratch/bpho')
datasets = data_path/'datasets'
datasources = data_path/'datasources'
dataset = datasets/datasetname

em_data = datasources/'em_tacc'

hr = em_data/'hr_2'

hr_files = hr.glob('*.tif')

# wipe dataset so we can create fresh
if dataset.exists(): shutil.rmtree(dataset)


hr_dest = dataset/'hr'
lr_dest = dataset/'lr'
lr_up_dest = dataset/'lr_up'


num_samples = 100000
img_files = np.random.choice(list(hr_files), num_samples, replace=False)

def em_crappify(x, scale=4):
    lvar = filters.gaussian(x, sigma=3)
    x = random_noise(x, mode='localvar', local_vars=lvar*0.05)
    x_down = npzoom(x, 1/scale, order=1)
    x_up = npzoom(x_down, scale, order=1)
    return x_down, x_up

def saveimg(img, fn):
    fn.parent.mkdir(parents=True, mode=0o775, exist_ok=True)
    img.save(fn)

for hr_fn in progress_bar(img_files):
    try:
        hr_img = PIL.Image.open(hr_fn)
        img_data = np.array(hr_img)[:,:,0]
        lr_down, lr_up = em_crappify(img_data)
        lr_down_img = PIL.Image.fromarray(img_as_ubyte(lr_down))
        lr_up_img = PIL.Image.fromarray(img_as_ubyte(lr_up))
        k = hr_fn.stem
        saveimg(hr_img, hr_dest/f'{k}.tif')
        saveimg(lr_up_img, lr_up_dest/f'{k}.tif')
        saveimg(lr_down_img, lr_dest/f'{k}.tif')
    except Exception as exc:
        print('oops')
        print(exc)
