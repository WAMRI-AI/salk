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

datasetname = 'emsynth_001'
data_path = Path('/scratch/bpho')
datasets = data_path/'datasets'
datasources = data_path/'datasources'
dataset = datasets/datasetname

em_data = datasources/'EM_manually_aquired_pairs_01242019'

hr = em_data/'aligned_hr'
lr = em_data/'aligned_lr'

hr_files = hr.glob('*.tif')
lr_files = lr.glob('*.tif')

def get_key(fn):
    return fn.stem[0:(fn.stem.find('Region')-1)]

hr_map = { get_key(fn): fn for fn in hr_files }
lr_map = { get_key(fn): fn for fn in lr_files }



# wipe dataset so we can create fresh
if dataset.exists(): shutil.rmtree(dataset)


hr_dest = dataset/'hr'
lr_dest = dataset/'lr'
lr_up_dest = dataset/'lr_up'
lr_real = dataset/'lr_real'


valid_pct = 0.2

img_keys = list(hr_map.keys())
random.shuffle(img_keys)
split_idx = int(valid_pct * len(img_keys))
train_keys = img_keys[:-split_idx]
valid_keys = img_keys[-split_idx:]


def em_crappify(x, scale=4):
    lvar = filters.gaussian(x, sigma=3)
    x = random_noise(x, mode='localvar', local_vars=lvar*0.05)
    x_down = npzoom(x, 1/scale, order=1)
    x_up = npzoom(x_down, scale, order=1)
    return x_down, x_up

def saveimg(img, fn):
    fn.parent.mkdir(parents=True, mode=0o775, exist_ok=True)
    img.save(fn)

mbar = master_bar([('train', train_keys),('valid', valid_keys)])
for subdir, keys in mbar:
    for k in progress_bar(keys, parent=mbar):
        hr_fn = hr_map[k]
        lr_fn = lr_map[k]
        hr_img, lr_real_img = PIL.Image.open(hr_fn), PIL.Image.open(lr_fn)
        img_data = np.array(hr_img)
        lr_down, lr_up = em_crappify(img_data)
        lr_down_img = PIL.Image.fromarray(img_as_ubyte(lr_down))
        lr_up_img = PIL.Image.fromarray(img_as_ubyte(lr_up))

        saveimg(hr_img, hr_dest/subdir/f'{k}.tif')
        saveimg(lr_up_img, lr_up_dest/subdir/f'{k}.tif')
        saveimg(lr_down_img, lr_dest/subdir/f'{k}.tif')
        if subdir == 'valid': saveimg(lr_real_img, lr_real/f'{k}.tif')
