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
from skimage.util import random_noise
from skimage import filters

datasetname = 'multiframe_001'
data_path = Path('/scratch/bpho')
datasets = data_path/'datasets'
datasources = data_path/'datasources'
dataset = datasets/datasetname

hr_path = dataset/'hr'
lr_path = dataset/'lr'
lr_up_path = dataset/'lr_up'

valid_pct = 0.2
train_files = []
valid_files = []


# wipe dataset so we can create fresh
if dataset.exists(): shutil.rmtree(dataset)



def new_crappify(x, scale=4):
    x = random_noise(x, mode='salt', amount=0.005)
    x = random_noise(x, mode='pepper', amount=0.005)
    lvar = filters.gaussian(x, sigma=5)
    x = random_noise(x, mode='localvar', local_vars=lvar*0.5)
    x_down = npzoom(x, 1/scale, order=1)
    x_up = npzoom(x_down, scale, order=1)
    return x_down, x_up

def crappify_range(img_data, num_lrs=2, scale=4):
    lrs_downs = []
    lrs_ups = []
    for j in range(num_lrs):
        lrs_down = []
        lrs_up = []
        for i in range(img_data.shape[0]):
            lr_down, lr_up = new_crappify(img_data[i], scale=scale)
            lrs_down.append(lr_down)
            lrs_up.append(lr_up)
        lrs_downs.append(lrs_down)
        lrs_ups.append(lrs_up)
    lr_imgs = np.array(lrs_downs).astype(np.float32)
    lr_up_imgs = np.array(lrs_ups).astype(np.float32)
    hr_img = img_data[len(img_data)//2].astype(np.float32)
    return lr_imgs, lr_up_imgs, hr_img


def czi_to_multiframe(czi_fn, hr_dir, lr_dir, lr_up_dir, wsize=3, max_scale=1.05, pbar=None):
    with czifile.CziFile(czi_fn) as czi_f:
        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        channels = proc_shape['C']
        depths = proc_shape['Z']
        times = proc_shape['T']
        x,y = proc_shape['X'], proc_shape['Y']
        data = czi_f.asarray()
        for channel in range(channels):
            img_max = None
            timerange = list(range(0,times-wsize+1))
            if len(timerange) >= wsize:
                for time_col in progress_bar(timerange, parent=pbar):
                    save_name = f'{czi_fn.stem}_{channel:02d}_{time_col:03d}.npy'.replace(' ','_')
                    idx = build_index(proc_axes, {'T': slice(time_col,time_col+wsize), 'C': channel, 'X':slice(0,x),'Y':slice(0,y)})
                    img = data[idx].astype(np.float)
                    img_max = img.max() * max_scale
                    img /= img_max
                    lr_imgs, lr_up_imgs, hr_img = crappify_range(img)
                    hr_dest = hr_dir/save_name
                    lr_dest = lr_dir/save_name
                    lr_up_dest = lr_up_dir/save_name
                    for dest in [hr_dest, lr_dest, lr_up_dest]:
                        dest.parent.mkdir(parents=True, mode=0o775, exist_ok=True)
                    np.save(hr_dest, hr_img)
                    np.save(lr_dest, lr_imgs)
                    np.save(lr_up_dest, lr_up_imgs)

mito_path = datasources/'MitoTracker_Red_FM_movie_data'
hr_mito = list(mito_path.glob('*920*.czi'))
lr_mito = list(mito_path.glob('*230*.czi'))

mito_train = [fn for fn in hr_mito]

neuron_path = datasources/'live_neuron_mito_timelapse_for_deep_learning'
two_channel = list(neuron_path.glob('*MTGreen*.czi'))
one_channel = [x for x in neuron_path.glob('*.czi') if x not in two_channel]

airyscan_path = datasources/'Airyscan_processed_data_from_the_server'
hr_airyscan = list(airyscan_path.glob('*.czi'))



for fn in hr_mito:
    if '03-Airyscan' in fn.stem: valid_files.append(fn)
    else: train_files.append(fn)


for lst in [hr_airyscan, one_channel, two_channel]:
    lst.sort()
    random.shuffle(lst)
    split_idx = int(valid_pct * len(lst))
    print(split_idx)
    valid_files += lst[-split_idx:]
    train_files += lst[:-split_idx]
    for subdir, file_list in [('train', train_files),('valid', valid_files)]:
        print(f'\n\ncopy, crappify and upsample {subdir} files\n\n')
        pbar = master_bar(file_list)
        for czi_fn in pbar:
            czi_to_multiframe(czi_fn, hr_path/subdir, lr_path/subdir, lr_up_path/subdir, pbar=pbar)
