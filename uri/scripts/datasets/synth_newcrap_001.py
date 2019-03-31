from pathlib import Path
import shutil
import random
import os
from fastprogress import progress_bar
from scipy.ndimage.interpolation import zoom as npzoom
from fastprogress import progress_bar, master_bar
import PIL.Image
import numpy as np
import czifile

from superres import get_czi_shape_info, build_index
from skimage.util import random_noise, img_as_ubyte
from skimage import filters
from pdb import set_trace

data_path = Path('/scratch/bpho')
sources = data_path/'datasources'

dname = 'synth_newcrap_001'
dpath = data_path/'datasets'/dname
if dpath.exists(): shutil.rmtree(dpath)

hr_path = dpath/'hr'
lr_path = dpath/'lr'
lr_up_path = dpath/'lr_up'

valid_pct = 0.2

# unpaired images (mito, fromserver, neuron)
# mito movies
mito_path = sources/'MitoTracker_Red_FM_movie_data'
hr_mito = list(mito_path.glob('*920*.czi'))
lr_mito = list(mito_path.glob('*230*.czi'))

mito_train = [fn for fn in hr_mito]


# neuron movies
neuron_path = sources/'live_neuron_mito_timelapse_for_deep_learning'
two_channel = list(neuron_path.glob('*MTGreen*.czi'))
one_channel = [x for x in neuron_path.glob('*.czi') if x not in two_channel]


# airyscan from server
airyscan_path = sources/'Airyscan_processed_data_from_the_server'
hr_airyscan = list(airyscan_path.glob('*.czi'))

train_files = []
valid_files = []


for fn in progress_bar(hr_mito):
    if '03-Airyscan' in fn.stem: valid_files.append(fn)
    else: train_files.append(fn)



def new_crappify(x, scale=4):
    x = random_noise(x, mode='salt', amount=0.005)
    x = random_noise(x, mode='pepper', amount=0.005)
    lvar = filters.gaussian(x, sigma=5)
    x = random_noise(x, mode='localvar', local_vars=lvar*0.5)
    x_down = npzoom(x, 1/scale, order=1)
    x_up = npzoom(x_down, scale, order=1)
    return x_down, x_up

def new_crappify_movie_to_tifs(czi_fn, hr_dir, lr_dir, lr_up_dir, base_name, max_scale=1.05):
    hr_dir, lr_dir, lr_up_dir = Path(hr_dir), Path(lr_dir), Path(lr_up_dir)
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    lr_up_dir.mkdir(parents=True, exist_ok=True)
    with czifile.CziFile(czi_fn) as czi_f:
        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        channels = proc_shape['C']
        depths = proc_shape['Z']
        times = proc_shape['T']
        x,y = proc_shape['X'], proc_shape['Y']
        data = czi_f.asarray()
        for channel in range(channels):
            for depth in range(depths):
                img_max = None
                for time_col in range(times):
                    try:
                        save_fn = f'{base_name}_{channel:02d}_{depth:03d}_{time_col:03d}.tif'
                        idx = build_index(proc_axes, {'T': time_col, 'C': channel, 'Z':depth, 'X':slice(0,x),'Y':slice(0,y)})
                        img = data[idx].astype(np.float)
                        img_max = img.max() * max_scale
                        if img_max==0: continue #do not save images with no contents.
                        img /= img_max
                        down_img, down_up_img = new_crappify(img)
                    except:
                        continue

                    img = img_as_ubyte(img)
                    pimg = PIL.Image.fromarray(img, mode='L')
                    cur_size = pimg.size
                    pimg.save(hr_dir/save_fn)

                    small_img = PIL.Image.fromarray(img_as_ubyte(down_img))
                    big_img = PIL.Image.fromarray(img_as_ubyte(down_up_img))
                    small_img.save(lr_dir/save_fn)
                    big_img.save(lr_up_dir/save_fn)

for lst in [one_channel, two_channel, hr_airyscan]:
    lst.sort()
    random.shuffle(lst)
    split_idx = int(valid_pct * len(lst))
    valid_files += lst[-split_idx:]
    train_files += lst[:-split_idx]

    for subdir, file_list in [('train', train_files),('valid', valid_files)]:
        print(f'\n\ncopy, crappify and upsample {subdir} files\n\n')
        for fn in progress_bar(file_list):
            base_name = fn.stem
            new_crappify_movie_to_tifs(
                fn, hr_path/subdir, lr_path/subdir, lr_up_path/subdir,
                base_name, max_scale=1.05)

print('done')
