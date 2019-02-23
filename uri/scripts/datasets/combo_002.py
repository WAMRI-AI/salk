from pathlib import Path
import shutil
import random
import os
import superres.helpers as helpers
from fastprogress import progress_bar
from scipy.ndimage.interpolation import zoom
import PIL.Image
import numpy as np
import czifile
from superres.helpers import algo_crappify_movie_to_tifs
from pdb import set_trace

data_path = Path('/scratch/bpho')
sources = data_path/'datasources'

dname = 'combo_002'
dpath = data_path/'datasets'/dname
if dpath.exists(): shutil.rmtree(dpath)

hr_path = dpath/'hr'
lr_path = dpath/'lr'
lr_up_path = dpath/'lr_up'

valid_pct = 0.2

# paired images
paired = sources/'confocal_airyscan_pairs_mito'
paired_lr = paired/'lr_registered'
paired_hr = paired/'hr_registered'


def pull_id(x):
    id = int(x.name.split('#')[1].split('_')[0])
    return id

def pull_depth(x):
    id = int(x.stem.split('_')[-1])
    return id

file_ids = list(set([pull_id(x) for x in paired_lr.iterdir()]))
split_idx = int(valid_pct * len(file_ids))
valid_ids = file_ids[0:split_idx]
train_ids = file_ids[split_idx:]

def build_file_map(src): return  { (pull_id(x),pull_depth(x)):x for x in src.iterdir() }

def pair_copy_tif_files(file_map, dest_dir, scale_list=None):
    for (id, depth), fn in progress_bar(list(file_map.items())):
        if id in train_ids: dest = dest_dir/'train'
        else: dest = dest_dir/'valid'
        new_fn = f'pair_{id}_{depth}.tif'
        dest.mkdir(parents=True, mode=0o775, exist_ok=True)
        shutil.copy(fn, dest/new_fn)

        if scale_list:
            for scale, scale_dir in scale_list:
                scale_dir.mkdir(parents=True, mode=0o775, exist_ok=True)
                lr_data = np.array(PIL.Image.open(dest/fn))
                lr_up_data = zoom(lr_data, (scale, scale, 1), order=1)
                lr_img_up = PIL.Image.fromarray(lr_up_data)
                lr_img_up.save(scale_dir/new_fn)

hr_pair_map = build_file_map(paired_hr) #map the lr and hr names
lr_pair_map = build_file_map(paired_lr)

print('\n\ncopy HR paired\n\n')
pair_copy_tif_files(hr_pair_map, hr_path)
print('\n\ncopy lr paired and upsample\n\n')
pair_copy_tif_files(lr_pair_map, lr_path, [(4, lr_up_path)])

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
            algo_crappify_movie_to_tifs(
                fn, hr_path/subdir, lr_path/subdir, lr_up_path/subdir,
                base_name, max_scale=1.05, max_per_movie=True)

print('\n\nbuild tiles.\n\n')


def save_tiles(tile_size, untiled_files, num_tiles=5, scale=4, threshold=100):
    for sub_dir in ['train','valid']:
        print(f'\n\nbuild tiles {sub_dir}/{tile_size}\n\n')
        hr_ROI = dpath/f'roi_hr_{tile_size}'/subdir
        lr_ROI = dpath/f'roi_lr_{tile_size}'/subdir
        lr_up_ROI = dpath/f'roi_lr_up_{tile_size}'/subdir
        #print('\n', hr_ROI, '\n', lr_ROI, '\n', lr_up_ROI)
        print('Creating ROIs with tile size ' + str(tile_size))
        hrdir = hr_path/sub_dir
        lrdir = lr_path/sub_dir
        for hr_fn in progress_bar(list(hrdir.iterdir())):
            #print('Processing ' + hr_fn.name + ', tile_size is ' + str(tile_size) + '.')
            lr_fn = lrdir/hr_fn.name
            helpers.tif_to_tiles(lr_fn, hr_fn, hr_fn.stem, hr_ROI, lr_up_ROI, lr_ROI, size=tile_size,
                                 num_tiles=num_tiles, scale=scale, threshold=threshold, untiled_ls=untiled_files)

untiled_files = []
for tile_size in [128,256,512]:
    save_tiles(tile_size, untiled_files)

print('\n{len(untiled_files)} untiled files')
print('\n\nall done.\n\n')
