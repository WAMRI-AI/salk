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

dname = 'synth_002'
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

print('done')
