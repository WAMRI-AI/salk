from pathlib import Path
import shutil
import random
import os
import helpers
from fastprogress import progress_bar

data_path = Path('/DATA/WAMRI/salk/uri/bpho')
sources = data_path/'datasources'
src_1 = sources/'live_neuron_mito_timelapse_for_deep_learning'
src_2 = sources/'MitoTracker_Red_FM_movie_data'

datasets = data_path/'datasets'
movies_001 = datasets/'movies_001'
train = movies_001/'train'
valid = movies_001/'valid'
test = movies_001/'test'

if movies_001.exists(): shutil.rmtree(movies_001)
for fld in [train, valid, test]: fld.mkdir(parents=True, mode=0o777)

two_channel = list(src_1.glob('*MTGreen*.czi'))
one_channel = [x for x in src_1.glob('*.czi') if x not in two_channel]

test_mito = list(src_2.glob('*230*.czi'))
train_mito = list(src_2.glob('*920*.czi'))

valid_pct = 0.20
train_files = []
valid_files = []
test_files = []

random.seed(42)
for lst in [two_channel, one_channel, train_mito]:
    lst.sort()
    random.shuffle(lst)
    split_idx = int(valid_pct * len(lst))
    valid_files += lst[-split_idx:]
    train_files += lst[:-split_idx]

test_files = test_mito

for x in train_files: shutil.copy(x, train/x.name)
for x in valid_files: shutil.copy(x, valid/x.name)
for x in test_files: shutil.copy(x, test/x.name)

train_files = progress_bar(list(train.iterdir()))
valid_files = progress_bar(list(valid.iterdir()))

size=64
hr_ROI = movies_001/'roi_hr_64'
lr_ROI = movies_001/'roi_lr_64'

for fn in train_files:
 helpers.czi_to_tiles(fn, hr_ROI/'train', lr_ROI/'train', size=size, channels=1)
for fn in valid_files:
 helpers.czi_to_tiles(fn, hr_ROI/'valid', lr_ROI/'valid', size=size, channels=1)

size=128
hr_ROI = movies_001/'roi_hr_128'
lr_ROI = movies_001/'roi_lr_128'

for fn in train_files:
 helpers.czi_to_tiles(fn, hr_ROI/'train', lr_ROI/'train', size=size, channels=1)
for fn in valid_files:
 helpers.czi_to_tiles(fn, hr_ROI/'valid', lr_ROI/'valid', size=size, channels=1)

helpers.chmod_all_readwrite(movies_001)
