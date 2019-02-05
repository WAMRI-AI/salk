from pathlib import Path
import shutil
import random
import os
import superres.helpers as helpers
from fastprogress import progress_bar

data_path = Path('/scratch/bpho')
sources = data_path/'datasources'
src_1 = sources/'MitoTracker_Red_FM_movie_data'

datasets = data_path/'datasets'
movies_002 = datasets/'movies_002'
train = movies_002/'train'
valid = movies_002/'valid'
test = movies_002/'test'

if movies_002.exists(): shutil.rmtree(movies_002)
for fld in [train, valid, test]: fld.mkdir(parents=True, mode=0o777)

test_mito = list(src_1.glob('*230*.czi'))
train_mito = list(src_1.glob('*920*.czi'))

valid_pct = 0.20
train_files = []
valid_files = []
test_files = []

random.seed(42)
for lst in [train_mito]:
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

def save_tiles(tile_size):
    print(f'\n\nsave {tile_size} tiles')
    hr_ROI = movies_002/f'roi_hr_{tile_size}'
    lr_ROI = movies_002/f'roi_lr_{tile_size}'
    lr_ROI_small = movies_002/f'roi_lr_small_{tile_size}'

    for fn in train_files:
        helpers.czi_to_tiles(fn, hr_ROI/'train', lr_ROI/'train', lr_ROI_small/'train', 
        size=tile_size, channels=1, max_scale=1.05, max_per_movie=False)
    for fn in valid_files:
        helpers.czi_to_tiles(fn, hr_ROI/'valid', lr_ROI/'valid', lr_ROI_small/'valid', 
        size=tile_size, channels=1, max_scale=1.05, max_per_movie=False)

img_hr = movies_002/'img_hr'
img_lr = movies_002/'img_lr'
img_lr_small = movies_002/'img_lr_small'
print('save tiffs')
for fn in train_files: helpers.czi_to_tiffs(fn,img_hr/'train',img_lr/'train', img_lr_small/'train', max_scale=1.05, max_per_movie=False)
for fn in valid_files: helpers.czi_to_tiffs(fn,img_hr/'valid',img_lr/'valid', img_lr_small/'valid', max_scale=1.05, max_per_movie=False)

for sz in [64,128,256,512]: save_tiles(sz)


helpers.chmod_all_readwrite(movies_002)
print('\n\ndone')
