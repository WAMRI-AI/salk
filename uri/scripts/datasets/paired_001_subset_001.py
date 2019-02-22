#for extracting paired images
from pathlib import Path
import shutil
import random
import os
import superres.helpers as helpers
from fastprogress import progress_bar
import PIL

from pathlib import Path
import shutil
import random
import os
from fastprogress import progress_bar

data_path = Path('/scratch/bpho')
sources = data_path/'datasources'
src = sources/'confocal_airyscan_pairs_mito'
src_lr = src/'lr_registered'
src_hr = src/'hr_registered'

datasets = data_path/'datasets'
subset = datasets/'paired_001'/'confocal_airyscan_pairs_mito'
subset_lr = subset/'lr'
subset_lr_up = subset/'lr_up'
subset_hr = subset/'hr'
valid_pct = 0.2

if subset.exists(): shutil.rmtree(subset)

def pull_id(x):
    id = int(x.name.split('#')[1].split('_')[0])
    return id

def pull_depth(x):
    id = int(x.stem.split('_')[-1])
    return id


file_ids = list(set([pull_id(x) for x in src_lr.iterdir()]))
split_idx = int(valid_pct * len(file_ids))
valid_ids = file_ids[0:split_idx]
train_ids = file_ids[split_idx:]

def build_file_map(src): return  { (pull_id(x),pull_depth(x)):x for x in src.iterdir() }

hr_file_map = build_file_map(src_hr) #map the lr and hr names
lr_file_map = build_file_map(src_lr)

file_map = hr_file_map
dest_dir = subset_hr

def copy_tif_files(file_map, dest_dir):
    for (id, depth), fn in file_map.items():
        if id in train_ids: dest = dest_dir/'train'
        else: dest = dest_dir/'valid'
        new_fn = f'{id}_{depth}.tif'
        dest.mkdir(parents=True, mode=0o777, exist_ok=True)
        shutil.copy(fn, dest/new_fn)
        
def scale_tif_files(file_map, dest_dir, scale=4):
    for (id, depth), fn in file_map.items():
        if id in train_ids: dest = dest_dir/'train'
        else: dest = dest_dir/'valid'
        new_fn = f'{id}_{depth}.tif'
        dest.mkdir(parents=True, mode=0o777, exist_ok=True)
        img = PIL.Image.open(fn)
        cur_size = img.size
        new_size = (cur_size[0]*scale, cur_size[1]*scale)
        big_img = img.resize(new_size, resample=PIL.Image.BICUBIC) #can try zoom too.
        big_img.save(dest/new_fn)

def save_tiles(tile_size, num_tiles=5, scale=4, threshold=180):
    print(f'\n\nsave {tile_size} tiles')
    hr_ROI = subset/f'roi_hr_{tile_size}'
    lr_ROI = subset/f'roi_lr_{tile_size}'
    lr_ROI_small = subset/f'roi_lr_up_{tile_size}'

    for (id, depth), hr_fn in progress_bar(list(hr_file_map.items())):
        lr_fn = lr_file_map[(id, depth)]
        if id in train_ids: sub_dir = 'train'
        else: sub_dir = 'valid'
        base_name = f'{tile_size}_{id}_{depth}'
        helpers.tif_to_tiles(lr_fn, hr_fn, base_name, hr_ROI/sub_dir, lr_ROI/sub_dir, lr_ROI_small/sub_dir, 
                             size=tile_size, num_tiles=num_tiles, scale=scale, threshold=threshold, untiled_ls=untiled_files)

untiled_files = []
for tile_size in [128,256,512,1024]:
    save_tiles(tile_size)

with open(subset/'untiled_filelist.txt', 'w') as f:
    for item in untiled_files:
        f.write("%s\n" % item)
        
print('\n\nsave scaled up tifs')
scale_tif_files(lr_file_map, subset_lr_up)
print('\ncopy original tifs')
copy_tif_files(hr_file_map, subset_hr)
copy_tif_files(lr_file_map, subset_lr)

helpers.chmod_all_readwrite(subset)

print('\n\ndone')
