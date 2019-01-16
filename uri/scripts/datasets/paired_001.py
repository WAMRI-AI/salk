from pathlib import Path
import shutil
import random
import os
import helpers
from fastprogress import progress_bar


from pathlib import Path
import shutil
import random
import os
from fastprogress import progress_bar

data_path = Path('/scratch/bpho')
sources = data_path/'datasources'
src_1 = sources/'confocal_airyscan_pairs_mito'
src_1_lr = src_1/'lr_registered'
src_1_hr = src_1/'hr_registered'

datasets = data_path/'datasets'
paired_001 = datasets/'paired_001'
paired_001_lr = paired_001/'lr'
paired_001_hr = paired_001/'hr'
valid_pct = 0.2

if paired_001.exists(): shutil.rmtree(paired_001)

def pull_id(x):
    id = int(x.name.split('#')[1].split('_')[0])
    return id

def pull_depth(x):
    id = int(x.stem.split('_')[-1])
    return id


file_ids = list(set([pull_id(x) for x in src_1_lr.iterdir()]))
split_idx = int(valid_pct * len(file_ids))
valid_ids = file_ids[0:split_idx]
train_ids = file_ids[split_idx:]

def build_file_map(src): return  { (pull_id(x),pull_depth(x)):x for x in src.iterdir() }

hr_file_map = build_file_map(src_1_hr)
lr_file_map = build_file_map(src_1_lr)

file_map = hr_file_map
dest_dir = paired_001_hr

def copy_tif_files(file_map, dest_dir):
    for (id, depth), fn in file_map.items():
        if id in train_ids: dest = dest_dir/'train'
        else: dest = dest_dir/'valid'
        new_fn = f'{id}_{depth}.tif'
        dest.mkdir(parents=True, mode=0o777, exist_ok=True)
        shutil.copy(fn, dest/new_fn)
        
copy_tif_files(hr_file_map, paired_001_hr)
copy_tif_files(lr_file_map, paired_001_lr)

helpers.chmod_all_readwrite(paired_001)

print('\n\ndone')