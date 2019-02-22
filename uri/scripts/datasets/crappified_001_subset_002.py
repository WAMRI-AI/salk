from pathlib import Path
import shutil
import random
import os
import csv
import superres.helpers as helpers
from fastprogress import progress_bar

data_path = Path('/scratch/bpho')
sources = data_path/'datasources'
src = sources/'Airyscan_processed_data_from_the_server'
src_hr = list(src.glob('*.czi'))

datasets = data_path/'datasets'
subset = datasets/'crappified_001'/'Airyscan_processed_data_from_the_server'
subset_lr = subset/'lr'
subset_lr_up = subset/'lr_up'
subset_hr = subset/'hr'
valid_pct = 0.2
test_pct = 0.1

num_tiles = 5
scale = 4
threshold = 100

if subset.exists(): shutil.rmtree(subset)

valid_split_idx = int(valid_pct * len(src_hr))
test_split_idx = int(test_pct * len(src_hr))
non_train_idx = valid_split_idx + test_split_idx

valid_ids = src_hr[0:valid_split_idx] #Is this seed fixed?
test_ids = src_hr[valid_split_idx: non_train_idx]
train_ids = src_hr[non_train_idx:]

for fn in progress_bar(src_hr):
    if fn in valid_ids: 
        subdir = 'valid'
    elif fn in train_ids:
        subdir = 'train'
    elif fn in test_ids:
        continue
    else:
        print('Stepped into no mankind island.')
        sys.exit(1)
        
    hr_dir = subset/'hr'/subdir
    lr_dir = subset/'lr'/subdir
    lr_up_dir = subset/'lr_up'/subdir        
    base_name = fn.stem
    for fld in [hr_dir, lr_dir, lr_up_dir]: fld.mkdir(parents=True, exist_ok=True, mode=0o775)
        
    helpers.algo_crappify_movie_to_tifs(
            fn, hr_dir, lr_dir, lr_up_dir, base_name, max_scale=1.05, max_per_movie=False)

for subdir in ['valid','train']:
    untiled_files = [] #save image frames which doesn't satisfy the cropping criteria
    hr_dir = subset/'hr'/subdir
    lr_dir = subset/'lr'/subdir
    lr_up_dir = subset/'lr_up'/subdir      
    print('The number of files in '+subdir+' set is '+str(len(list(hr_dir.iterdir()))))
    
    for tile_size in [128,256,512]: 
        hr_ROI = subset/f'roi_hr_{tile_size}'/subdir
        lr_ROI = subset/f'roi_lr_{tile_size}'/subdir
        lr_up_ROI = subset/f'roi_lr_up_{tile_size}'/subdir
        #print('\n', hr_ROI, '\n', lr_ROI, '\n', lr_up_ROI)
        print('Creating ROIs with tile size ' + str(tile_size))
        for hr_fn in progress_bar(list(hr_dir.iterdir())):
            #print('Processing ' + hr_fn.name + ', tile_size is ' + str(tile_size) + '.')
            lr_fn = lr_dir/hr_fn.name
            helpers.tif_to_tiles(lr_fn, hr_fn, hr_fn.stem, hr_ROI, lr_up_ROI, lr_ROI, size=tile_size,
                                 num_tiles=num_tiles, scale=scale, threshold=threshold, untiled_ls=untiled_files)  

    with open(subset/(subdir+'_untiled_filelist.txt'), 'w') as f:
        for item in untiled_files:
            f.write("%s\n" % item)

test_dir = subset/'test'
test_dir.mkdir(parents=True, exist_ok=True, mode=0o775)
for fn in progress_bar(test_ids):
    shutil.copy(fn, test_dir/fn.stem)

print('\n\nAll done!')

