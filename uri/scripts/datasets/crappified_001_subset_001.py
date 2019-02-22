#A good example for crappifying mito/neuron/Airyscan_server images
from pathlib import Path
import shutil
import random
import os
import superres.helpers as helpers
from fastprogress import progress_bar

data_path = Path('/scratch/bpho')
sources = data_path/'datasources'
mito_movies = sources/'MitoTracker_Red_FM_movie_data'
hr_mito_movies = list(mito_movies.glob('*920*.czi'))
lr_mito_movies = list(mito_movies.glob('*230*.czi'))

datasets = data_path/'datasets'
mitomovies_001 = datasets/'crappified_001'/'MitoTracker_Red_FM_movie_data'

num_tiles = 5
scale = 4
threshold = 100
untiled_files_valid = []
untiled_files_train = []

if mitomovies_001.exists(): shutil.rmtree(mitomovies_001)

for movie_fn in progress_bar(hr_mito_movies):
    if '03-Airyscan' in movie_fn.stem:
        subdir = 'valid'
        untiled_files = untiled_files_valid
    else:
        subdir = 'train'
        untiled_files = untiled_files_train

    hr_dir = mitomovies_001/'hr'/subdir
    lr_dir = mitomovies_001/'lr'/subdir
    lr_up_dir = mitomovies_001/'lr_up'/subdir

    base_name = movie_fn.stem
    for fld in [hr_dir, lr_dir, lr_up_dir]: fld.mkdir(parents=True, exist_ok=True, mode=0o775)
    helpers.algo_crappify_movie_to_tifs(
                movie_fn, hr_dir, lr_dir, lr_up_dir, base_name, max_scale=1.05, max_per_movie=False)

    for tile_size in [128,256,512]:
        hr_ROI = mitomovies_001/f'roi_hr_{tile_size}'/subdir
        lr_ROI = mitomovies_001/f'roi_lr_{tile_size}'/subdir
        lr_up_ROI = mitomovies_001/f'roi_lr_up_{tile_size}'/subdir

        print(hr_ROI, '\n', lr_ROI, '\n', lr_up_ROI)
        for hr_fn in progress_bar(list(hr_dir.iterdir())):
            lr_fn = lr_dir/hr_fn.name
            helpers.tif_to_tiles(lr_fn, hr_fn, hr_fn.stem, hr_ROI, lr_up_ROI, lr_ROI, size=tile_size,
                                 num_tiles=num_tiles, scale=scale, threshold=threshold, untiled_ls=untiled_files)

with open(mitomovies_001/'valid_untiled_filelist.txt', 'w') as f:
    for item in untiled_files_valid:
        f.write("%s\n" % item)
with open(mitomovies_001/'train_untiled_filelist.txt', 'w') as f:
    for item in untiled_files_train:
        f.write("%s\n" % item)
            
test_dir = mitomovies_001/'test'
test_dir.mkdir(parents=True, exist_ok=True, mode=0o775)
for movie_fn in progress_bar(lr_mito_movies):
    shutil.copy(movie_fn, test_dir/movie_fn.name)
