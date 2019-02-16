from pathlib import Path
import shutil
import random
import os
import superres.helpers as helpers
from fastprogress import progress_bar
import sys

datasetname = 'combo_001'
data_path = Path('/scratch/bpho')
data_path = Path('/scratch/bpho')
datasources = data_path/'datasources'
datasets = data_path/'datasets'
dataset = datasets/datasetname

hr_tifs = dataset/'hr'
lr_tifs = dataset/'lr'
lr_up_tifs = dataset/'lr_up'

# wipe dataset so we can create fresh
if dataset.exists(): shutil.rmtree(dataset)

# copy in paired LR/HR
paired_001 = datasets/'paired_001'
paired_001_lr = paired_001/'lr'
paired_001_lr_up = paired_001/'lr_up'
paired_001_hr = paired_001/'hr'

# ensure the paired_001 dataset already created
if (paired_001_lr.exists() and
    paired_001_hr.exists() and
    paired_001_lr.exists() and
    paired_001_lr_up.exists()):
    # copy the hr and lr tiffs
    shutil.copytree(paired_001_hr, hr_tifs)
    shutil.copytree(paired_001_lr, lr_tifs)
    shutil.copytree(paired_001_lr_up, lr_up_tifs)
else:
    print('create paired_001 dataset before running combo script')
    sys.exit(1)

# crappify HR czifiles


mito_movies = sources/'MitoTracker_Red_FM_movie_data'
hr_mito_movies = list(mito_movies.glob('*920*.czi'))
lr_mito_movies = list(mito_movies.glob('*230*.czi'))

num_tiles = 5
scale = 4
threshold = 100
