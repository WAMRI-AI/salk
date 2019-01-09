from pathlib import Path
import shutil
import random
import os

data_path = Path('/scratch/bpho')
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

for root, dirs, files in os.walk(movies_001):
    os.chmod(root, 0o777)
    for d in dirs:
        os.chmod(os.path.join(root, d), 0o777)
    for f in files:
        os.chmod(os.path.join(root, f), 0o666)
