from pathlib import Path
import shutil
import random
import os
import superres.helpers as helpers
from fastprogress import progress_bar, master_bar
import PIL
import czifile
from superres.helpers import *

from pathlib import Path
import shutil
import random
import os
from fastprogress import progress_bar

data_path = Path('/scratch/bpho')
sources = data_path/'datasources'
raw = sources/'Processed_and_Raw_Movie'
processed_2d = raw/'2D SR Airyscan Process' # we won't use this bc is cropped
processed = raw/'Airyscan process'


datasets = data_path/'datasets'
raw2hr = datasets/'raw2hr_001'
raw2proc_map = { fn: next(iter(processed.glob(f'{fn.stem}*'))) for fn in raw.glob('*920*.czi')}

raw_tiffs = raw2hr/'raw'
proc_tiffs = raw2hr/'proc'

if raw2hr.exists(): shutil.rmtree(raw2hr)
raw_tiffs.mkdir(parents=True)
proc_tiffs.mkdir(parents=True)

validation_files = [
    'MitoTracker Red FM 920x920 10min 03',
    'MitoTracker Red FM 920x920 2min 03'
]

def print_czi_info(fn):
    with czifile.CziFile(fn) as czi_f:
        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        print(fn.stem, proc_shape)


def pull_raw_tiffs(fn, raw_dir, max_ratio=1.05, base_name=None, mbar=None):
    if base_name is None: base_name = fn.stem
    with czifile.CziFile(fn) as czi_f:
        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        times = proc_shape['T']
        x,y = proc_shape['X'], proc_shape['Y']
        data = czi_f.asarray()
        for tidx in progress_bar(range(times), parent=mbar):
            idx = build_index(proc_axes, {'T': tidx, 'H':0, 'X':slice(0,x),'Y':slice(0,y)})
            img = data[idx].astype(np.float32)
            img /= (img.max() * max_ratio)
            pimg = PIL.Image.fromarray(np.uint8(img*255))
            save_name = f'{base_name}_{tidx:04d}.tif'
            pimg.save(raw_dir/save_name)
            
def pull_proc_tiffs(fn, proc_dir, max_ratio=1.05, base_name=None, mbar=None):
    if base_name is None: base_name = fn.stem
    with czifile.CziFile(fn) as czi_f:
        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        times = proc_shape['T']
        x,y = proc_shape['X'], proc_shape['Y']
        data = czi_f.asarray()
        for tidx in progress_bar(range(times), parent=mbar):
            idx = build_index(proc_axes, {'T': tidx, 'X':slice(0,x),'Y':slice(0,y)})
            img = data[idx].astype(np.float32)
            img /= (img.max() * max_ratio)
            pimg = PIL.Image.fromarray(np.uint8(img*255))
            save_name = f'{base_name}_{tidx:04d}.tif'
            pimg.save(raw_dir/save_name)

mbar = master_bar(raw2proc_map.items())
for raw_fn, proc_fn in mbar:
    pull_raw_tiffs(raw_fn, raw_tiffs, mbar=mbar)
    pull_raw_tiffs(proc_fn, proc_tiffs, base_name=raw_fn.stem, mbar=mbar)

def random_crop_tile_no_upsample(base_name, hr_img, lr_img, tile_size, threshold):
    try:
        hr_crop = None
        tries = 0
        while hr_crop is None:
            w, h = hr_img.size
            th, tw = tile_size, tile_size
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            crop_rect = (j, i, j + tw, i + th) 
            hr_crop = hr_img.crop(crop_rect)
            if (np.asarray(hr_crop) > threshold).mean() < 0.01: hr_crop = None
            tries += 1
            if tries > 10:
                threshold /= 2
                tries = 0
        lr_crop = lr_img.crop(crop_rect)
        return hr_crop, lr_crop
    except Exception as e:
        print(base_name, e)
        import pdb; pdb.set_trace()


def tif_to_tiles_no_upsample(
        lr_tif_fn, hr_tif_fn, base_name, hr_ROI_dir, lr_ROI_dir,
        size=256, num_tiles=5, threshold=0.25):
    hr_ROI_dir, lr_ROI_dir = Path(hr_ROI_dir), Path(lr_ROI_dir)
    hr_ROI_dir.mkdir(parents=True, exist_ok=True)
    lr_ROI_dir.mkdir(parents=True, exist_ok=True)
    

    hr_img = PIL.Image.open(hr_tif_fn)
    lr_img = PIL.Image.open(lr_tif_fn)
    count = 0
    while count < num_tiles:
        save_name = f'{base_name}_{count:02d}.tif'
        HR_ROI, LR_ROI = random_crop_tile_no_upsample(base_name, hr_img, lr_img, size, threshold=threshold)
        count = count+1
        HR_ROI.save(hr_ROI_dir/save_name)
        LR_ROI.save(lr_ROI_dir/save_name)
        

def get_sub_dir(fn): 
    key = fn.stem.split('_')[0]
    if key in validation_files: return 'valid'
    else: return 'train'
    
def save_tiles(tile_size, num_tiles=5, threshold=25, mbar=None):
    print(f'\n\nsave {tile_size} tiles')
    hr_ROI = raw2hr/f'roi_hr_{tile_size}'
    lr_ROI = raw2hr/f'roi_lr_{tile_size}'

    for raw_fn in progress_bar(list(raw_tiffs.iterdir()), parent=mbar):
        proc_fn = proc_tiffs/raw_fn.name
        sub_dir = get_sub_dir(raw_fn)
        base_name = f'{raw_fn.stem}_{tile_size}.tif'
        tif_to_tiles_no_upsample(
            proc_fn, raw_fn, base_name, hr_ROI/sub_dir, lr_ROI/sub_dir,
            size=tile_size, num_tiles=num_tiles, threshold=threshold)

mbar = master_bar([128,256,512,768])
for tile_size in mbar:
    save_tiles(tile_size, mbar=mbar)