import os
import czifile
import PIL
import numpy as np
import torchvision as vision
from pathlib import Path

def chmod_all_readwrite(dirname):
    for root, dirs, files in os.walk(dirname):
        os.chmod(root, 0o775)
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o775)
        for f in files:
            os.chmod(os.path.join(root, f), 0o664)


def get_czi_shape_info(czi):
    shape = czi.shape
    axes = czi.axes
    axes_dict = {axis:idx for idx,axis in enumerate(czi.axes)}
    shape_dict = {axis:shape[axes_dict[axis]] for axis in czi.axes}
    return axes_dict, shape_dict

def build_index(axes, ix_select):
    idx = [ix_select.get(ax, 0) for ax in axes]
    return tuple(idx)

def czi_to_tiles(czi_fn, hr_ROI_dir, lr_ROI_dir, lr_ROI_small_dir, 
                 size=256, channels=None, depths=None, num_tiles=5, scale=4, max_scale=1.5, max_per_movie=True):
    #import pdb; pdb.set_trace()
    hr_ROI_dir, lr_ROI_dir, lr_ROI_small_dir= Path(hr_ROI_dir), Path(lr_ROI_dir), Path(lr_ROI_small_dir)
    hr_ROI_dir.mkdir(parents=True, exist_ok=True)
    lr_ROI_dir.mkdir(parents=True, exist_ok=True)
    lr_ROI_small_dir.mkdir(parents=True, exist_ok=True)
    with czifile.CziFile(czi_fn) as czi_f:
        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        if channels is None: channels = proc_shape['C']
        if depths is None: depths = proc_shape['Z']
        times = proc_shape['T']
        x,y = proc_shape['X'], proc_shape['Y']
        data = czi_f.asarray()
        for channel in range(channels):
            for depth in range(depths):
                img_max = None
                for time_col in range(times):
                    idx = build_index(proc_axes, {'T': time_col, 'C': channel, 'Z':depth, 'X':slice(0,x),'Y':slice(0,y)})
                    img = data[idx].astype(np.float)
                    if img_max is None: img_max = img.max() * max_scale
                    img /= img_max
                    pimg = PIL.Image.fromarray((img*255).astype(np.uint8), mode='L')
                    rc = vision.transforms.RandomCrop([size, size])
                    count = 0
                    #pass in num_tile sub ROIS drawn randomly from img and save them 
                    #(and its downsampled counterpart) to disk (similar to below)
                    #another for loop for samples (keep drawing ROIs and check until saving 5 samples to disk)
                    # - generate tiles (get it work first)
                    # - check tiles
                    while count < num_tiles:
                        save_fn = hr_ROI_dir/f'{czi_fn.stem}_{channel:02d}_{depth:03d}_{time_col:03d}_{count:02d}.tif' #add sample number.
                        ROI = rc(pimg)
                        ROI_stats = PIL.ImageStat.Stat(ROI)
                        if ROI_stats.stddev[0]>2:
                            count = count+1
                            ROI.save(save_fn)
                            cur_size = ROI.size
                            new_size = (cur_size[0]//scale, cur_size[1]//scale)
                            small_img = ROI.resize(new_size, resample=PIL.Image.BICUBIC)
                            big_img = small_img.resize(cur_size, resample=PIL.Image.BICUBIC)
                            small_img.save(lr_ROI_small_dir/save_fn.name)
                            big_img.save(lr_ROI_dir/save_fn.name)

                    if not max_per_movie: img_max = None

def czi_to_tiffs(czi_fn, hr_dir, lr_dir, channels=None, depths=None, max_scale=1.5, max_per_movie=True):
    hr_dir, lr_dir = Path(hr_dir), Path(lr_dir)
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    with czifile.CziFile(czi_fn) as czi_f:
        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        if channels is None: channels = proc_shape['C']
        if depths is None: depths = proc_shape['Z']
        times = proc_shape['T']
        x,y = proc_shape['X'], proc_shape['Y']
        data = czi_f.asarray()
        for channel in range(channels):
            for depth in range(depths):
                img_max = None
                for time_col in range(times):
                    idx = build_index(proc_axes, {'T': time_col, 'C': channel, 'Z':depth, 'X':slice(0,x),'Y':slice(0,y)})
                    img = data[idx].astype(np.float)
                    save_fn = f'{czi_fn.stem}_{channel:02d}_{depth:03d}_{time_col:03d}.tif'
                    if img_max is None: img_max = img.max() * max_scale
                    img /= img_max
                    if not max_per_movie: img_max = None
                    pimg = PIL.Image.fromarray((img*255).astype(np.uint8), mode='L')
                    pimg.save(hr_dir/save_fn)
                    cur_size = pimg.size
                    new_size = (cur_size[0]//4, cur_size[1]//4)
                    small_img = pimg.resize(new_size, resample=PIL.Image.BICUBIC)
                    small_img.save(lr_dir/save_fn)