import os
import czifile
import PIL
import numpy as np
import torchvision as vision
from pathlib import Path

def chmod_all_readwrite(dirname):
    for root, dirs, files in os.walk(dirname):
        os.chmod(root, 0o777)
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            os.chmod(os.path.join(root, f), 0o666)


def get_czi_shape_info(czi):
    shape = czi.shape
    axes = czi.axes
    axes_dict = {axis:idx for idx,axis in enumerate(czi.axes)}
    shape_dict = {axis:shape[axes_dict[axis]] for axis in czi.axes}
    return axes_dict, shape_dict

def build_index(axes, ix_select):
    idx = [ix_select.get(ax, 0) for ax in axes]
    return tuple(idx)

def czi_to_tiles(czi_fn, hr_ROI_dir, lr_ROI_dir, size=256, channels=None, depths=None, num_tiles=5, scale=4):
    #import pdb; pdb.set_trace()
    hr_ROI_dir, lr_ROI_dir = Path(hr_ROI_dir), Path(lr_ROI_dir)
    hr_ROI_dir.mkdir(parents=True, exist_ok=True)
    lr_ROI_dir.mkdir(parents=True, exist_ok=True)
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
                    if img_max is None: img_max = img.max() * 2.
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
                        if ROI_stats.stddev[0]>10:
                            count = count+1
                            ROI.save(save_fn)
                            cur_size = ROI.size
                            new_size = (cur_size[0]//4, cur_size[1]//4)
                            (ROI
                             .resize(new_size, resample=PIL.Image.BICUBIC)
                             .resize(cur_size, resample=PIL.Image.BICUBIC)
                             .save(lr_ROI_dir/save_fn.name))
