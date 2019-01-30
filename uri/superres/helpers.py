import os
import czifile
import PIL
import numpy as np
import torchvision as vision
from pathlib import Path
import random
from PIL import ImageFile

from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from torchvision.models import vgg16_bn
import sys
from superres import FeatureLoss


ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def czi_to_tiffs(czi_fn, hr_dir, lr_dir, lr_small_dir, channels=None, depths=None, max_scale=1.5, max_per_movie=True):
    hr_dir, lr_dir, lr_small_dir = Path(hr_dir), Path(lr_dir), Path(lr_small_dir)
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    lr_small_dir.mkdir(parents=True, exist_ok=True)
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
                    big_img = small_img.resize(cur_size, resample=PIL.Image.BICUBIC)
                    small_img.save(lr_small_dir/save_fn)
                    big_img.save(lr_dir/save_fn)

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
        big_img = img.resize(new_size, resample=PIL.Image.BICUBIC)
        big_img.save(dest/new_fn)

def random_crop_tile(base_name, hr_img, lr_img, tile_size, scale, threshold):
    try:
        hr_crop = None
        tries = 0
        while hr_crop is None:
            w, h = hr_img.size
            th, tw = tile_size, tile_size
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            crop_rect = (j, i, j + tw, i + th) 
            small_crop_rect = [i//scale for i in crop_rect]
            hr_crop = hr_img.crop(crop_rect)
            if (np.asarray(hr_crop) > threshold).mean() < 0.05: hr_crop = None
            tries += 1
            if tries > 10:
                threshold /= 2
                tries = 0
        lr_crop = lr_img.crop(small_crop_rect)
        lr_crop_upsampled = lr_img.resize(hr_img.size, resample=PIL.Image.BICUBIC).crop(crop_rect)
        return hr_crop, lr_crop, lr_crop_upsampled
    except Exception as e:
        print(base_name, e)
        import pdb; pdb.set_trace()

def tif_to_tiles(lr_tif_fn, hr_tif_fn, base_name, hr_ROI_dir, lr_ROI_dir, lr_ROI_up_dir,
                 size=256, num_tiles=5, scale=4, threshold=0.85):
    hr_ROI_dir, lr_ROI_dir, lr_ROI_up_dir= Path(hr_ROI_dir), Path(lr_ROI_dir), Path(lr_ROI_up_dir)
    hr_ROI_dir.mkdir(parents=True, exist_ok=True)
    lr_ROI_dir.mkdir(parents=True, exist_ok=True)
    lr_ROI_up_dir.mkdir(parents=True, exist_ok=True)

    hr_img = PIL.Image.open(hr_tif_fn)
    lr_img = PIL.Image.open(lr_tif_fn)
    count = 0
    while count < num_tiles:
        save_name = f'{base_name}_{count:02d}.tif'
        HR_ROI, LR_ROI, LR_ROI_Upsample = random_crop_tile(base_name, hr_img, lr_img, size, scale, threshold=threshold)
        count = count+1
        HR_ROI.save(hr_ROI_dir/save_name)
        LR_ROI.save(lr_ROI_dir/save_name)
        LR_ROI_Upsample.save(lr_ROI_up_dir/save_name)

def build_crappify_model(model_path, model, bs, img_size):
    img_data = Path('/scratch/bpho/datasets/paired_001/')
    model_path = Path('/scratch/bpho/models')

    def get_src(size=128):
        hr_tifs = img_data/f'roi_hr_{size}'
        lr_tifs = img_data/f'roi_lr_up_{size}'

        def map_to_lr(x):
            lr_name = x.relative_to(hr_tifs)
            return lr_tifs/lr_name
        src = (ImageImageList
                .from_folder(hr_tifs)
                .split_by_folder()
                .label_from_func(map_to_lr))
        return src

    def get_data(bs, size, tile_size=None):
        if tile_size is None: tile_size = size
        src = get_src(tile_size)
        tfms = [[rand_crop(size=size)],[]]
        tfms = get_transforms(flip_vert=True, max_zoom=2)
        y_tfms = [[t for t in tfms[0]], [t for t in tfms[1]]]

        data = (src
                .transform(tfms, size=size)
                .transform_y(y_tfms, size=size)
                .databunch(bs=bs).normalize(imagenet_stats, do_y=True))
        data.c = 3
        return data

    data = get_data(bs, img_size, tile_size=img_size)
    wd = 1e-3
    arch = models.resnet34 
    learn = unet_learner(data, arch, wd=wd, blur=True, 
                         norm_type=NormType.Weight, model_dir=model_path)
    gc.collect()
    learn = learn.load(model_name)
    return learn


def crappify_movie_to_tifs(czi_fn, hr_dir, lr_dir, lr_up_dir, base_name, model_path, model_name, max_scale=1.1, max_per_movie=True):
    learn = load_learner(model_path, model_name)
    hr_dir, lr_dir, lr_up_dir = Path(hr_dir), Path(lr_dir), Path(lr_up_dir)
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    lr_up_dir.mkdir(parents=True, exist_ok=True)
    with czifile.CziFile(czi_fn) as czi_f:
        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        channels = proc_shape['C']
        depths = proc_shape['Z']
        times = proc_shape['T']
        x,y = proc_shape['X'], proc_shape['Y']
        data = czi_f.asarray()
        for channel in range(channels):
            for depth in range(depths):
                img_max = None
                for time_col in range(times):
                    idx = build_index(proc_axes, {'T': time_col, 'C': channel, 'Z':depth, 'X':slice(0,x),'Y':slice(0,y)})
                    img = data[idx].astype(np.float)
                    save_fn = f'{base_name}_{channel:02d}_{depth:03d}_{time_col:03d}.tif'
                    if img_max is None: img_max = img.max() * max_scale
                    img /= img_max
                    if not max_per_movie: img_max = None
                    pimg = PIL.Image.fromarray((img*255).astype(np.uint8), mode='L')
                    cur_size = pimg.size
                    pimg.save(hr_dir/save_fn)
                    timg = Image(pil2tensor(pimg,np.float32).div_(255))
                    pred, _, _ = learn.predict(timg)
                    pred = image2np(pred.data)
                    crapimg = PIL.Image.fromarray((pred*255).astype(np.uint8))
                    new_size = (cur_size[0]//4, cur_size[1]//4)
                    small_img = pimg.resize(new_size, resample=PIL.Image.BICUBIC)
                    big_img = small_img.resize(cur_size, resample=PIL.Image.BICUBIC)
                    small_img.save(lr_dir/save_fn)
                    big_img.save(lr_up_dir/save_fn)


def image_from_tiles(learn, img, tile_sz=128, scale=4):
    pimg = PIL.Image.fromarray((img*255).astype(np.uint8), mode='L').convert('RGB')
    cur_size = pimg.size
    new_size = (cur_size[0]*scale, cur_size[1]*scale)
    in_img = Image(pil2tensor(pimg.resize(new_size, resample=PIL.Image.BICUBIC),np.float32).div_(255))
    c, w, h = in_img.shape

    in_tile = torch.zeros((c,tile_sz,tile_sz))
    out_img = torch.zeros((c,w,h))

    for x_tile in range(math.ceil(w/tile_sz)):
        for y_tile in range(math.ceil(h/tile_sz)):
            x_start = x_tile

            x_start = x_tile*tile_sz
            x_end = min(x_start+tile_sz, w)
            y_start = y_tile*tile_sz
            y_end = min(y_start+tile_sz, h)


            in_tile[:,0:(x_end-x_start), 0:(y_end-y_start)] = in_img.data[:,x_start:x_end, y_start:y_end]

            out_tile,_,_ = learn.predict(Image(in_tile))

            out_x_start = x_start
            out_x_end = x_end
            out_y_start = y_start
            out_y_end = y_end

            #print("out: ", out_x_start, out_y_start, ",", out_x_end, out_y_end)
            in_x_start = 0
            in_y_start = 0
            in_x_end = x_end-x_start
            in_y_end = y_end-y_start
            #print("tile: ",in_x_start, in_y_start, ",", in_x_end, in_y_end)

            out_img[:,out_x_start:out_x_end, out_y_start:out_y_end] = out_tile.data[:,
                                                                                              in_x_start:in_x_end,
                                                                                              in_y_start:in_y_end]
    return out_img


def czi_predict_movie(learn, czi_in, orig_out='orig.tif', pred_out='pred.tif', size=128):
    with czifile.CziFile(czi_in) as czi_f:
        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        channels = proc_shape['C']
        depths = proc_shape['Z']
        times = proc_shape['T']
        x,y = proc_shape['X'], proc_shape['Y']
        data = czi_f.asarray()
        preds = []
        origs = []
        img_max = None
        for t in progress_bar(list(range(times))):
            idx = build_index(proc_axes, {'T': t, 'C': 0, 'Z':0, 'X':slice(0,x),'Y':slice(0,y)})
            img = data[idx].astype(np.float32)
            if img_max is None: img_max = img.max() * 1.0
            img /= img_max
            out_img = image_from_tiles(learn, img, tile_sz=size).permute([1,2,0])
            pred = (out_img[None]*255).cpu().numpy().astype(np.uint8)
            preds.append(pred)
            orig = (img[None]*255).astype(np.uint8)
            origs.append(orig)

        all_y = np.concatenate(preds)
        #print(all_y.shape)
        imageio.mimwrite(pred_out, all_y) #, fps=30, macro_block_size=None) # for mp4
        all_y = np.concatenate(origs)
        #print(all_y.shape)
        imageio.mimwrite(orig_out, all_y) #, fps=30, macro_block_size=None)
