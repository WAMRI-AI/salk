"functions to create synthetic low res images"
import numpy as np
import czifile
import PIL
from skimage.util import random_noise, img_as_ubyte
from skimage import filters
from scipy.ndimage.interpolation import zoom as npzoom
from .czi import get_czi_shape_info, build_index, is_movie
from fastai.vision import *

__all__ = ['simple_crap', 'czi_movie_to_synth', 'tif_to_synth']


def simple_crap(img, down_scale=4, up_scale=4, add_noise=True):
    img = img[0]
    img = npzoom(img, 1/down_scale, order=0)
    if add_noise:
        img = random_noise(img, mode='speckle', var=0.02, clip=True)
    img = npzoom(img, up_scale, order=0)
    return img[None].astype(np.float32).copy()


def new_crappify(img, add_noise=True, scale=4):
    "a crappifier for our microscope images"
    if add_noise:
        img = random_noise(img, mode='salt', amount=0.005)
        img = random_noise(img, mode='pepper', amount=0.005)
        lvar = filters.gaussian(img, sigma=5)
        img = random_noise(img, mode='localvar', local_vars=lvar * 0.5)
    img_down = npzoom(img, 1 / scale, order=1)
    img_up = npzoom(img_down, scale, order=1)
    return img_down, img_up


def czi_data_to_tifs(data, axes, shape, crappify, max_scale=1.05):
    np.warnings.filterwarnings('ignore')
    lr_imgs = {}
    lr_up_imgs = {}
    hr_imgs = {}
    channels = shape['C']
    depths = shape['Z']
    times = shape['T']
    x, y = shape['X'], shape['Y']

    for channel in range(channels):
        for depth in range(depths):
            for time_col in range(times):
                try:
                    idx = build_index(
                        axes, {
                            'T': time_col,
                            'C': channel,
                            'Z': depth,
                            'X': slice(0, x),
                            'Y': slice(0, y)
                        })
                    img = data[idx].astype(np.float).copy()
                    img_max = img.max() * max_scale
                    if img_max == 0:
                        continue  #do not save images with no contents.
                    img /= img_max
                    down_img, down_up_img = crappify(img)
                except:
                    continue

                tag = (channel, depth, time_col)
                img = img_as_ubyte(img)
                pimg = PIL.Image.fromarray(img, mode='L')
                small_img = PIL.Image.fromarray(img_as_ubyte(down_img))
                big_img = PIL.Image.fromarray(img_as_ubyte(down_up_img))
                hr_imgs[tag] = pimg
                lr_imgs[tag] = small_img
                lr_up_imgs[tag] = big_img

    np.warnings.filterwarnings('default')
    return hr_imgs, lr_imgs, lr_up_imgs

def img_data_to_tifs(data, times, crappify, max_scale=1.05):
    np.warnings.filterwarnings('ignore')
    lr_imgs = {}
    lr_up_imgs = {}
    hr_imgs = {}
    for time_col in range(times):
        try:
            img = data[time_col].astype(np.float).copy()
            img_max = img.max() * max_scale
            if img_max == 0: continue  #do not save images with no contents.
            img /= img_max
            down_img, down_up_img = crappify(img)
        except:
            continue

        tag = (0, 0, time_col)
        img = img_as_ubyte(img)
        pimg = PIL.Image.fromarray(img, mode='L')
        small_img = PIL.Image.fromarray(img_as_ubyte(down_img))
        big_img = PIL.Image.fromarray(img_as_ubyte(down_up_img))
        hr_imgs[tag] = pimg
        lr_imgs[tag] = small_img
        lr_up_imgs[tag] = big_img

    np.warnings.filterwarnings('default')
    return hr_imgs, lr_imgs, lr_up_imgs


def tif_to_synth(tif_fn,
                 dest,
                 category,
                 mode,
                 single=True,
                 multi=False,
                 num_frames=5,
                 max_scale=1.05,
                 crappify_func=None):
    img = PIL.Image.open(tif_fn)
    n_frames = img.n_frames

    if crappify_func is None: crappify_func = new_crappify
    for i in range(n_frames):
        img.seek(i)
        img.load()
        data = np.array(img).copy()

        hr_imgs, lr_imgs, lr_up_imgs = img_data_to_tifs(data,
                                                        n_frames,
                                                        crappify_func,
                                                        max_scale=max_scale)
        if single:
            save_tiffs(tif_fn, dest, category, mode, hr_imgs, lr_imgs,
                       lr_up_imgs)
        if multi:
            save_movies(tif_fn, dest, category, mode, hr_imgs, lr_imgs,
                        lr_up_imgs, num_frames)


def save_tiffs(czi_fn, dest, category, mode, hr_imgs, lr_imgs, lr_up_imgs):
    hr_dir = dest / 'hr' / mode
    lr_dir = dest / 'lr' / mode
    lr_up_dir = dest / 'lr_up' / mode
    base_name = czi_fn.stem
    for tag, hr in hr_imgs.items():
        lr = lr_imgs[tag]
        lr_up = lr_up_imgs[tag]

        channel, depth, time_col = tag
        save_name = f'{base_name}_{category}_{channel:02d}_{depth:02d}_{time_col:06d}.tif'
        hr_name, lr_name, lr_up_name = [
            d / save_name for d in [hr_dir, lr_dir, lr_up_dir]
        ]
        if not hr_name.exists(): hr.save(hr_name)
        if not lr_name.exists(): lr.save(lr_name)
        if not lr_up_name.exists(): lr_up.save(lr_up_name)


def save_movies(czi_fn, dest, category, mode, hr_imgs, lr_imgs, lr_up_imgs,
                num_frames):
    print('WTF save_movies is empty dude')
    print('*****', czi_fn)


def czi_movie_to_synth(czi_fn,
                       dest,
                       category,
                       mode,
                       single=True,
                       multi=False,
                       num_frames=5,
                       max_scale=1.05,
                       crappify_func=None):
    with czifile.CziFile(czi_fn) as czi_f:
        if crappify_func is None: crappify_func = new_crappify

        do_multi = multi and is_movie(czi_f)
        axes, shape = get_czi_shape_info(czi_f)
        data = czi_f.asarray()

        hr_imgs, lr_imgs, lr_up_imgs = czi_data_to_tifs(data,
                                                        axes,
                                                        shape,
                                                        crappify_func,
                                                        max_scale=max_scale)
        if single:
            save_tiffs(czi_fn, dest, category, mode, hr_imgs, lr_imgs,
                       lr_up_imgs)
        if do_multi:
            save_movies(czi_fn, dest, category, mode, hr_imgs, lr_imgs,
                        lr_up_imgs, num_frames)

