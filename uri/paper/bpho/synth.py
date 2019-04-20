"functions to create synthetic low res images"
import numpy as np
import czifile
import PIL
import random
from skimage.util import random_noise, img_as_ubyte
from skimage import filters
from skimage.io import imsave
from scipy.ndimage.interpolation import zoom as npzoom
from .czi import get_czi_shape_info, build_index, is_movie
from .utils import ensure_folder
from fastai.vision import *

__all__ = ['speckle_crap', 'czi_movie_to_synth', 'tif_movie_to_synth']


def speckle_crap(img):
    img = random_noise(img, mode='speckle', var=0.02, clip=True)
    return img


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



def draw_tile(img, tile_sz):
    max_x,max_y = img.shape
    x = random.choice(range(max_x-tile_sz)) if max_x > tile_sz else 0
    y = random.choice(range(max_y-tile_sz)) if max_y > tile_sz else 0
    xs = slice(x,min(x+tile_sz, max_x))
    ys = slice(y,min(y+tile_sz, max_y))
    tile = img[xs,ys].copy()
    return tile, (xs,ys)

def draw_tile_bounds(img, bounds):
    xs,ys = bounds
    tile = img[xs,ys].copy()
    return tile


def save_img(fn, img):
    # np.warnings.filterwarnings('ignore')
    # PIL.Image.fromarray(img_as_ubyte(img), mode='L').save(f'{fn}.tif')
    # np.warnings.filterwarnings('default')
    np.save(fn, img, allow_pickle=False)

def czi_movie_to_synth(czi_fn,
                       dest,
                       category,
                       mode,
                       single=True,
                       multi=False,
                       tiles=None,
                       scale=4,
                       n_tiles=5,
                       n_frames=5,
                       crappify_func=None):
    hr_dir = ensure_folder(dest / 'hr' / mode)
    lr_dir = ensure_folder(dest / 'lr' / mode)
    lrup_dir = ensure_folder(dest / 'lrup' / mode)
    base_name = czi_fn.stem

    with czifile.CziFile(czi_fn) as czi_f:
        data = czi_f.asarray()
        axes, shape = get_czi_shape_info(czi_f)
        channels = shape['C']
        depths = shape['Z']
        times = shape['T']
        x,y = shape['X'], shape['Y']

        for channel in range(channels):
            for depth in range(depths):
                for t in range(times):
                    save_name = f'{base_name}_{category}_{channel:02d}_{depth:02d}_{t:06d}'
                    idx = build_index( axes, {'T': t, 'C':channel, 'Z': depth, 'X':slice(0,x), 'Y':slice(0,y)})
                    img_data = data[idx].astype(np.float32).copy()
                    img_max = img_data.max()
                    if img_max != 0: img_data /= img_max


                    image_to_synth(img_data, dest, mode, hr_dir, lr_dir, lrup_dir, save_name, 
                                   single, multi, tiles, n_tiles, n_frames, scale, crappify_func)


def tif_movie_to_synth(tif_fn,
                       dest,
                       category,
                       mode,
                       single=True,
                       multi=False,
                       tiles=None,
                       scale=4,
                       n_tiles=5,
                       n_frames=5,
                       crappify_func=None):
    hr_dir = ensure_folder(dest / 'hr' / mode)
    lr_dir = ensure_folder(dest / 'lr' / mode)
    lrup_dir = ensure_folder(dest / 'lrup' / mode)
    base_name = tif_fn.stem

    img = PIL.Image.open(tif_fn)
    n_frames = img.n_frames

    with PIL.Image.open(tif_fn) as img:
        channels = 1
        depths = img.n_frames
        times = 1

        for channel in range(channels):
            for depth in range(depths):
                for t in range(times):
                    save_name = f'{base_name}_{category}_{channel:02d}_{depth:02d}_{t:06d}'
                    img.seek(depth)
                    img.load()
                    img_data = np.array(img).astype(np.float32).copy()
                    img_max = img_data.max()
                    if img_max != 0: img_data /= img_max

                    image_to_synth(img_data, dest, mode, hr_dir, lr_dir, lrup_dir, save_name, 
                                   single, multi, tiles, n_tiles, n_frames, scale, crappify_func)



def check_tile(img, thresh, thresh_pct):
    return (img > thresh).mean() > thresh_pct



def image_to_synth(img_data, dest, mode, hr_dir, lr_dir, lrup_dir, save_name, single, multi, tiles, n_tiles, n_frames, scale, crappify_func):
    if len(img_data.shape) > 2:
        if len(img_data.shape) == 3:
            img_data = img_data[:,:,0]
        else:
            print(f'skip {save_name} multichannel')
            return

    h,w = img_data.shape
    adjh, adjw = (h//4) * 4, (w//4)*4
    hr_img = img_data[0:adjh, 0:adjw]

    crap_img = crappify_func(hr_img).astype(np.float32).copy() if crappify_func else hr_img 
    lr_img = npzoom(crap_img, 1/scale, order=0).astype(np.float32).copy()
    lrup_img = npzoom(lr_img, scale, order=0).astype(np.float32).copy()
    if single:
        hr_name, lr_name, lrup_name = [d / save_name for d in [hr_dir, lr_dir, lrup_dir]]
        save_img(hr_name, hr_img)
        save_img(lr_name, lr_img)
        save_img(lrup_name, lrup_img)

    if tiles:
        for tile_sz in tiles:
            hr_tile_dir = ensure_folder(dest/f'hr_t_{tile_sz}'/mode)
            lr_tile_dir = ensure_folder(dest/f'lr_t_{tile_sz}'/mode)
            lrup_tile_dir = ensure_folder(dest/f'lrup_t_{tile_sz}'/mode)

            tile_id = 0
            tries = 0
            max_tries = 200
            thresh = 0.01
            thresh_pct = (hr_img > thresh).mean() * 1.5
            while tile_id < n_tiles:
                hr_tile, bounds = draw_tile(hr_img, tile_sz)
                if check_tile(hr_tile, thresh, thresh_pct):
                    tile_name = f'{save_name}_{tile_id:03d}'
                    hr_tile_name, lr_tile_name, lrup_tile_name = [d / tile_name for d 
                                                                in [hr_tile_dir, lr_tile_dir, lrup_tile_dir]]
                    crap_tile = draw_tile_bounds(crap_img, bounds=bounds)
                    lr_tile = npzoom(crap_tile, 1/scale, order=0).astype(np.float32).copy()
                    lrup_tile = npzoom(lr_tile, scale, order=0).astype(np.float32).copy()
                    save_img(hr_tile_name, hr_tile)
                    save_img(lr_tile_name, lr_tile)
                    save_img(lrup_tile_name, lrup_tile)
                    tile_id += 1
                    tries = 0
                else:
                    tries += 1
                    if tries > (max_tries//2):
                        thresh_pct /= 2
                    if tries > max_tries:
                        print(f'timed out on {save_name}')
                        tries = 0
                        tile_id += 1
