
import yaml
yaml.warnings({'YAMLLoadWarning': False})

from fastai.script import *
from fastai.vision import *
from bpho import *
from pathlib import Path
from fastprogress import master_bar, progress_bar

from time import sleep
from pdb import set_trace
import shutil
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 99999999999999

from scipy.ndimage.interpolation import zoom as npzoom
from skimage.util import random_noise, img_as_ubyte
from skimage import filters

def default_crap(img, scale=4, upsample=True, gauss_sigma=1, poisson_loop=10):
    h,w = img.size
    x = np.array(img)
    for n in range(poisson_loop):
        x = np.random.poisson(np.maximum(0,x).astype(np.int))
    x = x.astype(np.float32)
    noise = np.random.normal(0,gauss_sigma,size=x.shape).astype(np.float32)
    x = np.maximum(0,x+noise)
    x -= x.min()
    x /= x.max()
    x *= 255.
    x = x.astype(np.uint8)
    x = npzoom(x, 1/scale, order=0)
    if upsample:
        new_scale = max(float(h)/x.shape[0], float(w)/x.shape[1])
        x = npzoom(x, scale, order=0)
    crap_img = PIL.Image.fromarray(x.astype(np.uint8))
    return crap_img

@call_parse
def main(out: Param("dataset folder", Path, required=True),
         sources: Param('src folder', Path, required=True),
         tile: Param('generated tile size', int, nargs='+', required=True),
         n_train: Param('number of train tiles', int, required=True),
         n_valid: Param('number of validation tiles', int, required=True),
         crap_func: Param('crappifier name', object) = default_crap,
         scale: Param('amount to scale', int) = 4,
         not_unet: Param('unet style (down and back upsample)', action='store_true') = False,
         only: Param('limit to these categories', nargs='+') = None,
         skip: Param("categories to skip", str, nargs='+') = ['random', 'centrioles','ArgoSIMDL', 'neurons', 'fixed_neurons'],
         clean: Param("wipe existing data first", action='store_true') = False):
    "generate tiles from source tiffs"
    is_unet = not not_unet
    up = 'up' if is_unet else ''

    if not callable(crap_func):
        print('crap_func is not callable')
        crap_func = None
        crap_dir = None
    else:
        crap_func = partial(crap_func, scale=scale, upsample=is_unet)

    out = ensure_folder(out)
    if clean:
        shutil.rmtree(out)

    for tile_sz in tile:
        hr_dir = ensure_folder(out/f'hr_t_{tile_sz}')
        crap_dir = ensure_folder(out/f'lr{up}_t_{tile_sz}') if crap_func else None

        tile_info = build_tile_info(sources, tile_sz, n_train, n_valid, only_categories=only, skip_categories=skip)
        print(tile_info.groupby('category').fn.count())
        generate_tiles(hr_dir, tile_info, crap_dir=crap_dir, crap_func=crap_func)
    print('finished tiles')
