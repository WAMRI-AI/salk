
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


def default_crap(img, scale=4, upsample=True, pscale=12, gscale=0.0001):
    from skimage.transform import rescale

    # pull pixed data from PIL image
    x = np.array(img)
    multichannel = len(x.shape) > 2

    # remember the range of the original image
    img_max = x.max()

    # normalize 0-1.0
    x = x.astype(np.float32)
    x /= float(img_max)

    # downsample the image
    x = rescale(x, scale=1/scale, order=1, multichannel=multichannel)

    # add poisson and gaussian noise to the image
    x = np.random.poisson(x * pscale)/pscale
    x += np.random.normal(0, gscale, size=x.shape)
    x = np.maximum(0,x)

    # normalize to 0-1.0
    x /= x.max()

    # back to original range of incoming image
    x *= img_max

    if upsample: x = rescale(x, scale=scale, order=0, multichannel=multichannel)

    return PIL.Image.fromarray(x.astype(np.uint8))


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

    tile_infos = []
    crap_dirs = {}

    for tile_sz in tile:
        hr_dir = ensure_folder(out/f'hr_t_{tile_sz}')
        crap_dirs[tile_sz] = ensure_folder(out/f'lr{up}_t_{tile_sz}') if crap_func else None

        tile_info = build_tile_info(sources, tile_sz, n_train, n_valid, only_categories=only, skip_categories=skip)
        tile_infos.append(tile_info)
    print(tile_infos[0].groupby('category').fn.count())

    tile_df = pd.concat(tile_infos)
    generate_tiles(hr_dir, tile_df, scale=scale, crap_dirs=crap_dirs, crap_func=crap_func)
    print('finished tiles')
