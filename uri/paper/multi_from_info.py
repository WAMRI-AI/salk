
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
import czifile
PIL.Image.MAX_IMAGE_PIXELS = 99999999999999


def default_crap(img, scale=4, upsample=True, pscale=12, gscale=0.0001):
    from skimage.transform import rescale


    # set_trace()
    # pull pixel data from PIL image
    x = np.array(img)
    multichannel = len(x.shape) > 2

    # remember the range of the original image
    dmax = np.iinfo(x.dtype).max
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

    x /= dmax
    x *= np.iinfo(np.uint8).max
    return PIL.Image.fromarray(x.astype(np.uint8))

def need_cache_flush(tile_stats, last_stats):
    if last_stats is None: return True
    if tile_stats['fn'] != last_stats['fn']: return True
    # for fld in ['c','z','t']:
    #     if tile_stats[fld] != last_stats[fld]:
    #         return True
    return False

def get_tile_puller(tile_stat, crap_func, n_frames):
    fn = tile_stat['fn']
    ftype = tile_stat['ftype']
    nz = tile_stat['nz']
    nt = tile_stat['nt']
    half_frames = n_frames // 2

    if ftype == 'czi':
        img_f = czifile.CziFile(fn)
        proc_axes, proc_shape = get_czi_shape_info(img_f)
        img_data = img_f.asarray()
        img_max = img_data.max() # np.iinfo(img_data.dtype).max
        img_data = img_data.astype(np.float32)

        def czi_get(istat):
            c,z,t,x,y,mi,ma = [istat[fld] for fld in ['c','z','t','x','y','mi','ma']]
            if nz > n_frames:
                idx = build_index(
                    proc_axes, {
                        'T': t,
                        'C': c,
                        'Z': slice(z-half_frames, z+half_frames+1),
                        'X': slice(0, x),
                        'Y': slice(0, y)
                    })
                type_frames = 'Z'
            else:
                idx = build_index(
                    proc_axes, {
                        'T': slice(t-half_frames, t+half_frames+1),
                        'C': c,
                        'Z': z,
                        'X': slice(0, x),
                        'Y': slice(0, y)
                    })
                type_frames = 'T'

            img = img_data[idx].copy()
            # img /= img_max
            eps = 1e-20
            img = (img - mi) / (ma - mi + eps)
            img = img.clip(0,1.)
            # return img.clip(0.,1.)
            return img, type_frames

        img_get = czi_get
        img_get._to_close = img_f
    else:
        pil_img = PIL.Image.open(fn)
        n_frames = pil_img.n_frames

        img = np.array(pil_img)
        if len(img.shape) > 2: img = img[:,:,0]
        img_max = img.max() # np.iinfo(img.dtype).max
        img = img.astype(np.float32) / img_max


        def pil_get(istat):
            type_frams = 'Z'
            c,z,t,x,y,mi,ma = [istat[fld] for fld in ['c','z','t','x','y','mi','ma']]
            start_z, end_z = z - half_frames, z + half_framesi + 1

            img_array = []
            for iz in range(start_z, end_z):
                pil_img.seek(z)
                pil_img.load()
                img = np.array(pil_img)
                if len(img.shape) > 2: img = img[:,:,0]
                img_array.append(img.copy())

            # img_max = img.max()
            # img = img.astype(np.float32) / img_max
            img = np.stack(img_array)
            img = img.astype(np.float32)
            eps = 1e-20
            img = (img - mi) / (ma - mi + eps)
            img = img.clip(0,1.)
            return img, type_frames

        img_get = pil_get
        img_get._to_close = pil_img

    def puller(istat, z_tile_folder, t_tile_folder, z_crap_folder, t_crap_folder, close_me=False):
        if close_me:
            img_get._to_close.close()
            return None

        id = istat['index']
        fn = Path(istat['fn'])
        tile_sz = istat['tile_sz']
        c,z,t,x,y,mi,ma = [istat[fld] for fld in ['c','z','t','x','y','mi','ma']]

        raw_data, type_frames = img_get(istat)
        if type_frames == 'Z':
            tile_folder = z_tile_folder
            crap_folder = z_crap_folder
        else:
            assert(type_frames == 'T')
            tile_folder = t_tile_folder
            crap_folder = t_crap_folder

        img_data = (np.iinfo(np.uint8).max * raw_data).astype(np.uint8)

        thresh = np.percentile(img_data, 2)
        thresh_pct = (img_data > thresh).mean() * 0.75

        mid_frame = n_frames // 2
        crop_img, box = draw_random_tile(img_data[mid_frame], istat['tile_sz'], thresh, thresh_pct)
        crop_img.save(tile_folder/f'{id:06d}_{fn.stem}.tif')
        if crap_func and crap_folder:
            crap_data = []
            for i in range(n_frames):
                frame_img = PIL.Image.fromarray(img_data[i, box[1]:box[3], box[0]:box[2]])
                crap_frame = crap_func(crop_img)
                crap_data.append(np.array(crap_frame))
            multi_array = np.stack(crap_data)
            np.save(crap_folder/f'{id:06d}_{fn.stem}.tif', multi_array)

        info = dict(istat)
        info['id'] = id
        info['box'] = box
        info['tile_sz'] = tile_sz
        return info

    return puller

def check_info(info, n_frames):
    space = n_frames // 2
    if info['nz'] > n_frames:
        return (info['z'] > space) and (info['z'] < (info['nz']-space))
    elif info['nt'] > n_frames:
        return (info['t'] > space) and (info['t'] < (info['nt']-space))

    return False

@call_parse
def main(out: Param("dataset folder", Path, required=True),
         info: Param('info file', Path, required=True),
         tile: Param('generated tile size', int, nargs='+', required=True),
         n_train: Param('number of train tiles', int, required=True),
         n_valid: Param('number of validation tiles', int, required=True),
         crap_func: Param('crappifier name', object) = default_crap,
         n_frames: Param('number of input frames', int) = 5,
         scale: Param('amount to scale', int) = 4,
         ftypes: Param('ftypes allowed e.g. - czi, tif', str, nargs='+') = None,
         not_unet: Param('unet style (down and back upsample)', action='store_true') = False,
         only: Param('limit to these categories', nargs='+') = None,
         skip: Param("categories to skip", str, nargs='+') = ['random', 'ArgoSIMDL'],
         clean: Param("wipe existing data first", action='store_true') = False):
    "generate tiles from source tiffs"
    is_unet = not not_unet
    up = 'up' if is_unet else ''

    if not callable(crap_func):
        print('crap_func is not callable')
        crap_func = None
    else:
        crap_func = partial(crap_func, scale=scale, upsample=is_unet)

    out = ensure_folder(out)
    if clean:
        shutil.rmtree(out)

    info = pd.read_csv(info)

    if ftypes: info = info.loc[info.ftype.isin(ftypes)]
    if only: info = info.loc[info.category.isin(only)]
    elif skip: info = info.loc[~info.category.isin(skip)]

    # filter to multi
    max_frames = info[['nt','nz']].apply(max, axis=1)
    info = info.loc[max_frames > n_frames]

    tile_infos = []

    for mode, n_samples in [('train', n_train),('valid', n_valid)]:
        mode_info = info.loc[info.dsplit == mode]
        categories = list(mode_info.groupby('category'))
        files_by_category  = {c:list(info.groupby('fn')) for c,info in categories}

        for i in range(n_samples):
            category, cat_df = random.choice(categories)
            fn, item_df = random.choice(files_by_category[category])
            legal_choices = [item_info for ix, item_info in item_df.iterrows() if check_info(item_info, n_frames)]
            if not legal_choices:
                set_trace()
            item_info = random.choice(legal_choices)
            for tile_sz in tile:
                item_d = dict(item_info)
                item_d['tile_sz'] = tile_sz
                tile_infos.append(item_d)

    tile_info_df = pd.DataFrame(tile_infos).reset_index()
    print('num tile pulls:', len(tile_infos))
    print(tile_info_df.groupby('category').fn.count())

    last_stat = None
    tile_pull_info = []
    tile_puller = None

    mbar = master_bar(tile_info_df.groupby('fn'))
    for fn, tile_stats in mbar:
        if Path(fn).stem == 'high res microtubules for testing before stitching - great quality':
            continue
        for i, tile_stat in progress_bar(list(tile_stats.iterrows()), parent=mbar):
            try:
                mode = tile_stat['dsplit']
                category = tile_stat['category']
                tile_sz = tile_stat['tile_sz']
                z_tile_folder = ensure_folder(out / f'hr_t_{tile_sz}_mz_{n_frames}' / mode / category)
                t_tile_folder = ensure_folder(out / f'hr_t_{tile_sz}_mt_{n_frames}' / mode / category)
                if crap_func:
                    z_crap_folder = ensure_folder(out / f'lr{up}_t_{tile_sz}_mz_{n_frames}' / mode / category)
                    t_crap_folder = ensure_folder(out / f'lr{up}_t_{tile_sz}_mt_{n_frames}' / mode / category)
                else: crap_folder = None

                if need_cache_flush(tile_stat, last_stat):
                    if tile_puller:
                        tile_puller(None, None, None, close_me=True)
                    last_stat = tile_stat.copy()
                    tile_sz = tile_stat['tile_sz']
                    tile_puller = get_tile_puller(tile_stat, crap_func, n_frames)
                tile_pull_info.append(tile_puller(tile_stat, z_tile_folder, t_tile_folder, z_crap_folder, t_crap_folder))
            except MemoryError as error:
                # some files are too big to read
                fn = Path(tile_stat['fn'])
                print(f'too big: {fn.stem}')

    pd.DataFrame(tile_pull_info).to_csv(out/'tiles.csv', index = False)

