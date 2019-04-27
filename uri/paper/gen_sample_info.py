"build combo dataset"
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

def process_czi(item, category, mode):
    tif_srcs = []
    base_name = item.stem
    with czifile.CziFile(item) as czi_f:
        data = czi_f.asarray()
        axes, shape = get_czi_shape_info(czi_f)
        channels = shape['C']
        depths = shape['Z']
        times = shape['T']

        x,y = shape['X'], shape['Y']

        mid_depth = depths // 2
        depth_range = range(max(0,mid_depth-2), min(depths, mid_depth+2))
        is_multi = (times > 1) or (depths > 1)
        for channel in range(channels):
            for z in depth_range:
                for t in range(times):
                    tif_srcs.append({'fn': item, 'ftype': 'czi', 'multi':int(is_multi), 'category': category, 'dsplit': mode,
                                     'nc': channels, 'nz': depths, 'nt': times,
                                     'z': mid_depth, 't': t, 'c':channel, 'x': x, 'y': y})
    return tif_srcs


def process_tif(item, category, mode):
    tif_srcs = []
    img = PIL.Image.open(item)
    n_frames = img.n_frames
    x,y = img.size
    is_multi = n_frames > 1
    for z in range(n_frames):
        tif_srcs.append({'fn': item, 'ftype': 'tif', 'multi':int(is_multi), 'category': category, 'dsplit': mode,
                         'nc': 1, 'nz': n_frames, 'nt': 1,
                         'z': z, 't': 0, 'c':0, 'x': x, 'y': y})

    return tif_srcs

def process_unk(item, category, mode):
    print(f"**** WTF: {item}")
    return []

def process_item(item, category, mode):
    try:
        if mode == 'test': return []
        else:
            item_map = {
                '.tif': process_tif,
                '.tiff': process_tif,
                '.czi': process_czi,
            }
            map_f = item_map.get(item.suffix, process_unk)
            return map_f(item, category, mode)
    except Exception as ex:
        print(f'err procesing: {item}')
        print(ex)
        return []

def build_tifs(src, mbar=None):
    tif_srcs = []
    for mode in ['train', 'valid', 'test']:
        src_dir = src / mode
        category = src.stem
        items = list(src_dir.iterdir()) if src_dir.exists() else []
        if items:
            for p in progress_bar(items, parent=mbar):
                mbar.child.comment = mode
                tif_srcs += process_item(p, category=category, mode=mode)
    return tif_srcs

@call_parse
def main(out: Param("tif source name", Path, required=True),
         sources: Param('src folders', Path, nargs='...', opt=False) = None,
         only: Param('whitelist subfolders to include', str, nargs='+') = None,
         skip: Param("subfolders to skip", str, nargs='+') = None):

    "generate comobo dataset"
    if skip and only:
        print('you can skip subfolder or whitelist them but not both')
        return 1

    src_dirs = []
    for src in sources:
        sub_fldrs = subfolders(src)
        if skip:  src_dirs += [fldr for fldr in sub_fldrs if fldr.stem not in skip]
        elif only: src_dirs += [fldr for fldr in sub_fldrs if fldr.stem in only]
        else: src_dirs += sub_fldrs

    mbar = master_bar(src_dirs)
    tif_srcs = []
    for src in mbar:
        mbar.write(f'process {src.stem}')
        tif_srcs += build_tifs(src, mbar=mbar)

    tif_src_df = pd.DataFrame(tif_srcs)
    tif_src_df[['category','dsplit','multi','ftype','nc','nz','nt','c','z','t','x','y','fn']].to_csv(out, header=True, index=False)