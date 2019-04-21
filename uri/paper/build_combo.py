"build combo dataset"
import yaml
yaml.warnings({'YAMLLoadWarning': False})

from fastai.script import *
from bpho import *
from pathlib import Path
from fastprogress import master_bar, progress_bar

from time import sleep
from pdb import set_trace
import shutil
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 99999999999999

def process_czi(item, category, mode, dest, single, multi, tiles, scale, n_tiles, n_frames, crappify_func):
    czi_movie_to_synth(item,
                       dest,
                       category,
                       mode,
                       single=single,
                       multi=multi,
                       tiles=tiles,
                       scale=scale,
                       n_tiles=n_tiles,
                       n_frames=n_frames,
                       crappify_func=crappify_func)


def process_tif(item, category, mode, dest, single, multi, tiles, scale, n_tiles, n_frames, crappify_func):
    tif_movie_to_synth(item,
                       dest,
                       category,
                       mode,
                       single=single,
                       multi=multi,
                       tiles=tiles,
                       scale=scale,
                       n_tiles=n_tiles,
                       n_frames=n_frames,
                       crappify_func=crappify_func)


def process_unk(item, category, mode, dest, single, multi, tiles, scale, n_tiles, n_frames, crappify_func):
    print(f'unknown {item.name}')


def just_copy(item, category, mode, dest):
    cat_dest = ensure_folder(dest / mode / category)
    sleep(0.01)
    return
    shutil.copy(item, cat_dest / item.name)


def process_item(item, category, mode, dest, single, multi, tiles, scale, n_tiles, n_frames, crappify_func):
    if mode == 'test': just_copy(item, category, mode, dest)
    else:
        item_map = {
            '.tif': process_tif,
            '.tiff': process_tif,
            '.czi': process_czi,
        }
        map_f = item_map.get(item.suffix, process_unk)
        map_f(item, category, mode, dest, single, multi, tiles, scale, n_tiles, n_frames, crappify_func)


def build_from_datasource(src, dest, single=False, multi=False, tiles=None, scale=4, 
                          n_tiles=5, n_frames=5, crappify_func=None, mbar=None):
    for mode in ['train', 'valid', 'test']:
        src_dir = src / mode
        dest_dir = dest
        category = src.stem
        items = list(src_dir.iterdir()) if src_dir.exists() else []
        if items:
            for p in progress_bar(items, parent=mbar):
                mbar.child.comment = mode 
                try:
                    process_item(p,
                                 category,
                                 mode,
                                 dest_dir,
                                 single=single,
                                 multi=multi,
                                 tiles=tiles,
                                 scale=scale,
                                 n_tiles=n_tiles,
                                 n_frames=n_frames,
                                 crappify_func=crappify_func)
                except:
                    print(f'err procesing: {p}')

@call_parse
def main(out: Param("dataset folder", Path, required=True),
         sources: Param('src folders', Path, nargs='...', opt=False) = None,
         single: Param("single dataset", action='store_true') = False,
         multi: Param("multiframe dataset", action='store_true') = False,
         tile: Param("insteger list of tiles to create, like '64 128 256'", int, nargs='+') = None,
         scale: Param('amount to scale', int) = 4,
         n_tiles: Param("number of tiles per image", int) = 5,
         n_frames: Param("number of frames per multi-image", int) = 5,
         only: Param('whitelist subfolders to include', str, nargs='+') = None,
         skip: Param("subfolders to skip", str, nargs='+') = None,
         crappify: Param("subfolders to skip", str) = None,
         clean: Param("wipe existing data first", action='store_true') = False):
    "generate comobo dataset"

    if not (single or multi or tile):
        print('you need to specify one or more output formats')
        return 1

    if skip and only:
        print('you can skip subfolder or whitelist them but not both')
        return 1

    out = ensure_folder(out, clean=clean)

    if crappify: 
        crappify_func = eval(crappify)

    src_dirs = []
    for src in sources:
        sub_fldrs = subfolders(src)
        if skip:  src_dirs += [fldr for fldr in sub_fldrs if fldr.stem not in skip]
        elif only: src_dirs += [fldr for fldr in sub_fldrs if fldr.stem in only]
        else: src_dirs += sub_fldrs

    mbar = master_bar(src_dirs)
    for src in mbar:
        mbar.write(f'process {src.stem}')
        build_from_datasource(src,
                              out,
                              single=single,
                              multi=multi,
                              tiles=tile,
                              scale=scale,
                              n_tiles=n_tiles,
                              n_frames=n_frames,
                              crappify_func=crappify_func,
                              mbar=mbar)
