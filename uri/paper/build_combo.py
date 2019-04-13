"build combo dataset"

from fastai.script import *
from bpho import *
from pathlib import Path
from fastprogress import master_bar, progress_bar
from time import sleep
from pdb import set_trace
import shutil

def process_czi(item, mode, dest, single, multi, num_frames=5):
    hr_dir = dest/'hr'/mode
    lr_dir = dest/'lr'/mode
    lr_up_dir = dest/'lr_up'/mode

    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    lr_up_dir.mkdir(parents=True, exist_ok=True)
    czi_movie_to_synth(item, dest, mode, single=single, multi=multi, num_frames=num_frames)

def process_tif(item, mode, dest, single, multi):
    print(f"we don't handle tiff yet {item.name}")

def process_unk(item, mode, dest, single, multi):
    print(f'unknown {item.name}')

def process_item(item, mode, dest, single=True, multi=False):
    item_map = {
        '.tif': process_tif,
        '.tiff': process_tif,
        '.czi': process_czi,
    }
    map_f = item_map.get(item.suffix, process_unk)
    map_f(item, mode, dest, single, multi)



def build_from_datasource(src, dest, single=True, multi=False, mbar=None):
    for mode in ['train', 'valid', 'test']:
        src_dir = src/mode
        dest_dir = dest
        items = list(src_dir.iterdir()) if src_dir.exists() else []
        if items:
            for p in progress_bar(items, parent=mbar):
                process_item(p, mode, dest_dir, single=single, multi=multi)

def subfolders(p):
    return [sub for sub in p.iterdir() if sub.is_dir()]

@call_parse
def main(
    name: Param("dataset name", str, opt=False),
    singleframe: Param("multiframe dataset", bool)=True,
    multiframe: Param("multiframe dataset", bool)=False,
    skip: Param("data to skip", str)='random',
    dest: Param("destination dir", str)='datasets',
    src: Param("data source dir", str)='data',
    clean: Param("wipe existing data first", bool)=False
):
    "generate comobo dataset"
    dest = Path(dest)/name
    src = Path(src)
    if dest.exists() and clean: shutil.rmtree(dest)
    if not dest.exists(): dest.mkdir(parents=True, mode=0o775, exist_ok=True)
    
    skip =  skip.split(',') if skip else []
    live_data = [fldr for fldr in subfolders(src/'live') if fldr.stem not in skip]
    fixed_data = [fldr for fldr in subfolders(src/'fixed') if fldr.stem not in skip]
    #  print([fn.stem for fn in fixed_data])
    #  print([fn.stem for fn in live_data])
      
    sources = live_data + fixed_data
    mbar = master_bar(sources)    
    for src in mbar:
        build_from_datasource(src, dest, single=singleframe, multi=multiframe, mbar=mbar)


