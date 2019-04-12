"build combo dataset"

from fastai.script import *
from bpho import *
from pathlib import Path
from fastprogress import master_bar, progress_bar
from time import sleep
from pdb import set_trace


def process_item(item, mode, dest, multiframe=False):
    sleep(0.01)

def build_from_datasource(src, dest, multiframe=False, mbar=None):
    for mode in ['train', 'valid', 'test']:
        src_dir = src/mode
        dest_dir = dest/mode
        items = list(src_dir.iterdir()) if src_dir.exists() else []
        if items:
            if mode == 'test':
                print(src.stem)
            for p in progress_bar(items, parent=mbar):
                process_item(p, mode, dest_dir, multiframe=multiframe)

def subfolders(p):
    return [sub for sub in p.iterdir() if sub.is_dir()]

@call_parse
def main(
    name: Param("dataset name", str, opt=False),
    no_neuron: Param("disable neuron data", bool)=False,
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
    print([fn.stem for fn in fixed_data])
    print([fn.stem for fn in live_data])
      
    sources = live_data
    if not multiframe: sources += fixed_data

    mbar = master_bar(sources)    
    for src in mbar:
        build_from_datasource(src, dest, multiframe=multiframe, mbar=mbar)


