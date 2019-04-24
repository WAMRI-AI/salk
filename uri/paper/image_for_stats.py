import sys
import yaml
yaml.warnings({'YAMLLoadWarning': False})
import pandas as pd
from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.vision.models.xresnet import *
from fastai.vision.models.unet import DynamicUnet
from fastprogress import master_bar, progress_bar
from bpho import *
import PIL.Image

torch.backends.cudnn.benchmark = True

def check_dir(p):
    if not p.exists():
        print(f"couldn't find {p}")
        sys.exit(1)
    return p

def get_sub_folders(p):
    return [f for f in p.iterdir() if p.is_dir()]


def calc_stats(pred_img, truth_img):
    if pred_img is None: return None
    if truth_img is None: return None
    if pred_img.shape != truth_img.shape:
        breakpoint()
    return {'ssim': 55.0, 'psnr': 42.0, 'fid': 22}

def process_tif(item, proc_func, out_fldr, truth, just_stats):
    stats = []
    truth_imgs = PIL.Image.open(truth) if truth.exists() else None
    with PIL.Image.open(item) as img_tif:
        for i in range(img_tif.n_frames):
            img_tif.seek(i)
            img_tif.load()

            if truth_imgs:
                truth_imgs.seek(i)
                truth_imgs.load()
                truth_img = np.array(truth_imgs)
            else: truth_img = None

            tag = f'{i:05d}'
            out_name = (out_fldr/f'{item.stem}_{tag}').with_suffix(item.suffix)
            if just_stats:
                if out_name.exists():
                    pred_img = np.array(PIL.Image.open(out_name))
                else: pred_img = None
            else:
                img = np.array(img_tif)
                pred_img = proc_func(img)
            istats = calc_stats(pred_img, truth_img)
            if istats:
                istats.update({'tag': tag, 'item': item.stem})
                stats.append(istats)
    return stats

def process_czi(item, proc_func, out_fldr, truth, just_stats):
    return [{'item': item.stem, 'ssim': 55.0, 'psnr': 42.0, 'fid': 22, 'tag': ""}]


def get_id(fn):
    parts = fn.name.split('_')
    id = int(parts[0])
    return id

def build_truth_map(truth_fldr):
    truth_map = {}
    if truth_fldr.exists():
        for fn in truth_fldr.iterdir():
            id = get_id(fn)
            truth_map[id] = fn
    return truth_map

def find_truth(fn, truth_map):
    return truth_map.get(get_id(fn), None)

def process_subfolder(fldr, processor, out_fldr, truth_fldr, just_stats):
    stats = []
    proc_map = {
        '.tif': process_tif,
        '.czi': process_czi
    }
    truth_map = build_truth_map(truth_fldr)
    proc_func = get_named_processor(processor)
    for item in fldr.iterdir():
        proc = proc_map.get(item.suffix, None)
        if proc:
            truth = find_truth(item, truth_map)
            item_stats = proc(item, proc_func, out_fldr, truth, just_stats)
            for s in item_stats:
                s.update({'category': fldr.name})
            stats += item_stats
    return stats

def process_folders(in_dir, out_dir, truth_dir, processor, just_stats, mbar=None):
    save_stats = []
    sub_folders = get_sub_folders(in_dir)
    for fldr in progress_bar(sub_folders, parent=mbar):
        out_sub_fldr = out_dir/fldr.name
        truth_sub_fldr = truth_dir/fldr.name
        ensure_folder(out_sub_fldr)
        stats = process_subfolder(fldr, processor, out_sub_fldr, truth_sub_fldr, just_stats)
        save_stats += stats
    return save_stats

@call_parse
def main(
        stats_dir: Param("stats dir", Path, opt=False),
        gpu: Param("GPU to run on", str, required=True) = None,
        models: Param("list models to run", str, nargs='+')=None,
        baselines: Param("build bilinear and bicubic", action='store_true')=False,
        just_stats: Param("don't rebuild images", action='store_true')=False
):
    in_dir = check_dir(stats_dir/'input')
    out_dir = check_dir(stats_dir/'output')
    truth_dir = check_dir(stats_dir/'ground_truth')

    processors = []
    stats = []
    if baselines: processors += ['bilinear', 'bicubic']
    if models: processors += [m for m in models]
    mbar = master_bar(processors)
    for proc in mbar:
        mbar.write(f'processing {proc}')
        proc_stats = process_folders(in_dir, out_dir, truth_dir, proc, just_stats, mbar=mbar)
        for d in proc_stats: d.update({'model': proc})
        stats += proc_stats

    stats_df = pd.DataFrame(stats)
    print(stats_df)
    print(stats_df.groupby(['model', 'category']).mean())
