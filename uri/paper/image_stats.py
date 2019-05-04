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
import czifile

torch.backends.cudnn.benchmark = True

def check_dir(p):
    if not p.exists():
        print(f"couldn't find {p}")
        sys.exit(1)
    return p

def get_sub_folders(p):
    return [f for f in p.iterdir() if p.is_dir()]

def _norm_t(img):
    mi, ma = img.min(), img.max()
    nimg = (img - mi) / (ma - mi)
    return tensor(nimg[None,None])

def calc_stats(pred_img, truth_img):
    if pred_img is None: return None
    if truth_img is None: return None
    if pred_img.shape != truth_img.shape:
        h_offset = (pred_img.shape[0] - truth_img.shape[0]) // 2
        w_offset = (pred_img.shape[1] - truth_img.shape[1]) // 2
        pred_img = pred_img[h_offset:-h_offset, w_offset:-w_offset]
    if (pred_img.shape != truth_img.shape):
        print('unable to match input and ground truth sizes')
        breakpoint()


    t_pred, t_truth = _norm_t(pred_img), _norm_t(truth_img)
    ssim_val = ssim(t_pred, t_truth, window_size=10)
    psnr_val = psnr(t_pred, t_truth)
    return {'ssim': float(ssim_val), 'psnr': float(psnr_val), 'fid': 0.0}

def process_tif(item, proc_name, proc_func, out_fldr, truth, just_stats):
    stats = []
    truth_imgs = PIL.Image.open(truth) if truth and truth.exists() else None
    with PIL.Image.open(item) as img_tif:
        mid_frame = img_tif.n_frames // 2
        img_tif.seek(mid_frame)
        img_tif.load()

        if truth_imgs:
            truth_imgs.seek(mid_frame)
            truth_imgs.load()
            truth_img, truth_info = img_to_float(np.array(truth_imgs))
        else: truth_img = None

        tag = f'{mid_frame:05d}'
        save_name = f'{proc_name}_{item.stem}_{tag}'
        out_name = (out_fldr/save_name).with_suffix('.tif')
        pred_img = None
        if just_stats:
            if out_name.exists():
                pred_img, pred_info = img_to_float(np.array(PIL.Image.open(out_name)))

        if pred_img is None:
            img, img_info = img_to_float(np.array(img_tif))
            pred_img = proc_func(img, img_info=img_info)
            PIL.Image.fromarray(img_to_uint8(pred_img)).save(out_name)

        if not truth_img is None and not pred_img is None:
            truth_folder = ensure_folder(out_fldr/'../truth')
            truth_name = f'truth_{item.stem}_{tag}'
            PIL.Image.fromarray(img_to_uint8(truth_img)).save((truth_folder/truth_name).with_suffix('.tif'))
            istats = calc_stats(pred_img, truth_img)
            if istats:
                istats.update({'tag': tag, 'item': item.stem})
                stats.append(istats)
    return stats

def process_czi(item, proc_name, proc_func, out_fldr, truth, just_stats):
    stats = []
    truth_czi = czifile.CziFile(truth) if truth and truth.exists() else None
    with czifile.CziFile(item) as czi_f:
        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        channels = proc_shape['C']
        depths = proc_shape['Z']
        times = proc_shape['T']
        x, y = proc_shape['X'], proc_shape['Y']

        mid_time = times // 2
        mid_depth = depths // 2

        data, img_info = img_to_float(czi_f.asarray().astype(np.float32))

        if truth_czi:
            truth_data, truth_info = img_to_float(truth_czi.asarray())
            truth_proc_axes, truth_proc_shape = get_czi_shape_info(truth_czi)
            truth_x, truth_y = truth_proc_shape['X'], truth_proc_shape['Y']
        else: truth_data = None

        for c in range(channels):
            idx = build_index(
                proc_axes, {
                    'T': mid_time,
                    'C': c,
                    'Z': mid_depth,
                    'X': slice(0, x),
                    'Y': slice(0, y)
            })

            img = data[idx].copy()
            if not truth_data is None:
                truth_idx = build_index(
                    proc_axes, {
                        'T': mid_time,
                        'C': c,
                        'Z': mid_depth,
                        'X': slice(0, truth_x),
                        'Y': slice(0, truth_y)
                })
                truth_img = truth_data[truth_idx].copy()
            else:
                truth_img = None
            tag = f'{c:02d}_{mid_time:05d}_{mid_depth:05d}'
            save_name = f'{proc_name}_{item.stem}_{tag}'
            out_name = (out_fldr/save_name).with_suffix(".tif")

            pred_img = None
            if just_stats:
                if out_name.exists():
                    pred_img, pred_img_info = img_to_float(np.array(PIL.Image.open(out_name)))
            if pred_img is None:
                pred_img = proc_func(img, img_info=img_info)
                PIL.Image.fromarray(img_to_uint8(pred_img)).save(out_name)

            if not truth_img is None and not pred_img is None:
                truth_folder = ensure_folder(out_fldr/'../truth')
                truth_name = f'truth_{item.stem}_{tag}'
                PIL.Image.fromarray(img_to_uint8(truth_img)).save((truth_folder/truth_name).with_suffix('.tif'))
                istats = calc_stats(pred_img, truth_img)
                if istats:
                    istats.update({'tag': tag, 'item': item.stem})
                    stats.append(istats)
    return stats


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

def process_subfolder(fldr, processor, out_fldr, truth_fldr, model_dir, just_stats):
    stats = []
    proc_map = {
        '.tif': process_tif,
        '.czi': process_czi
    }
    truth_map = build_truth_map(truth_fldr)
    proc_func = get_named_processor(processor, model_dir)
    for item in fldr.iterdir():
        proc = proc_map.get(item.suffix, None)
        if proc:
            truth = find_truth(item, truth_map)
            item_stats = proc(item, processor, proc_func, ensure_folder(out_fldr/processor), truth, just_stats)
            for s in item_stats:
                s.update({'category': fldr.name})
            stats += item_stats
    return stats

def process_folders(in_dir, out_dir, truth_dir, model_dir, processor, just_stats, mbar=None):
    save_stats = []
    sub_folders = get_sub_folders(in_dir)
    for fldr in progress_bar(sub_folders, parent=mbar):
        out_sub_fldr = out_dir/fldr.name
        truth_sub_fldr = truth_dir/fldr.name
        ensure_folder(out_sub_fldr)
        stats = process_subfolder(fldr, processor, out_sub_fldr, truth_sub_fldr, model_dir, just_stats)
        save_stats += stats
    return save_stats

@call_parse
def main(
        stats_dir: Param("stats dir", Path, opt=False),
        gpu: Param("GPU to run on", int, required=True) = None,
        models: Param("list models to run", str, nargs='+')=None,
        baselines: Param("build bilinear and bicubic", action='store_true')=False,
        just_stats: Param("don't rebuild images", action='store_true')=False
):
    print('on gpu: ', gpu)
    torch.cuda.set_device(gpu)
    in_dir = check_dir(stats_dir/'input')
    out_dir = check_dir(stats_dir/'output')
    truth_dir = check_dir(stats_dir/'ground_truth')
    model_dir = check_dir(stats_dir/'models')

    processors = []
    stats = []
    if baselines: processors += ['bilinear', 'bicubic', 'original']
    if models: processors += [m for m in models]
    mbar = master_bar(processors)
    for proc in mbar:
        mbar.write(f'processing {proc}')
        proc_stats = process_folders(in_dir, out_dir, truth_dir, model_dir, proc, just_stats, mbar=mbar)
        for d in proc_stats: d.update({'model': proc})
        stats += proc_stats

    stats_df = pd.DataFrame(stats)
    print(stats_df)
    summary_df = stats_df.groupby(['model', 'category']).aggregate({'ssim':'mean', 'psnr':'mean'})
    summary_df.reset_index().to_csv('stats_summary.csv', index=False)
    stats_df.to_csv('stats.csv', index=False)
    print(summary_df)
