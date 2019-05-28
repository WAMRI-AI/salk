import sys
import yaml
import pandas as pd
from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.vision.models.xresnet import *
from fastai.vision.models.unet import DynamicUnet
from fastprogress import master_bar, progress_bar
import imageio
from bpho import *
import PIL.Image
import czifile

torch.backends.cudnn.benchmark = True

def check_dir(p):
    if not p.exists():
        print(f"couldn't find {p}")
        sys.exit(1)
    return p

def process_tif(fn, processor, proc_func, out_fn, n_depth=1, n_time=1):
    print('process tif not done yet', fn.stem)
    return
    stats = []
    truth_imgs = PIL.Image.open(truth) if truth and truth.exists() else None
    with PIL.Image.open(item) as img_tif:
        n_frame = max(n_depth, n_time)
        offset_frame = n_frame // 2
        if n_frame > img_tif.n_frames: return []
        mid_frame = img_tif.n_frames // 2
        if n_frame > 1:
            img_tifs = []
            for i in range(mid_frame-offset_frame, mid_frame+offset_frame+1):
                img_tif.seek(i)
                img_tif.load()
                img_tifs.append(np.array(img_tif).copy())
            imgs = np.stack(img_tifs)
            img, img_info = img_to_float(imgs)
        else:
            img_tif.seek(mid_frame)
            img_tif.load()
            img, img_info = img_to_float(np.array(img_tif))

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
            pred_img = proc_func(img, img_info=img_info)
            pred_img8 = img_to_uint8(pred_img, img_info=img_info)
            PIL.Image.fromarray(pred_img8).save(out_name)

        if not truth_img is None and not pred_img is None:
            truth_folder = ensure_folder(out_fldr/'../truth')
            truth_name = f'truth_{item.stem}_{tag}'
            truth_img8 = img_to_uint8(truth_img, img_info=truth_info)
            PIL.Image.fromarray(truth_img8).save((truth_folder/truth_name).with_suffix('.tif'))
            istats = calc_stats(pred_img8, truth_img8)
            if istats:
                istats.update({'tag': tag, 'item': item.stem})
                stats.append(istats)
    return stats

def process_czi(fn, processor, proc_func, out_fn, n_depth=1, n_time=1):
    stats = []
    with czifile.CziFile(fn) as czi_f:
        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        channels = proc_shape['C']
        depths = proc_shape['Z']
        times = proc_shape['T']
        x, y = proc_shape['X'], proc_shape['Y']

        data = czi_f.asarray().astype(np.float32)

        if depths < n_depth: return
        if times < n_time: return

        if n_depth > 1:
            set_trace()
            offset_frames = n_depth // 2
            for c in range(channels):
                for t in range(times):
                    for z in range(offset_frames, depths - offset_frame):
                        depth_slice = slice(z-offset_frames, z+offset_frame+1)
                        idx = build_index(
                            proc_axes, {
                                'T': t,
                                'C': c,
                                'Z': depth_slice,
                                'X': slice(0, x),
                                'Y': slice(0, y)
                        })
                        img = data[idx].copy()
                        img, img_info = img_to_float(img)
                        tag = f'{c}_{t}_{z+offset_frames}_'

                        save_name = f'{proc_name}_{item.stem}_{tag}'

                        pred_img = proc_func(img, img_info=img_info)
                        pred_img8 = img_to_uint8(pred_img, img_info=img_info)
                        PIL.Image.fromarray(pred_img8).save(out_fn)
        elif n_time > 1:
            # times = 30
            offset_frames = n_time // 2
            for c in range(channels):
                for z in range(depths):
                    imgs = []
                    time_range = list(range(offset_frames, times - offset_frames))
                    for t in progress_bar(time_range):
                        time_slice = slice(t-offset_frames, t+offset_frames+1)
                        idx = build_index(
                            proc_axes, {
                                'T': time_slice,
                                'C': c,
                                'Z': z,
                                'X': slice(0, x),
                                'Y': slice(0, y)
                        })
                        img = data[idx].copy()
                        img, img_info = img_to_float(img)

                        pred_img = proc_func(img, img_info=img_info)
                        pred_img8 = img_to_uint8(pred_img, img_info=img_info)
                        imgs.append(pred_img8[None])

                    all_y = np.concatenate(imgs)
                    save_name = f'{processor}.tif'
                    fldr_name = out_fn.parent/fn.stem
                    if c > 1 or z > 1:
                        fldr_name = fldr_name/f'{c}_{z}'
                    out_fldr = ensure_folder(fldr_name)
                    imageio.mimwrite(out_fldr/save_name, all_y) #, fps=30, macro_block_size=None) # for mp4
                    imageio.mimwrite((out_fldr/save_name).with_suffix('.mp4'), all_y, fps=30, macro_block_size=None) # for mp4
        else:
            for c in range(channels):
                for z in range(depths):
                    for t in range(times):
                        idx = build_index(
                            proc_axes, {
                                'T': t,
                                'C': c,
                                'Z': z,
                                'X': slice(0, x),
                                'Y': slice(0, y)
                        })
                        img = data[idx].copy()
                        img, img_info = img_to_float(img)

                        tag = f'{c}_{t}_{z}'
                        out_fldr = ensure_folder(out_fn.parent/out_fn.stem)
                        save_name = f'{processor}_{tag}.tif'
                        pred_img = proc_func(img, img_info=img_info)
                        pred_img8 = img_to_uint8(pred_img, img_info=img_info)
                        PIL.Image.fromarray(pred_img8).save(out_fldr/save_name)

def process_files(src_dir, out_dir, model_dir, processor, mbar=None):
    proc_map = {
        '.tif': process_tif,
        '.czi': process_czi
    }
    proc_func, num_chan = get_named_processor(processor, model_dir)
    src_files = list(src_dir.glob('**/*.czi'))
    src_files += list(src_dir.glob('**/*.tif'))

    for fn in progress_bar(src_files, parent=mbar):
        out_fn = out_dir/fn.relative_to(src_dir)
        ensure_folder(out_fn.parent)
        file_proc = proc_map.get(fn.suffix, None)
        if file_proc:
            n_depth = n_time = 1
            if 'multiz' in processor: n_depth = num_chan
            if 'multit' in processor: n_time = num_chan
            file_proc(fn, processor, proc_func, out_fn, n_depth=n_depth, n_time=n_time)

@call_parse
def main(
        src_dir: Param("source dir", Path, opt=False),
        out_dir: Param("ouput dir", Path, opt=False),
        model_dir: Param("model dir", Path) = 'stats/models',
        gpu: Param("GPU to run on", int, required=True) = None,
        models: Param("list models to run", str, nargs='+')=None,
        baselines: Param("build bilinear and bicubic", action='store_true')=False,
):
    print('on gpu: ', gpu)
    torch.cuda.set_device(gpu)
    out_dir = ensure_folder(out_dir)
    src_dir = check_dir(src_dir)
    model_dir = check_dir(model_dir)


    processors = []
    stats = []
    if baselines: processors += ['bilinear', 'bicubic', 'original']
    if models: processors += [m for m in models]
    mbar = master_bar(processors)
    for proc in mbar:
        mbar.write(f'processing {proc}')
        process_files(src_dir, out_dir, model_dir, proc, mbar=mbar)
