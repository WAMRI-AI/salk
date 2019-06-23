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

def process_tif(fn, processor, proc_func, out_fn, n_depth=1, n_time=1, mode='L'):
    with PIL.Image.open(fn) as img_tif:
        n_frame = max(n_depth, n_time)
        offset_frame = n_frame // 2
        if n_frame > img_tif.n_frames: return []

        if n_frame > 1:
            img_tifs = []
            for i in range(offset_frame, img_tif.n_frames-offset_frame):
                img_tif.seek(i)
                img_tif.load()
                img_tifs.append(np.array(img_tif).copy())
            imgs = np.stack(img_tifs)
            img, img_info = img_to_float(imgs)
            out_fldr = ensure_folder(out_fn.parent/out_fn.stem)
            save_name = f'{processor}.tif'
            pred_img = proc_func(img, img_info=img_info, mode=mode)
            pred_img8 = (pred_img * np.iinfo(np.uint8).max).astype(np.uint8)
            imageio.mimwrite(out_fldr/save_name, imgs)
            imageio.mimwrite((out_fldr/save_name).with_suffix('.mp4'), imgs, fps=30, macro_block_size=None) # for mp4
        else:
            for i in range(offset_frame, img_tif.n_frames-offset_frame):
                img_tif.seek(i)
                img_tif.load()
                img = np.array(img_tif).copy()
                img, img_info = img_to_float(img)

                tag = f'0_{i}_0'
                out_fldr = ensure_folder(out_fn.parent/out_fn.stem)
                save_name = f'{processor}_{tag}.tif'
                pred_img = proc_func(img, img_info=img_info, mode=mode)
                pred_img8 = (pred_img * np.iinfo(np.uint8).max).astype(np.uint8)
                PIL.Image.fromarray(pred_img8).save(out_fldr/save_name)

def process_czi(fn, processor, proc_func, out_fn, n_depth=1, n_time=1, mode='L'):
    stats = []
    with czifile.CziFile(fn) as czi_f:
        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        channels = proc_shape['C']
        depths = proc_shape['Z']
        times = proc_shape['T']
        x, y = proc_shape['X'], proc_shape['Y']

        data = czi_f.asarray().astype(np.float32)
        data, img_info = img_to_float(data)
        img_max = img_info['real_max']

        if depths < n_depth: return
        if times < n_time: return

        if n_depth > 1:
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
                        tag = f'{c}_{t}_{z+offset_frames}_'

                        save_name = f'{proc_name}_{item.stem}_{tag}'

                        pred_img = proc_func(img, img_info=img_info, mode=mode)
                        pred_img8 = (pred_img * np.iinfo(np.uint8).max).astype(np.uint8)
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
                        pred_img = proc_func(img, img_info=img_info, mode=mode)
                        pred_img8 = (pred_img * np.iinfo(np.uint8).max).astype(np.uint8)
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
                        tag = f'{c}_{t}_{z}'
                        out_fldr = ensure_folder(out_fn.parent/out_fn.stem)
                        save_name = f'{processor}_{tag}.tif'
                        pred_img = proc_func(img, img_info=img_info, mode=mode)
                        pred_img8 = (pred_img * np.iinfo(np.uint8).max).astype(np.uint8)
                        PIL.Image.fromarray(pred_img8).save(out_fldr/save_name)

def process_files(src_dir, out_dir, model_dir, processor, mode, mbar=None):
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
            file_proc(fn, processor, proc_func, out_fn, n_depth=n_depth, n_time=n_time, mode=mode)

@call_parse
def main(
        src_dir: Param("source dir", Path, opt=False),
        out_dir: Param("ouput dir", Path, opt=False),
        model_dir: Param("model dir", Path) = 'stats/models',
        gpu: Param("GPU to run on", int, required=True) = None,
        models: Param("list models to run", str, nargs='+')=None,
        baselines: Param("build bilinear and bicubic", action='store_true')=False,
        mode: Param("L or RGBA", str)='L',
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
        process_files(src_dir, out_dir, model_dir, proc, mode, mbar=mbar)
