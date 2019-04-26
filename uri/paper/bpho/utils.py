"utility methods for generating movies from learners"
from fastai import *
from fastai.vision import *
import shutil
from skimage.io import imsave
import PIL
import imageio
from scipy.ndimage.interpolation import zoom as npzoom
from .czi import get_czi_shape_info, build_index, is_movie
import czifile
import PIL
import numpy as np
from fastprogress import progress_bar
from pathlib import Path
import torch
import math
from .multi import MultiImage
from time import sleep


__all__ = ['generate_movies', 'generate_tifs', 'ensure_folder', 'subfolders',
           'build_tile_info', 'generate_tiles', 'unet_image_from_tiles_blend']

def make_mask(shape, overlap, top=True, left=True, right=True, bottom=True):
    mask = np.full(shape, 1.)
    if overlap > 0:
        h,w = shape
        for i in range(min(shape[0], shape[0])):
            for j in range(shape[1]):
                if top: mask[i,j] = min((i+1)/overlap, mask[i,j])
                if bottom: mask[h-i-1,j] = min((i+1)/overlap, mask[h-i-1,j])
                if left: mask[i,j] = min((j+1)/overlap, mask[i,j])
                if right: mask[i,w-j-1] = min((j+1)/overlap, mask[i,w-j-1])
    return mask.astype(np.uint8)


def unet_multi_image_from_tiles(learn, in_img, tile_sz=128, scale=4, wsize=3):
    cur_size = in_img.shape[1:3]
    c = in_img.shape[0]
    new_size = (cur_size[0] * scale, cur_size[1] * scale)
    w, h = cur_size

    in_tile = torch.zeros((c, tile_sz // scale, tile_sz // scale))
    out_img = torch.zeros((1, w * scale, h * scale))
    tile_sz //= scale

    for x_tile in range(math.ceil(w / tile_sz)):
        for y_tile in range(math.ceil(h / tile_sz)):
            x_start = x_tile

            x_start = x_tile * tile_sz
            x_end = min(x_start + tile_sz, w)
            y_start = y_tile * tile_sz
            y_end = min(y_start + tile_sz, h)

            in_tile[:, 0:(x_end - x_start), 0:(y_end - y_start)] = tensor(
                in_img[:, x_start:x_end, y_start:y_end])

            img_list = [
                Image(tensor(npzoom(in_tile[i], scale, order=1))[None])
                for i in range(wsize)
            ]
            #img_list += img_list

            tlist = MultiImage(img_list)
            out_tile, _, _ = learn.predict(tlist)

            out_x_start = x_start * scale
            out_x_end = x_end * scale
            out_y_start = y_start * scale
            out_y_end = y_end * scale

            #print("out: ", out_x_start, out_y_start, ",", out_x_end, out_y_end)
            in_x_start = 0
            in_y_start = 0
            in_x_end = (x_end - x_start) * scale
            in_y_end = (y_end - y_start) * scale
            #print("tile: ",in_x_start, in_y_start, ",", in_x_end, in_y_end)

            out_img[:, out_x_start:out_x_end, out_y_start:
                    out_y_end] = out_tile.data[:, in_x_start:in_x_end,
                                               in_y_start:in_y_end]
    return out_img



def unet_image_from_tiles_blend(learn, in_img, tile_sz=256, scale=4, overlap_pct=0.0):
    in_img = npzoom(in_img[0], scale, order=1)
    h,w = in_img.shape
    overlap = int(max(h,w)*overlap_pct)
    step_sz = tile_sz - overlap
    assembled = PIL.Image.new('RGB',in_img.shape)
    for x_tile in range(0,math.ceil(w/step_sz)):
        for y_tile in range(0,math.ceil(h/step_sz)):
            x_start = x_tile*step_sz
            x_end = min(x_start + tile_sz, w)
            y_start = y_tile*step_sz
            y_end = min(y_start + tile_sz, h)
            src_tile = in_img[y_start:y_end,x_start:x_end]
            mask = make_mask(src_tile.shape, overlap,
                            top=y_start!=0,
                            left=x_start!=0,
                            bottom=y_end!=h,
                            right=x_end!=w)
            in_tile = torch.zeros((1, tile_sz, tile_sz))
            in_x_size = x_end - x_start
            in_y_size = y_end - y_start
            if (in_y_size, in_x_size) != src_tile.shape: set_trace()
            in_tile[0,0:in_y_size, 0:in_x_size] = tensor(src_tile)
            out_tile, _, _ = learn.predict(Image(in_tile))
            out_tile = (out_tile.data[0].numpy() * 255.).astype(np.uint8)
            combined = np.stack([out_tile[0:in_y_size, 0:in_x_size]]*3 +[mask]).transpose(1,2,0)
            t_img = PIL.Image.fromarray(combined, mode='RGBA')
            assembled.paste(t_img, box=(x_start,y_start))
    out_img = np.array(assembled.convert('L'))
    return out_img



def unet_image_from_tiles(learn, in_img, tile_sz=128, scale=4):
    cur_size = in_img.shape[1:3]
    c = in_img.shape[0]
    new_size = (cur_size[0] * scale, cur_size[1] * scale)
    w, h = cur_size

    in_tile = torch.zeros((c, tile_sz // scale, tile_sz // scale))
    out_img = torch.zeros((1, w * scale, h * scale))
    tile_sz //= scale

    for x_tile in range(math.ceil(w / tile_sz)):
        for y_tile in range(math.ceil(h / tile_sz)):
            x_start = x_tile

            x_start = x_tile * tile_sz
            x_end = min(x_start + tile_sz, w)
            y_start = y_tile * tile_sz
            y_end = min(y_start + tile_sz, h)

            in_tile[:, 0:(x_end - x_start), 0:(y_end - y_start)] = tensor(
                in_img[:, x_start:x_end, y_start:y_end])
            img = Image(tensor(npzoom(in_tile[0], scale, order=1)[None]))
            out_tile, _, _ = learn.predict(img)

            out_x_start = x_start * scale
            out_x_end = x_end * scale
            out_y_start = y_start * scale
            out_y_end = y_end * scale

            #print("out: ", out_x_start, out_y_start, ",", out_x_end, out_y_end)
            in_x_start = 0
            in_y_start = 0
            in_x_end = (x_end - x_start) * scale
            in_y_end = (y_end - y_start) * scale
            #print("tile: ",in_x_start, in_y_start, ",", in_x_end, in_y_end)

            out_img[:, out_x_start:out_x_end, out_y_start:
                    out_y_end] = out_tile.data[:, in_x_start:in_x_end,
                                               in_y_start:in_y_end]
    return out_img


def tif_predict_movie(learn,
                      tif_in,
                      orig_out='orig.tif',
                      pred_out='pred.tif',
                      size=128,
                      wsize=3):
    im = PIL.Image.open(tif_in)
    im.load()
    times = im.n_frames
    #times = min(times,100)
    imgs = []

    if times < (wsize + 2):
        print(f'skip {tif_in} only {times} frames')
        return

    for i in range(times):
        im.seek(i)
        imgs.append(np.array(im).astype(np.float32) / 255.)
    img_data = np.stack(imgs)

    def pull_frame(i):
        im.seek(i)
        im.load()
        return np.array(im)

    preds = []
    origs = []
    img_max = img_data.max()

    x, y = im.size
    #print(f'tif: x:{x} y:{y} t:{times}')
    for t in progress_bar(list(range(0, times - wsize + 1))):
        img = img_data[t:(t + wsize)].copy()
        img /= img_max

        out_img = unet_multi_image_from_tiles(learn,
                                              img,
                                              tile_sz=size,
                                              wsize=wsize)
        pred = (out_img * 255).cpu().numpy().astype(np.uint8)
        preds.append(pred)
        orig = (img[1][None] * 255).astype(np.uint8)
        origs.append(orig)
    if len(preds) > 0:
        all_y = np.concatenate(preds)
        #print(all_y.shape)
        imageio.mimwrite(
            pred_out, all_y,
            bigtiff=True)  #, fps=30, macro_block_size=None) # for mp4
        all_y = np.concatenate(origs)
        #print(all_y.shape)
        imageio.mimwrite(orig_out, all_y,
                         bigtiff=True)  #, fps=30, macro_block_size=None)


def czi_predict_movie(learn,
                      czi_in,
                      orig_out='orig.tif',
                      pred_out='pred.tif',
                      size=128,
                      wsize=3):
    with czifile.CziFile(czi_in) as czi_f:
        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        channels = proc_shape['C']
        depths = proc_shape['Z']
        times = proc_shape['T']
        #times = min(times, 100)
        x, y = proc_shape['X'], proc_shape['Y']
        #print(f'czi: x:{x} y:{y} t:{times} z:{depths}')
        if times < (wsize + 2):
            print(f'skip {czi_in} only {times} frames')
            return

        #folder_name = Path(pred_out).stem
        #folder = Path(folder_name)
        #if folder.exists(): shutil.rmtree(folder)
        #folder.mkdir()

        data = czi_f.asarray().astype(np.float32) / 255.
        preds = []
        origs = []

        img_max = data.max()
        #print(img_max)
        for t in progress_bar(list(range(0, times - wsize + 1))):
            idx = build_index(
                proc_axes, {
                    'T': slice(t, t + wsize),
                    'C': 0,
                    'Z': 0,
                    'X': slice(0, x),
                    'Y': slice(0, y)
                })
            img = data[idx].copy()
            img /= img_max

            out_img = unet_multi_image_from_tiles(learn,
                                                  img,
                                                  tile_sz=size,
                                                  wsize=wsize)
            pred = (out_img * 255).cpu().numpy().astype(np.uint8)
            preds.append(pred)
            #imsave(folder/f'{t}.tif', pred[0])

            orig = (img[wsize // 2][None] * 255).astype(np.uint8)
            origs.append(orig)
        if len(preds) > 0:
            all_y = np.concatenate(preds)
            #print(all_y.shape)
            imageio.mimwrite(
                pred_out, all_y,
                bigtiff=True)  #, fps=30, macro_block_size=None) # for mp4
            all_y = np.concatenate(origs)
            #print(all_y.shape)
            imageio.mimwrite(orig_out, all_y,
                             bigtiff=True)  #, fps=30, macro_block_size=None)


def generate_movies(dest_dir, movie_files, learn, size, wsize=5):
    for fn in progress_bar(movie_files):
        ensure_folder(dest_dir)
        pred_name = dest_dir/f'{fn.stem}_pred.tif'
        orig_name = dest_dir/f'{fn.stem}_orig.tif'
        if not Path(pred_name).exists():
            if fn.suffix == '.czi':
                #  print(f'czi {fn.stem}')
                czi_predict_movie(learn,
                                  fn,
                                  size=size,
                                  orig_out=orig_name,
                                  pred_out=pred_name,
                                  wsize=wsize)
            elif fn.suffix == '.tif':
                tif_predict_movie(learn,
                                  fn,
                                  size=size,
                                  orig_out=orig_name,
                                  pred_out=pred_name,
                                  wsize=wsize)
                tif_fn = fn
                #  print(f'tif {fn.stem}')
        else:
            print(f'skip: {fn.stem} - doesn\'t exist')



def tif_predict_images(learn,
                       tif_in,
                       dest,
                       category,
                       tag=None,
                       size=128,
                       max_imgs=None):
    under_tag = f'_' if tag is None else f'_{tag}_'
    dest_folder = Path(dest / category)
    dest_folder.mkdir(exist_ok=True, parents=True)
    pred_out = dest_folder / f'{tif_in.stem}{under_tag}pred.tif'
    orig_out = dest_folder / f'{tif_in.stem}{under_tag}orig.tif'
    if pred_out.exists():
        print(f'{pred_out.stem} exists')
        return

    im = PIL.Image.open(tif_in)
    im.load()
    times = im.n_frames
    if not max_imgs is None: times = min(max_imgs, times)

    imgs = []

    for i in range(times):
        im.seek(i)
        im.load()
        imgs.append(np.array(im).astype(np.float32) / 255.)
    img_data = np.stack(imgs)

    preds = []
    origs = []
    img_max = img_data.max()

    x, y = im.size
    #print(f'tif: x:{x} y:{y} t:{times}')
    for t in progress_bar(list(range(times))):
        img = img_data[t].copy()
        img /= img_max

        pred = unet_image_from_tiles_blend(learn, img[None], tile_sz=size)
        preds.append(pred[None])
        orig = (img[None] * 255).astype(np.uint8)
        origs.append(orig)

    if len(preds) > 0:
        all_y = np.concatenate(preds)
        imageio.mimwrite(pred_out, all_y, bigtiff=True)
        all_y = np.concatenate(origs)
        imageio.mimwrite(orig_out, all_y, bigtiff=True)


def czi_predict_images(learn,
                       czi_in,
                       dest,
                       category,
                       tag=None,
                       size=128,
                       max_imgs=None):
    with czifile.CziFile(czi_in) as czi_f:

        under_tag = f'_' if tag is None else f'_{tag}_'
        dest_folder = Path(dest / category)
        dest_folder.mkdir(exist_ok=True, parents=True)

        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        channels = proc_shape['C']
        depths = proc_shape['Z']
        times = proc_shape['T']
        if not max_imgs is None: times = min(max_imgs, times)

        x, y = proc_shape['X'], proc_shape['Y']

        data = czi_f.asarray().astype(np.float32) / 255.

        img_max = data.max()
        #print(f'czi: x:{x} y:{y} t:{times} c:{channels} z:{depths} {img_max}')

        channels_bar = progress_bar(
            range(channels)) if channels > 1 else range(channels)
        depths_bar = progress_bar(
            range(depths)) if depths > 1 else range(depths)
        times_bar = progress_bar(range(times)) if times > 1 else range(times)

        for c in channels_bar:
            for z in depths_bar:
                preds = []
                origs = []
                if (depths > 1) or (channels > 1):
                    pred_out = dest_folder / f'{czi_in.stem}_c{c:02d}_z{z:02d}_{under_tag}_pred.tif'
                    orig_out = dest_folder / f'{czi_in.stem}_c{c:02d}_z{z:02d}_{under_tag}_orig.tif'
                else:
                    pred_out = dest_folder / f'{czi_in.stem}_{under_tag}_pred.tif'
                    orig_out = dest_folder / f'{czi_in.stem}_{under_tag}_orig.tif'
                if not pred_out.exists():
                    for t in times_bar:
                        idx = build_index(
                            proc_axes, {
                                'T': t,
                                'C': c,
                                'Z': z,
                                'X': slice(0, x),
                                'Y': slice(0, y)
                            })
                        img = data[idx].copy()
                        img /= img_max

                        pred = unet_image_from_tiles_blend(learn,
                                                        img[None],
                                                        tile_sz=size)
                        preds.append(pred[None])
                        #imsave(folder/f'{t}.tif', pred[0])

                        orig = (img[None] * 255).astype(np.uint8)
                        origs.append(orig)

                    if len(preds) > 0:
                        all_y = np.concatenate(preds)
                        imageio.mimwrite(pred_out, all_y, bigtiff=True)
                        all_y = np.concatenate(origs)
                        imageio.mimwrite(orig_out, all_y, bigtiff=True)


def generate_tifs(src, dest, learn, size, tag=None, max_imgs=None):
    for fn in progress_bar(src):
        category = fn.parts[-3]
        try:
            if fn.suffix == '.czi':
                czi_predict_images(learn,
                                fn,
                                dest,
                                category,
                                size=size,
                                tag=tag,
                                max_imgs=max_imgs)
            elif fn.suffix == '.tif':
                tif_predict_images(learn,
                                fn,
                                dest,
                                category,
                                size=size,
                                tag=tag,
                                max_imgs=max_imgs)
        except Exception as e:
             print(f'exception with {fn.stem}')
             print(e)


def ensure_folder(fldr, clean=False):
    fldr = Path(fldr)
    if fldr.exists() and clean:
        print(f'wiping {fldr.stem} in 5 seconds')
        sleep(5.)
        shutil.rmtree(fldr)
    if not fldr.exists(): fldr.mkdir(parents=True, mode=0o775, exist_ok=True)
    return fldr


def subfolders(p):
    return [sub for sub in p.iterdir() if sub.is_dir()]


def build_tile_info(data, tile_sz, train_samples, valid_samples, only_categories=None, skip_categories=None):
    if skip_categories == None: skip_categories = []
    if only_categories == None: only_categories = []
    if only_categories: skip_categories = [c for c in skip_categories if c not in only_categories]

    def get_category(p):
        return p.parts[-2]

    def get_mode(p):
        return p.parts[-3]

    def is_only(fn):
        return (not only_categories) or (get_category(fn) in only_categories)

    def is_skip(fn):
        return get_category(fn) in skip_categories

    def get_img_size(p):
        with PIL.Image.open(p) as img:
            h,w = img.size
        return h,w

    all_files = [fn for fn in list(data.glob('**/*.tif')) if is_only(fn) and not is_skip(fn)]
    img_sizes = {str(p):get_img_size(p) for p in progress_bar(all_files)}

    files_by_mode = {}

    for p in progress_bar(all_files):
        category = get_category(p)
        mode = get_mode(p)
        mode_list = files_by_mode.get(mode, {})
        cat_list = mode_list.get(category, [])
        cat_list.append(p)
        mode_list[category] = cat_list
        files_by_mode[mode] = mode_list

    def pull_random_tile_info(mode, tile_sz):
        files_by_cat = files_by_mode[mode]
        category=random.choice(list(files_by_cat.keys()))
        img_file=random.choice(files_by_cat[category])
        h,w = img_sizes[str(img_file)]
        return {'mode': mode,'category': category,'fn': img_file, 'tile_sz': tile_sz, 'h': h, 'w':w}


    tile_infos = []
    for i in range(train_samples):
        tile_infos.append(pull_random_tile_info('train', tile_sz))
    for i in range(valid_samples):
        tile_infos.append(pull_random_tile_info('valid', tile_sz))

    tile_df = pd.DataFrame(tile_infos)[['mode','category','tile_sz','h','w','fn']]
    return tile_df


def draw_tile(img, tile_sz):
    max_x,max_y = img.shape
    x = random.choice(range(max_x-tile_sz)) if max_x > tile_sz else 0
    y = random.choice(range(max_y-tile_sz)) if max_y > tile_sz else 0
    xs = slice(x,min(x+tile_sz, max_x))
    ys = slice(y,min(y+tile_sz, max_y))
    tile = img[xs,ys].copy()
    return tile, (xs,ys)

def check_tile(img, thresh, thresh_pct):
    return (img > thresh).mean() > thresh_pct

def draw_random_tile(img_data, tile_sz, thresh, thresh_pct):
    max_tries = 200

    found_tile = False
    tries = 0
    while not found_tile:
        tile, (xs,ys) = draw_tile(img_data, tile_sz)
        found_tile = check_tile(tile, thresh, thresh_pct)
        #found_tile = True
        tries += 1
        if tries > (max_tries/2): thresh_pct /=2
        if tries > max_tries: found_tile = True
    box = [xs.start, ys.start, xs.stop, ys.stop]
    return PIL.Image.fromarray(tile), box

def generate_tiles(dest_dir, tile_info, crap_dir=None, crap_func=None):
    tile_data = []
    dest_dir = ensure_folder(dest_dir)
    shutil.rmtree(dest_dir)

    last_fn = None
    tile_info = tile_info.sort_values('fn')
    for row_id, tile_stats in progress_bar(list(tile_info.iterrows())):
        mode = tile_stats['mode']
        fn = tile_stats['fn']
        if fn != last_fn:
            img = PIL.Image.open(fn)
            img_data = np.array(img).astype(np.float32)
            img_data /= img_data.max()
            thresh = 0.01
            thresh_pct = (img_data > thresh).mean() * 0.8
            last_fn = fn
        tile_sz = tile_stats['tile_sz']
        category = tile_stats['category']

        crop_img, box = draw_random_tile(img_data, tile_sz, thresh, thresh_pct)
        tile_folder = ensure_folder(dest_dir/mode/category)
        crop_img.save(tile_folder/f'{row_id:05d}_{fn.stem}.tif')
        if crap_func and crap_dir:
            crap_tile_folder = ensure_folder(crap_dir/mode/category)
            crap_img = crap_func(crop_img)
            crap_img.save(crap_tile_folder/f'{row_id:05d}_{fn.stem}.tif')
        tile_data.append({'tile_id': row_id, 'category': category, 'mode': mode, 'tile_sz': tile_sz, 'box': box, 'fn': fn})
    pd.DataFrame(tile_data).to_csv(dest_dir/'tiles.csv', index=False)
