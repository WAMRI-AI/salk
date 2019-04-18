"utility methods for generating movies from learners"
from fastai import *
from fastai.vision import *
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
import pyvips

__all__ = ['generate_movies', 'generate_tifs']


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


# map vips formats to np dtypes
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}


# numpy array to vips image
def numpy2vips(a):
    height, width, bands = a.shape
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(a.dtype)])
    return vi

def tile2vips(t):
    a = t.cpu().detach().numpy()
    c,h,w = a.shape
    return numpy2vips(a.reshape((h,w,c)))

# vips image to numpy array
def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])

def vip2tensor(vi):
    a = vips2numpy(vi)
    return tensor(a.transpose(2, 0, 1))

def unet_image_from_tiles_blend(learn, in_img, tile_sz=128, scale=4, overlap=80):
    #  set_trace()
    cur_size = in_img.shape[1:3]
    c = in_img.shape[0]
    new_size = (cur_size[0] * scale, cur_size[1] * scale)
    h, w = cur_size

    in_tile = torch.zeros((c, tile_sz // scale, tile_sz // scale))
    #out_img = torch.zeros((1, w * scale, h * scale))
    tile_sz //= scale
    overlap //= scale
    over_sz = tile_sz - overlap
    tiles = []

    out_img = None
    tiles_across = math.ceil(w/over_sz)
    tiles_down = math.ceil(h/over_sz)

    for y_tile in range(tiles_down):
        row_img = None
        y_start = y_tile * over_sz
        y_end = min(y_start + tile_sz, h)

        out_y_start = y_start * scale
        out_y_end = y_end * scale 

        in_y_start = 0
        in_y_end = (y_end - y_start) * scale

        for x_tile in range(tiles_across):
            x_start = x_tile * over_sz
            x_end = min(x_start + tile_sz, w)

            out_x_start = x_start * scale
            out_x_end = x_end * scale 

            in_x_start = 0
            in_x_end = (x_end - x_start) * scale

            in_tile[:, 0:(y_end - y_start), 0:(x_end - x_start)] = tensor(in_img[:, y_start:y_end, x_start:x_end])
            img = Image(tensor(npzoom(in_tile[0], scale, order=1)[None]))
            out_tile, _, _ = learn.predict(img)
            tile_data = out_tile.data[:, in_y_start:in_y_end, in_x_start:in_x_end]
            tile = tile2vips(tile_data)
            #  print(tile.height, tile.width, in_y_end, in_x_end)
            tiles.append(tile)

    spacing = over_sz * scale # - 1
    print(len(tiles), tiles_across, tiles_down, spacing)

    #  base = pyvips.Image.black(new_size[1], new_size[0], bands=1)
    for y in range(tiles_down):
        row_img = None
        for x in range(tiles_across):
            idx = y*tiles_across + x
            #  print('idx:', idx)
            a_tile = tiles[idx]
            if row_img is None: row_img = a_tile
            else: 
                row_img = a_tile.merge(row_img, 'horizontal', x*spacing, 0) #:w, mblend=spacing//2).copy_memory()
                #  print(x,y, x*spacing, x*spacing + spacing//2)
                #  row_img = row_img.mosaic(a_tile, 'horizontal', x*spacing,0, x*spacing + spacing//2, 0, harea=overlap//2, hwindow=3)
                #  print('coo')
            #  print('row: ', row_img)
        if out_img is None: out_img = row_img
        else:
            out_img = row_img.merge(out_img, 'vertical', 0, y*spacing, mblend=spacing//2)
        #  print('out: ', out_img)
            #  print(row_img, out_img, x, y)
            #  out_img = out_img.mosaic(row_img, 'vertical', 0, y*spacing, 0, y*spacing + spacing//2, harea=overlap//2, hwindow=3)
    #  out_img = pyvips.Image.arrayjoin(tiles, across=tiles_across, halign='low', valign='low', hspacing=spacing, vspacing=spacing, shim=-3) #, shim=10, halign='low', valign='low', hspacing=spacing, vspacing=spacing)
     
    return vip2tensor(out_img)


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
    print(f'tif: x:{x} y:{y} t:{times}')
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
        print(f'czi: x:{x} y:{y} t:{times} z:{depths}')
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
        print(img_max)
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


def generate_movies(movie_files, learn, size, wsize=5):
    for fn in progress_bar(movie_files):
        pred_name = f'{fn.stem}_pred.tif'
        orig_name = f'{fn.stem}_orig.tif'
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
    print(f'tif: x:{x} y:{y} t:{times}')
    for t in progress_bar(list(range(times))):
        img = img_data[t].copy()
        img /= img_max

        out_img = unet_image_from_tiles_blend(learn, img[None], tile_sz=size)
        pred = (out_img * 255).cpu().numpy().astype(np.uint8)
        preds.append(pred)
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
        print(f'czi: x:{x} y:{y} t:{times} c:{channels} z:{depths} {img_max}')

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

                        out_img = unet_image_from_tiles_blend(learn,
                                                        img[None],
                                                        tile_sz=size)
                        pred = (out_img * 255).cpu().numpy().astype(np.uint8)
                        preds.append(pred)
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
        #  try:
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
        #  except:
        #      print(f'exception with {fn.stem}')
