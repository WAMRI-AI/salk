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

__all__ = ['generate_movies', 'tif_predict_movie', 'czi_predict_movie', 'unet_image_from_tiles']


def unet_image_from_tiles(learn, in_img, tile_sz=128, scale=4, wsize=3):
    cur_size = in_img.shape[1:3]
    c = in_img.shape[0]
    new_size = (cur_size[0]*scale, cur_size[1]*scale)
    w, h = cur_size

    in_tile = torch.zeros((c,tile_sz//scale,tile_sz//scale))
    out_img = torch.zeros((1,w*scale,h*scale))
    tile_sz //= scale

    for x_tile in range(math.ceil(w/tile_sz)):
        for y_tile in range(math.ceil(h/tile_sz)):
            x_start = x_tile

            x_start = x_tile*tile_sz
            x_end = min(x_start+tile_sz, w)
            y_start = y_tile*tile_sz
            y_end = min(y_start+tile_sz, h)


            in_tile[:,0:(x_end-x_start), 0:(y_end-y_start)] = tensor(in_img[:,x_start:x_end, y_start:y_end])

            img_list = [Image(tensor(npzoom(in_tile[i], scale, order=1))[None]) for i in range(wsize)]
            #img_list += img_list

            tlist = MultiImage(img_list)
            out_tile,_,_ = learn.predict(tlist)

            out_x_start = x_start * scale
            out_x_end = x_end * scale
            out_y_start = y_start * scale
            out_y_end = y_end * scale

            #print("out: ", out_x_start, out_y_start, ",", out_x_end, out_y_end)
            in_x_start = 0
            in_y_start = 0
            in_x_end = (x_end-x_start) * scale
            in_y_end = (y_end-y_start) * scale
            #print("tile: ",in_x_start, in_y_start, ",", in_x_end, in_y_end)

            out_img[:,out_x_start:out_x_end, out_y_start:out_y_end] = out_tile.data[:,
                                                                                  in_x_start:in_x_end,
                                                                                  in_y_start:in_y_end]
    return out_img

def tif_predict_movie(learn, tif_in, orig_out='orig.tif', pred_out='pred.tif', size=128, wsize=3):
    im = PIL.Image.open(tif_in)
    im.load()
    times = im.n_frames
    #times = min(times,100)
    imgs = []

    if times < (wsize+2):
        print(f'skip {tif_in} only {times} frames')
        return

    for i in range(times):
        im.seek(i)
        imgs.append(np.array(im).astype(np.float32)/255.)
    img_data = np.stack(imgs)

    def pull_frame(i):
        im.seek(i)
        im.load()
        return np.array(im)

    preds = []
    origs = []
    img_max = img_data.max()

    x,y = im.size
    print(f'tif: x:{x} y:{y} t:{times}')
    for t in progress_bar(list(range(0,times-wsize+1))):
        img = img_data[t:(t+wsize)].copy()
        img /= img_max

        out_img = unet_image_from_tiles(learn, img, tile_sz=size, wsize=wsize)
        pred = (out_img*255).cpu().numpy().astype(np.uint8)
        preds.append(pred)
        orig = (img[1][None]*255).astype(np.uint8)
        origs.append(orig)
    if len(preds) > 0:
        all_y = np.concatenate(preds)
        #print(all_y.shape)
        imageio.mimwrite(pred_out, all_y, bigtiff=True) #, fps=30, macro_block_size=None) # for mp4
        all_y = np.concatenate(origs)
        #print(all_y.shape)
        imageio.mimwrite(orig_out, all_y, bigtiff=True) #, fps=30, macro_block_size=None)


def czi_predict_movie(learn, czi_in, orig_out='orig.tif', pred_out='pred.tif', size=128, wsize=3):
    with czifile.CziFile(czi_in) as czi_f:
        proc_axes, proc_shape = get_czi_shape_info(czi_f)
        channels = proc_shape['C']
        depths = proc_shape['Z']
        times = proc_shape['T']
        #times = min(times, 100)
        x,y = proc_shape['X'], proc_shape['Y']
        print(f'czi: x:{x} y:{y} t:{times} z:{depths}')
        if times < (wsize+2):
            print(f'skip {czi_in} only {times} frames')
            return

        #folder_name = Path(pred_out).stem
        #folder = Path(folder_name)
        #if folder.exists(): shutil.rmtree(folder)
        #folder.mkdir()

        data = czi_f.asarray().astype(np.float32)/255.
        preds = []
        origs = []

        img_max = data.max()
        print(img_max)
        for t in progress_bar(list(range(0,times-wsize+1))):
            idx = build_index(proc_axes, {'T': slice(t,t+wsize), 'C': 0, 'Z':0, 'X':slice(0,x),'Y':slice(0,y)})
            img = data[idx].copy()
            img /= img_max

            out_img = unet_image_from_tiles(learn, img, tile_sz=size, wsize=wsize)
            pred = (out_img*255).cpu().numpy().astype(np.uint8)
            preds.append(pred)
            #imsave(folder/f'{t}.tif', pred[0])

            orig = (img[wsize//2][None]*255).astype(np.uint8)
            origs.append(orig)
        if len(preds) > 0:
            all_y = np.concatenate(preds)
            #print(all_y.shape)
            imageio.mimwrite(pred_out, all_y, bigtiff=True) #, fps=30, macro_block_size=None) # for mp4
            all_y = np.concatenate(origs)
            #print(all_y.shape)
            imageio.mimwrite(orig_out, all_y, bigtiff=True) #, fps=30, macro_block_size=None)

def generate_movies(movie_files, learn, size, wsize=5):
    for fn in progress_bar(movie_files):
        pred_name = f'{fn.stem}_pred.tif'
        orig_name = f'{fn.stem}_orig.tif'
        if not Path(pred_name).exists():
            if fn.suffix == '.czi':
                #  print(f'czi {fn.stem}')
                czi_predict_movie(learn, fn, size=size, orig_out=orig_name, pred_out=pred_name, wsize=wsize)
            elif fn.suffix == '.tif':
                tif_predict_movie(learn, fn, size=size, orig_out=orig_name, pred_out=pred_name, wsize=wsize)
                tif_fn = fn
                #  print(f'tif {fn.stem}')
        else:
            print(f'skip: {fn.stem} - doesn\'t exist')
