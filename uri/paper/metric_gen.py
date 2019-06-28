from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
import czifile
from csbdeep.utils import normalize
from tqdm import tqdm
from scipy.ndimage import zoom 
import argparse
import csv
import glob
import PIL
from PIL import Image
import torchvision
from fastprogress import *
from skimage.measure import compare_ssim, compare_psnr
from pdb import set_trace

def norm_minmse(gt, x, normalize_gt=True):
    """
    normalizes and affinely scales an image pair such that the MSE is minimized  
     
    Parameters
    ----------
    gt: ndarray
        the ground truth image      
    x: ndarray
        the image that will be affinely scaled 
    normalize_gt: bool
        set to True of gt image should be normalized (default)

    Returns
    -------
    gt_scaled, x_scaled 

    """
    if normalize_gt:
        gt = normalize(gt, 0.1, 99.9, clip=False).astype(np.float32, copy = False)
    x = x.astype(np.float32, copy=False) - np.mean(x)
    gt = gt.astype(np.float32, copy=False) - np.mean(gt)
    scale = np.cov(x.flatten(), gt.flatten())[0, 1] / np.var(x.flatten())
    return gt, scale * x

def save_stats(stats, save_dir):
    with open(save_dir, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(stats)
    csvFile.close() 

def slice_process(x1, x2, y):
    if len(x1.shape) == 3: x1 = x1[0,:,:]
    if len(x2.shape) == 3: x2 = x2[0,:,:]
    if len(y.shape) == 3: y = y[0,:,:]

    # a scaled and shifted version of pred and bilinear
    x1 = 2*x1 + 100
    x2 = 2*x2 + 100

    # normalize/scale images
    (y_norm1, x1_norm) = norm_minmse(y, x1)	
    (y_norm2, x2_norm) = norm_minmse(y, x2)
    
    # calulate psnr and ssim of the normalized/scaled images
    psnr1 = compare_psnr(*(y_norm1, x1_norm), data_range = 1.)
    psnr2 = compare_psnr(*(y_norm2, x2_norm), data_range = 1.)
    ssim1 = compare_ssim(*(y_norm1, x1_norm), data_range = 1.)
    ssim2 = compare_ssim(*(y_norm2, x2_norm), data_range = 1.)
    return psnr1, ssim1, psnr2, ssim2, y_norm1, x1_norm, y_norm2, x2_norm   
  
def stack_process(pred, bilinear, gt, offset_frames=0):
    stack_pred = PIL.Image.open(pred)
    stack_bilinear = PIL.Image.open(bilinear)
    stack_gt  = PIL.Image.open(gt)

    frames = stack_pred.n_frames
    stack_psnr = []
    stack_lpsnr = []
    stack_ssim = []
    stack_lssim = []

    y_norm1s = []
    x1_norms = []
    y_norm2s = []
    x2_norms = []

    for i in range(frames):
        stack_pred.seek(i)
        stack_bilinear.seek(i+offset_frames)
        stack_gt.seek(i+offset_frames)

        x1 = np.array(stack_pred).astype(np.float32)
        x2 = np.array(stack_bilinear).astype(np.float32)
        y = np.array(stack_gt).astype(np.float32)
        psnr, ssim, l_psnr, l_ssim, y_norm1, x1_norm, y_norm2, x2_norm = slice_process(x1, x2, y)

        stack_psnr.append(psnr)
        stack_lpsnr.append(l_psnr)       
        stack_ssim.append(ssim)
        stack_lssim.append(l_ssim)  

        y_norm1s.append(np.array(y_norm1).copy())
        x1_norms.append(np.array(x1_norm).copy())
        y_norm2s.append(np.array(y_norm2).copy())
        x2_norms.append(np.array(x2_norm).copy())
    stem = Path(gt).stem
    #tifffile.imsave(str(exp_dir/f"{stem}_GTnormtopred.tif"), np.stack(y_norm1s).astype(np.float32))
    #tifffile.imsave(str(exp_dir/f"{stem}_prednorm.tif"), np.stack(x1_norms).astype(np.float32))
    #tifffile.imsave(str(exp_dir/f"{stem}_GTnormtobilinear.tif"), np.stack(y_norm2s).astype(np.float32))
    #tifffile.imsave(str(exp_dir/f"{stem}_bilinearnorm.tif"), np.stack(x2_norms).astype(np.float32))
    return stack_psnr,stack_ssim,stack_lpsnr,stack_lssim

def metric_gen(predset, testset, stats_dir, offset_frames):
    save_dir = Path('stats/csv')/f'stats_{testset}_{predset}.csv'

    pred_dir = stats_dir/f'output/{testset}/{predset}'
    bilinear_dir = stats_dir/f'bilinear/{testset}'
    gt_dir = stats_dir/f'ground_truth/{testset}'

    pred_list = list(pred_dir.glob(f'*.tif'))
    bilinear_list = list(bilinear_dir.glob(f'*.tif'))
    gt_list = list(gt_dir.glob(f'*.tif'))

    pred_list.sort()
    bilinear_list.sort()
    gt_list.sort()

    ssims = []
    l_ssims = []
    psnrs = []
    l_psnrs = []

    for p, l, t in progress_bar(list(zip(pred_list, bilinear_list, gt_list))):
        stack_psnr,stack_ssim,stack_lpsnr,stack_lssim = stack_process(p,l,t,offset_frames)

        psnrs = np.concatenate((psnrs, stack_psnr), out=None)
        l_psnrs = np.concatenate((l_psnrs, stack_lpsnr), out=None)    
        ssims = np.concatenate((ssims, stack_ssim), out=None)
        l_ssims = np.concatenate((l_ssims, stack_lssim), out=None)
    stats = zip(ssims, l_ssims, psnrs, l_psnrs)
    save_stats(stats, save_dir)

if __name__ == '__main__':
    np.random.seed(32)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--experiment', type = str, default = "semisynth_mito")
    parser.add_argument('-p', '--predset', type = str, default = "single_100mito_best_512")
    args = parser.parse_args()

    predset = args.predset
    testset = args.experiment
    stats_dir = Path('stats/')
    offset_frames = 2 if 'multi' in predset else 0
    metric_gen(predset, testset, stats_dir, offset_frames)

