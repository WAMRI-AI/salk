
import numpy as np
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from scipy.ndimage.interpolation import zoom as npzoom
from .utils import unet_image_from_tiles_blend

__all__ = ['get_named_processor', 'add_model_processor']

def bilinear(img):
    pred_img = npzoom(img, 4, order=1)
    return pred_img

def bicubic(img):
    pred_img=npzoom(img, 4, order=2)
    return pred_img

processors = {
    'bilinear': bilinear,
    'bicubic': bicubic
}

def build_processor(name, model_dir):
    learn = load_learner(model_dir, f'{name}.pkl').to_fp16()
    tile_sz = int(name.split('_')[-1])

    def learn_processor(img):
        img = (img * np.iinfo(np.uint8).max).astype(np.uint8)
        pred_img = unet_image_from_tiles_blend(learn, img[None], tile_sz=tile_sz)
        return pred_img

    return learn_processor


def get_named_processor(name, model_dir):
    if not name in processors:
        proc = build_processor(name, model_dir)
        if proc:
            processors[name] = proc
    proc = processors.get(name, None)
    return proc

def make_learner(model_name, model_dir, path):
    print('make_learner here')
    return None

def add_model_processor(model_name, model_dir, path='.'):
    if model_name not in processors:
        def learner_proc(lrn, img):
            return lrn.predict(img)
        learner = make_learner(model_name, model_dir, path)
        processors[model_name] = partial(learner_proc, learner)
