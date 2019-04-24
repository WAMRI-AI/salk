
import numpy as np
from fastai import *
from scipy.ndimage.interpolation import zoom as npzoom

__all__ = ['get_named_processor', 'add_model_processor']

def bilinear(img):
    return npzoom(img, 4, order=1)

def bicubic(img):
    return npzoom(img, 4, order=2)

processors = {
    'bilinear': bilinear,
    'bicubic': bicubic
}

def build_processor(name):
    print('build learner based processor here')
    builder = learn_builders.get(name, None)

def get_named_processor(name):
    if not name in processors:
        build_processor(name)
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
