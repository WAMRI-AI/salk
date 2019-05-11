#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'paper/'))
	print(os.getcwd())
except:
	pass

#%%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.vision.models.xresnet import *
from fastai.vision.models.unet import DynamicUnet
from skimage.util import random_noise
from skimage import filters
from skimage.transform import rescale
from scipy.ndimage.interpolation import zoom as npzoom
from bpho import *
#%%
torch.cuda.set_device(1)

#%%
datasetname = 'mito_tiles'
data_path = Path('.')
datasets = data_path/'datasets'
datasources = data_path/'data'
dataset = datasets/datasetname

test_files = dataset/'test'
hr_tifs = dataset/'hr_t_1024'
lr_tifs = dataset/'lr_t_1024'

stats_inputs = Path('stats/input')
mname = 'combo'
model_dir = 'models'

loss = F.mse_loss
metrics = sr_metrics


#%%
ensure_folder(lr_tifs)
def downsample(fn, i, scale=4):
    dest = lr_tifs/fn.relative_to(hr_tifs)
    img = PIL.Image.open(fn)
    down_img = npzoom(np.array(img), 1./scale, order=1)
    ensure_folder(dest.parent)
    img = PIL.Image.fromarray(down_img).save(dest)

il = ImageList.from_folder(hr_tifs)

#%%
parallel(downsample, il.items)

#%%
def get_src(x_data, y_data):
    def map_to_hr(x):
        return y_data/x.relative_to(x_data)

    src = (ImageImageList
            .from_folder(x_data, convert_mode='L')
            .split_by_folder()
            .label_from_func(map_to_hr, convert_mode='L'))
    return src


def get_data(bs, size, x_data, y_data, 
             max_rotate=10.,
             min_zoom=1., max_zoom=1.1, 
             scale=4,
             use_cutout=False, 
             use_noise=False, 
             xtra_tfms=None, 
             **kwargs):
    
    src = get_src(x_data, y_data)
    x_tfms, y_tfms = get_xy_transforms(
                          max_rotate=max_rotate, 
                          min_zoom=min_zoom, max_zoom=max_zoom, 
                          use_cutout=use_cutout, 
                          use_noise=use_noise, 
                          xtra_tfms = xtra_tfms)
    x_size = size // scale
    data = (src
            .transform(x_tfms, size=x_size)
            .transform_y(y_tfms, size=size)
            .databunch(bs=bs,**kwargs))#.normalize())
    data.c = 3
    return data

def do_fit(learn, save_name, lrs=slice(1e-3), pct_start=0.9, cycle_len=10):
    learn.to_fp16().fit_one_cycle(cycle_len, lrs, pct_start=pct_start)
    learn.save(save_name)
    print(f'saved: {save_name}')
    num_rows = min(learn.data.batch_size, 3)
    learn.to_fp32().show_results(rows=num_rows, imgsize=5)


#%%
def data_for_size(bs, size):
    return get_data(bs, size, lr_tifs,  hr_tifs, 
                max_rotate=30.,
                min_zoom=1., max_zoom=4.,
                use_cutout=False, use_noise=True)

#%%
step = 0
lr = 1e-3
cycles = 2
loss = F.l1_loss
metrics = sr_metrics


bs = 16
size = 256
arch = xresnet18
data = data_for_size(bs, size)
learn = xres_unet_learner(data, arch, loss_func=loss, metrics=metrics, model_dir=model_dir, path='.')
gc.collect()


#%%
if True:
    learn.lr_find()
    learn.recorder.plot()


#%%
#data.train_ds.y.items[0]


#%%
#data.show_batch(3)


#%%
do_fit(learn, f'{mname}.{step:02d}', lrs=lr, cycle_len=cycles)


#%%
step = 1
lr = 3e-4
cycles = 10
loss = F.l1_loss
metrics = sr_metrics


bs = 16
size = 256
arch = xresnet18
data = data_for_size(bs, size)
learn = xres_unet_learner(data, arch, loss_func=loss, metrics=metrics, model_dir=model_dir, path='.')
learn.load(f'{mname}.{(step-1):02d}')

gc.collect()

#%%
do_fit(learn, f'{mname}.{step:02d}', lrs=lr, cycle_len=cycles)


#%%
learn.unfreeze()
do_fit(learn, f'{mname}.{step:02d}', lrs=slice(lr/300, lr/10, None), cycle_len=cycles*3)


#%%
hr_tifs = dataset/'hr_t_1024'
lr_up_tifs = dataset/'lrup_t_1024'


step = 0
lr = 1e-4
cycles = 2
loss = F.mse_loss
metrics = sr_metrics


bs = 2
size = 256
max_zoom = 1.2
arch = xresnet50

data = get_data(bs, size, lr_up_tifs, hr_tifs, max_zoom=max_zoom, num_workers=4)
learn = xres_unet_learner(data, arch, loss_func=loss, metrics=metrics, model_dir=model_dir, path='.')
learn.load(f'{mname}.{(step-1):02d}')
gc.collect()


#%%
do_fit(learn, f'{mname}.{step:02d}', lrs=lr, cycle_len=cycles)


#%%
do_fit(learn, f'{mname}.{step:02d}.1', lrs=lr/50, cycle_len=cycles*5)


#%%



#%%



#%%



#%%



#%%
p = Path('stats/input/mitotracker/')

test_fns = []

test_fns += list(p.glob('*.tif'))
test_fns += list(p.glob('**/**/*.czi'))

#test_fns = test_fns[0:1]


#%%
test_fns


#%%
#test_fns = []
#test_fns += list(test_files.glob('**/*.tif'))
#test_fns += list(test_files.glob('**/*.czi'))


#%%
step = 0
lr = 1e-4
cycles = 2
loss = F.mse_loss
metrics = sr_metrics


bs = 1
size = 512
max_zoom = 2
arch = xresnet50

data = get_data(bs, size, lr_up_tifs, hr_tifs, max_zoom=max_zoom)
learn = xres_unet_learner(data, arch, loss_func=loss, metrics=metrics, model_dir=model_dir, path='.')
learn.load(f'combo2_best').to_fp16()
gc.collect()


#%%
print('READY')


#%%
dest = Path('/DATA/temp/')
shutil.rmtree(dest)
dest.mkdir(exist_ok=True, parents=True)
generate_tifs(test_fns, dest, learn, size, tag=mname, max_imgs=10)


#%%
#test = PIL.Image.open('/DATA/temp/actin/C2-low lres confocal mito and actin 3_combo_orig.tif')


#%%
9*256


#%%
learn.export('combotile.pkl')


#%%
learn.path


#%%
get_ipython().run_line_magic('pinfo', 'shutil.copy')


#%%
learn.path = Path('.')


#%%
import czifile
from pathlib import Path
from fastprogress import progress_bar
import pandas as pd
import numpy as np
import PIL.Image
from bpho import *
PIL.Image.MAX_IMAGE_PIXELS = 99999999999999


#%%


p = Path('data/')
fns = list(p.glob('**/mitotracker/**/*.czi'))
print(len(fns))
fn = fns[0]


#%%
info = []
for fn in progress_bar(fns):
     with PIL.Image.open(fn) as tif_f:
        img = np.array(tif_f)
        info.append({'fn': fn, 'category': fn.parts[-3],
                     'ndtype': img.dtype})
df = pd.DataFrame(info)


#%%
df


#%%
f = fns[50]


#%%
get_ipython().system('ls data/fixed')


#%%
get_ipython().system('ls data/fixed/fixed_cell_mitochondria/train')


#%%
tif_f = PIL.Image.open(fn)
a = np.array(tif_f)
a.shape


#%%
mi,ma = np.percentile(a, [2,99.8])
print(mi,ma)


#%%
eps = 1e-20 
a = (a - mi) / (ma - mi + eps)


#%%
from fastai import *
from fastai.vision import *
img = Image(tensor(a[None]))


#%%
img


#%%
info = []
for fn in progress_bar(fns):
    if fn.parts[-4] != 'random' and 'great quality' not in fn.stem:
        try:
            with czifile.CziFile(fn) as czi_f:
                data = czi_f.asarray()                
                mi, ma = np.percentile(data, [2, 99.8])
                info.append({'fn': fn, 'ndtype': czi_f.dtype, 'category': fn.parts[-3],
                             'maxval': data.max(), 'mi': mi, 'ma': ma})
        except:
            print('exception', fn)
            pass
df = pd.DataFrame(info)


#%%
df.columns = ['category', 'dtype', 'fn', 'maxval']


#%%
mi, ma = np.percentile(data, [2, 99.8])


#%%
ndata = (data - mi)


#%%
mi


#%%
df


#%%
czi_f = czifile.CziFile(fn)


#%%
data = czi_f.asarray()


#%%
data.dtype, data.shape


#%%
proc_axes, proc_shape = get_czi_shape_info(czi_f)
idx = build_index(proc_axes, {'X': slice(0,18975), 'Y': slice(0,18886)})


#%%
np.percentile(data, 10)


#%%
czi_f.metadata


#%%
get_ipython().run_line_magic('pylab', 'inline')


#%%
2**10


#%%
2**12


#%%
2*14


#%%
2**14


#%%
2**13


#%%
df


#%%



