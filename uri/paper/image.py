
import yaml
yaml.warnings({'YAMLLoadWarning': False})
from fastai.script import *
from fastai.vision import *
from fastai.vision.models.wrn import wrn_22
from fastai.distributed import *
from fastai.vision.models.xresnet import *
from fastai.vision.models.unet import DynamicUnet
from bpho import *


torch.backends.cudnn.benchmark = True




datasetname = 'foo_001'
data_path = Path('.')
datasets = data_path/'datasets'
datasources = data_path/'data'
dataset = datasets/datasetname

test_files = dataset/'test'
hr_tifs = dataset/'hr_t_256'
lr_tifs = dataset/'lr_t_256'
lrup_tifs = dataset/'lrup_t_256'

mname = 'combo_tile'
model_dir = 'models'

loss = F.mse_loss
metrics = sr_metrics

def get_src(x_data, y_data_):
    def map_to_hr(x):
        hr_name = x.relative_to(x_data)
        return y_data_/hr_name

    src = (NpyRawImageList
            .from_folder(x_data)
            .split_by_rand_pct()
            .label_from_func(map_to_hr, label_cls=NpyRawImageList))
    return src


def get_data(bs, size, x_data, y_data, **kwargs):
    src = get_src(x_data, y_data)
    tfms = get_transforms(flip_vert=True)
    data = (src
            .transform(tfms, size=size)
            .transform_y(tfms, size=size)
            .databunch(bs=bs,**kwargs))
    data.c = 3
    return data

def do_fit(learn, save_name, lrs=slice(1e-3), pct_start=0.9, cycle_len=10):
    learn.to_fp16().fit_one_cycle(cycle_len, lrs, pct_start=pct_start)
    learn.save(save_name)
    print(f'saved: {save_name}')
    num_rows = min(learn.data.batch_size, 3)




test_fns = []
test_fns += list(datasources.glob('**/test/*.tif'))
test_fns += list(datasources.glob('**/test/*.czi'))

# test_fns = test_fns[0:1]

step = 0
lr = 1e-4
cycles = 2
loss = F.mse_loss
metrics = sr_metrics

bs = 1
size = 512
max_zoom = 2
arch = xresnet50

data = get_data(bs, size, lrup_tifs, hr_tifs)
learn = xres_unet_learner(data, arch, path=Path('.'), loss_func=loss, metrics=metrics, model_dir=model_dir)
learn.load(f'distrib2_best').to_fp16()
gc.collect()



dest = Path('/DATA/temp/')
shutil.rmtree(dest)
dest.mkdir(exist_ok=True, parents=True)
generate_tifs(test_fns, dest, learn, size, tag=mname, max_imgs=10)
