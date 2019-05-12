
import yaml
yaml.warnings({'YAMLLoadWarning': False})
from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.vision.models.unet import DynamicUnet
from skimage.util import random_noise
from skimage import filters
from bpho import *
from bpho.resnet import *

torch.backends.cudnn.benchmark = True

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
             use_cutout=False,
             use_noise=True,
             scale=4,
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
            .databunch(bs=bs, **kwargs).normalize(do_y=True))
    data.c = 3
    return data


def do_fit(learn, save_name, lrs=slice(1e-3), pct_start=0.9, cycle_len=10):
    learn.to_fp16().fit_one_cycle(cycle_len, lrs, pct_start=pct_start)
    learn.save(save_name)
    print(f'saved: {save_name}')


@call_parse
def main(
        gpu: Param("GPU to run on", str)=None,
        arch: Param("encode architecture", str) = 'xresnet34',
        bs: Param("batch size per gpu", int) = 8,
        lr: Param("learning rate", float) = 1e-4,
        size: Param("img size", int) = 256,
        cycles: Param("num cyles", int) = 5,
        load_name: Param("load model name", str) = None,
        save_name: Param("model save name", str) = 'combo',
        datasetname: Param('dataset name', str) = 'tiles_002',
        tile_sz: Param('tile_sz', int) = 512,
        attn: Param('self attention', action='store_true')=False,
        blur: Param('upsample blur', action='store_true')=False,
        final_blur: Param('final upsample blur', action='store_true')=False,
        bottle: Param('bottleneck', action='store_true')=False,
        cutout: Param('bottleneck', action='store_true')=False,
        feat_loss: Param('bottleneck', action='store_true')=False
):
    data_path = Path('.')
    datasets = data_path/'datasets'
    datasources = data_path/'data'
    dataset = datasets/datasetname
    pickle_models = data_path/'stats/models'

    if tile_sz is None:
        hr_tifs = dataset/f'hr'
        lr_tifs = dataset/f'lr'
    else:
        hr_tifs = dataset/f'hr_t_{tile_sz:d}'
        lr_tifs = dataset/f'lr_t_{tile_sz:d}'

    print(datasets, dataset, hr_tifs)

    model_dir = 'models'

    gpu = setup_distrib(gpu)
    print('on gpu: ', gpu)
    n_gpus = num_distrib()

    loss = get_feat_loss() if feat_loss else F.l1_loss
    metrics = sr_metrics

    bs = max(bs, bs * n_gpus)
    size = size
    arch = eval(arch)

    print('bs:', bs, 'size: ', size, 'ngpu:', n_gpus)
    data = get_data(bs, size, lr_tifs, hr_tifs, max_zoom=4., use_cutout=cutout)
    xres_args = {
        'blur': blur,
        'blur_final': final_blur,
        'bottle': bottle,
        'self_attention': attn
    }

    callback_fns = []
    if gpu == 0 or gpu is None:
        if feat_loss:
            callback_fns = [LossMetrics]
        callback_fns.append(partial(SaveModelCallback, name=f'{save_name}_best_{size}'))
    learn = xres_unet_learner(data, arch, path=Path('.'), xres_args=xres_args, loss_func=loss, metrics=metrics, model_dir=model_dir, callback_fns=callback_fns)
    gc.collect()

    if load_name:
        learn = learn.load(f'{load_name}')
        print(f'loaded {load_name}')

    if gpu is None: learn.model = nn.DataParallel(learn.model)
    else: learn.to_distributed(gpu)
    learn = learn.to_fp16()
    learn.fit_one_cycle(cycles, lr)

    if gpu == 0 or gpu is None:
        learn.save(save_name)
        print(f'saved: {save_name}')
        learn.export(pickle_models/f'{save_name}_{size}.pkl')
        learn.load(f'{save_name}_best_{size}')
        learn.export(pickle_models/f'{save_name}_best_{size}.pkl')
        print('exported')
