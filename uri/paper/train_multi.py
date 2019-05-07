
import yaml
yaml.warnings({'YAMLLoadWarning': False})
from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.vision.models.xresnet import *
from fastai.vision.models.unet import DynamicUnet
from bpho import *


torch.backends.cudnn.benchmark = True

def get_src(x_data, y_data):
    def map_to_hr(x):
        hr_name = x.relative_to(x_data).with_suffix('.tif')
        return y_data/hr_name

    src = (MultiImageImageList
            .from_folder(x_data, extensions=['.npy'])
            .split_by_folder()
            .label_from_func(map_to_hr, convert_mode='L'))
    return src


def get_data(bs, size, x_data, y_data, use_noise=False, use_cutout=False):
    src = get_src(x_data, y_data)
    x_tfms, y_tfms = get_xy_transforms(use_noise=use_noise, use_cutout=use_cutout)
    data = (src
            .transform(x_tfms, size=size)
            .transform_y(y_tfms, size=size)
            .databunch(bs=bs))
    data.c = 3
    return data


def do_fit(learn, save_name, lrs=slice(1e-3), pct_start=0.9, cycle_len=10):
    learn.to_fp16().fit_one_cycle(cycle_len, lrs, pct_start=pct_start)
    learn.save(save_name)
    print(f'saved: {save_name}')
    #num_rows = min(learn.data.batch_size, 3)
    #learn.to_fp32().show_results(rows=num_rows, imgsize=5)


def get_model(in_c, out_c, arch):
    body = nn.Sequential(*list(arch(c_in=in_c).children())[:-2])
    model = DynamicUnet(
        body, n_classes=out_c,
        blur=True, blur_final=True,
        self_attention=True, norm_type=NormType.Weight,
        last_cross=True, bottle=True
    )
    return model


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
        datasetname: Param('dataset name', str) = 'multi_norm_001',
        tile_sz: Param('tile_sz', int) = 512,
        blur: Param("blur upsample layers", bool)=True,
        blur_final: Param("blur final upsample layer", bool)=True,
        noise: Param("train on depth", action='store_true')=False,
        cutout: Param("train on depth", action='store_true')=False,
        depth: Param("train on depth", action='store_true')=False,
        times: Param("train on depth", action='store_true')=False,
        n_frames: Param('n_frames', int) = 5
):
    if (not depth and not times) or (depth and times):
        print('need to train on depth or time, and not both')

    if depth: t_z = 'mz'
    if times: t_z = 'mt'
    data_path = Path('.')
    datasets = data_path/'datasets'
    datasources = data_path/'data'
    dataset = datasets/datasetname
    pickle_models = data_path/'stats/models'

    hr_multi_tifs = dataset/f'hr_t_{tile_sz}_{t_z}_{n_frames}'
    lrup_multi_tifs = dataset/f'lrup_t_{tile_sz}_{t_z}_{n_frames}'

    print(datasets, dataset, hr_multi_tifs, lrup_multi_tifs)

    model_dir = 'models'

    gpu = setup_distrib(gpu)
    print('on gpu: ', gpu)
    n_gpus = num_distrib()

    loss = F.mse_loss
    metrics = sr_metrics

    bs = max(bs, bs * n_gpus)
    size = size
    arch = eval(arch)

    data = get_data(bs, size, lrup_multi_tifs, hr_multi_tifs, use_noise=noise, use_cutout=cutout)
    xres_args = {
        'blur_final': blur_final,
        'blur': blur
    }
    if gpu == 0 or gpu is None:
        callback_fns = []
        callback_fns.append(partial(SaveModelCallback, name=f'{save_name}_best_{size}'))
        learn = xres_unet_learner(data, arch, in_c=n_frames, path=Path('.'),
                                  loss_func=loss, metrics=metrics, model_dir=model_dir,
                                  callback_fns=callback_fns, xres_args=xres_args)
    else:
        learn = xres_unet_learner(data, arch, in_c=n_frames, path=Path('.'),
                                  loss_func=loss, metrics=metrics, model_dir=model_dir, xres_args=xres_args)
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
