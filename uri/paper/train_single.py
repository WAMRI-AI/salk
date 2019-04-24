
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
        hr_name = x.relative_to(x_data)
        return y_data/hr_name

    src = (MultiImageImageList
            .from_folder(x_data, extensions=['.npy'])
            .split_by_rand_pct()
            .label_from_func(map_to_hr, label_cls=NpyRawImageList))
    return src


def get_data(bs, size, x_data, y_data, max_zoom=1.1):
    src = get_src(x_data, y_data)
    tfms = get_transforms(flip_vert=True, max_lighting=None, max_zoom=max_zoom)
    data = (src
            .transform(tfms, size=size)
            .transform_y(tfms, size=size)
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
        arch: Param("encode architecture", str) = 'xresnet50',
        bs: Param("batch size per gpu", int) = 4,
        lr: Param("learning rate", float) = 1e-4,
        size: Param("img size", int) = 256,
        cycles: Param("num cyles", int) = 2,
        load_name: Param("load model name", str) = None,
        save_name: Param("model save name", str) = 'multi',
        datasetname: Param('dataset name', str) = 'foo_003',
        tile_sz: Param('tile_sz', int) = 512,
        n_frames: Param('n_frames', int) = 5
):
    data_path = Path('.')
    datasets = data_path/'datasets'
    datasources = data_path/'data'
    dataset = datasets/datasetname

    hr_multi_tifs = dataset/f'hr_mt_{n_frames:02d}_tiles_{tile_sz:04d}'
    lrup_multi_tifs = dataset/f'lrup_mt_{n_frames:02d}_tiles_{tile_sz:04d}'
    mname = 'combo_multi_tile'
    model_dir = 'models'

    gpu = setup_distrib(gpu)
    print('on gpu: ', gpu)
    n_gpus = num_distrib()

    loss = F.mse_loss
    metrics = sr_metrics

    bs = bs * n_gpus
    size = size
    arch = eval(arch)
    data = get_data(bs, size, lrup_multi_tifs, hr_multi_tifs)
    if gpu == 0 or gpu is None:
        callback_fns = []
        callback_fns.append(partial(SaveModelCallback, name=f'{save_name}_best'))
        learn = xres_unet_learner(data, arch, in_c=n_frames, path=Path('.'), loss_func=loss, metrics=metrics, model_dir=model_dir, callback_fns=callback_fns)
    else:
        learn = xres_unet_learner(data, arch, in_c=n_frames, path=Path('.'), loss_func=loss, metrics=metrics, model_dir=model_dir)
    gc.collect()

    if load_name: 
        learn.load(load_name)
        print(f'loaded {load_name}')
    if gpu is None: learn.model = nn.DataParallel(learn.model)
    else: learn.to_distributed(gpu)
    learn = learn.to_fp16()
    learn.fit_one_cycle(cycles, lr)
    if gpu == 0 or gpu is None:
        learn.save(save_name)
        print(f'saved: {save_name}')
