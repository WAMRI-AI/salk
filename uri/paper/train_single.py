
import yaml
yaml.warnings({'YAMLLoadWarning': False})
from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.vision.models.xresnet import *
from fastai.vision.models.unet import DynamicUnet
from skimage.util import random_noise
from skimage import filters
from bpho import *


torch.backends.cudnn.benchmark = True

def get_src(x_data, y_data):
    def map_to_hr(x):
        hr_name = x.relative_to(x_data)
        return y_data/hr_name

    src = (ImageImageList
            .from_folder(x_data, convert_mode='L')
            .split_by_rand_pct()
            .label_from_func(map_to_hr, convert_mode='L'))
    return src

def _my_noise(x, gauss_sigma=1.):
    c,h,w = x.shape
    noise = torch.zeros((1,h,w))
    noise.normal_(0, gauss_sigma)
    img_max = np.minimum(1.1 * x.max(), 1.)
    x = np.minimum(np.maximum(0,x+noise), img_max)
    x = random_noise(x, mode='salt', amount=0.005)
    x = random_noise(x, mode='pepper', amount=0.005)
    return x

my_noise = TfmPixel(_my_noise)

def get_xy_transforms(max_rotate=10., min_zoom=1., max_zoom=2.):
    base_tfms = [[rand_crop(),
                   dihedral_affine(),
                   rotate(degrees=(-max_rotate,max_rotate)),
                   rand_zoom(min_zoom, max_zoom)],
                 [crop_pad()]]

    y_tfms = [[tfm for tfm in base_tfms[0]], [tfm for tfm in base_tfms[1]]]
    x_tfms = [[tfm for tfm in base_tfms[0]], [tfm for tfm in base_tfms[1]]]
    x_tfms[0].append(cutout())
    x_tfms[0].append(my_noise())

    return x_tfms, y_tfms

def get_data(bs, size, x_data, y_data, max_zoom=1.1):
    src = get_src(x_data, y_data)
    x_tfms, y_tfms = get_xy_transforms()
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
        datasetname: Param('dataset name', str) = 'tiles_002',
        tile_sz: Param('tile_sz', int) = 512,
):
    data_path = Path('.')
    datasets = data_path/'datasets'
    datasources = data_path/'data'
    dataset = datasets/datasetname
    pickle_models = data_path/'stats/models'

    if tile_sz is None:
        hr_tifs = dataset/f'hr'
        lrup_tifs = dataset/f'lrup'
    else:
        hr_tifs = dataset/f'hr_t_{tile_sz:d}'
        lrup_tifs = dataset/f'lrup_t_{tile_sz:d}'

    print(datasets, dataset, hr_tifs, lrup_tifs)

    model_dir = 'models'

    gpu = setup_distrib(gpu)
    print('on gpu: ', gpu)
    n_gpus = num_distrib()

    loss = F.mse_loss
    metrics = sr_metrics

    bs = min(bs, bs * n_gpus)
    size = size
    arch = eval(arch)

    data = get_data(bs, size, lrup_tifs, hr_tifs)
    if gpu == 0 or gpu is None:
        callback_fns = []
        callback_fns.append(partial(SaveModelCallback, name=f'{save_name}_best_{size}'))
        learn = xres_unet_learner(data, arch, path=Path('.'), loss_func=loss, metrics=metrics, model_dir=model_dir, callback_fns=callback_fns)
    else:
        learn = xres_unet_learner(data, arch, path=Path('.'), loss_func=loss, metrics=metrics, model_dir=model_dir)
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
