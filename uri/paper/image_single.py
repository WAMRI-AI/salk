
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

    src = (NpyRawImageList
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
        load_name: Param("load model name", str, opt=False),
        src_dir: Param("input image dir", Path, opt=False),
        dest_dir: Param("output image dir", Path, opt=False),
        gpu: Param("GPU to run on", str)=None,
        arch: Param("encode architecture", str) = 'xresnet18',
        size: Param("img size", int) = 256,
        tile_sz: Param('tile_sz', int) = 512
):
    print('src_dir', src_dir)
    print('dest_dir', dest_dir)
    src_dir, dest_dir = Path(src_dir), Path(dest_dir)
    if not src_dir.exists():
        print("src_dir doesn't exist")
        return 1

    datasetname = 'combo_001'
    bs = 1
    lr = 1e-4
    data_path = Path('.')
    datasets = data_path/'datasets'
    datasources = data_path/'data'
    dataset = datasets/datasetname

    hr_multi_tifs = dataset/f'hr_t_{tile_sz}'
    lrup_multi_tifs = dataset/f'lrup_t_{tile_sz}'
    model_dir = 'models'

    gpu = setup_distrib(gpu)
    print('on gpu: ', gpu)
    n_gpus = num_distrib()

    loss = F.mse_loss
    metrics = sr_metrics

    size = size
    arch = eval(arch)
    data = get_data(bs, size, lrup_multi_tifs, hr_multi_tifs)
    learn = xres_unet_learner(data, arch, path=Path('.'), loss_func=loss, metrics=metrics, model_dir=model_dir)
    gc.collect()

    if load_name:
        learn.load(load_name)
        print(f'loaded {load_name}')
    if gpu is None: learn.model = nn.DataParallel(learn.model)
    else: learn.to_distributed(gpu)
    learn = learn.to_fp16()


    movie_files = []
    movie_files += list(src_dir.glob('**/*.czi'))
    movie_files += list(src_dir.glob('**/*.tif'))
    generate_tifs(movie_files, dest_dir, learn, size, tag='load_name', max_imgs=10)
