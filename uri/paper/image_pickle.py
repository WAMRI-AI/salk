
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
        load_name: Param("load model name", Path, opt=False),
        src_dir: Param("input image dir", Path, opt=False),
        dest_dir: Param("output image dir", Path, opt=False),
        gpu: Param("GPU to run on", int)=None,
):
    torch.cuda.set_device(gpu)

    print('load_name', load_name)
    print('src_dir', src_dir)
    print('dest_dir', dest_dir)

    src_dir, dest_dir = Path(src_dir), Path(dest_dir)
    if not src_dir.exists():
        print("src_dir doesn't exist")
        return 1

    size = tile_sz = int(load_name.stem.split('_')[-1])
    learn = load_learner(load_name.parent, load_name.name)
    learn = learn.to_fp16()



    movie_files = []
    movie_files += list(src_dir.glob('**/*.czi'))
    movie_files += list(src_dir.glob('**/*.tif'))
    generate_tifs(movie_files, dest_dir, learn, size, tag=f'{load_name.stem}', max_imgs=10)
