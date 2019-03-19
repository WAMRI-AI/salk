from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import vgg16_bn
import PIL
import imageio
from superres import *
from scipy.ndimage.interpolation import zoom as npzoom
from skimage.util import img_as_float32, img_as_ubyte
from skimage.measure import compare_ssim, compare_psnr
import skimage.io as io

img_data = Path('/scratch/bpho/datasets/emsynth_003/')
model_path = Path('/scratch/bpho/models')


def get_src():
    hr_tifs = img_data/f'hr'
    lr_tifs = img_data/f'lr_up'

    def map_to_hr(x):
        hr_name = x.relative_to(lr_tifs)
        return hr_tifs/hr_name
    print(lr_tifs)
    src = (ImageImageList
            .from_folder(lr_tifs)
            .split_by_rand_pct()
            .label_from_func(map_to_hr))
    return src


def get_data(bs, size, noise=None, max_zoom=1.1):
    src = get_src()
    tfms = get_transforms(flip_vert=True, max_zoom=max_zoom)
    data = (src
            .transform(tfms, size=size)
            .transform_y(tfms, size=size)
            .databunch(bs=bs).normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data

@call_parse
def main(
        load_name: Param("load learner name", str)="em_save",
        save_dir: Param("dir to save to:", str)="/scratch/bpho/results/emsynth_crap",
        gpu:Param("GPU to run on", str)=0,
        ):
    torch.cuda.set_device(gpu)

    bs = 1
    size = 1920
    data = get_data(bs, size)

    arch = models.resnet34
    wd = 1e-3
    learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, metrics=superres_metrics,
                        callback_fns=LossMetrics, blur=True, norm_type=NormType.Weight, model_dir=model_path)
    gc.collect()

    learn = learn.load(load_name)

    test_files = Path('/scratch/bpho/datasources/EM_manually_aquired_pairs_01242019/')
    test_hr = list((test_files/'aligned_hr').glob('*.tif'))
    test_lr = list((test_files/'aligned_lr').glob('*.tif'))
    results = Path(save_dir)

    if results.exists(): shutil.rmtree(results)
    results.mkdir(parents=True, mode=0o775, exist_ok=True)

    def get_key(fn):
        return fn.stem[0:(fn.stem.find('Region')-1)]

    hr_map = { get_key(fn): fn for fn in test_hr }
    lr_map = { get_key(fn): fn for fn in test_lr }

    ssims = []
    psnrs = []
    for k in progress_bar(hr_map):
        hr_fn, lr_fn = hr_map[k], lr_map[k]
        hr_img = PIL.Image.open(hr_fn)
        lr_img = PIL.Image.open(lr_fn)
        lr_img_data = img_as_float32(lr_img)
        lr_up_data = npzoom(lr_img_data, 4, order=1)
        lr_up_img = Image(tensor(lr_up_data[None]))
        hr_pred_img, aaa, bbb = learn.predict(lr_up_img)
        pred_img = PIL.Image.fromarray(img_as_ubyte(np.array(hr_pred_img.data))[0,:,:])

        lr_img.save(results/f'{k}_orig.tif')
        hr_img.save(results/f'{k}_truth.tif')
        pred_img.save(results/f'{k}_pred.tif')
        hr_img_data = np.array(hr_img)

        ssims.append(compare_ssim(img_as_float32(np.array(hr_img)), img_as_float32(np.array(pred_img))))
        psnrs.append(compare_psnr(img_as_float32(np.array(hr_img)), img_as_float32(np.array(pred_img))))
    print(np.array(ssims).mean(), np.array(psnrs).mean())

    #target_path = Path('/DATA/Dropbox/bpho_movie_results/emsynth_003/')
    target_path = results

    orig,tru,pred = [list(target_path.glob(f'*{tag}*')) for tag in ['orig','tru','pred']]
    orig.sort()
    tru.sort()
    pred.sort()


    ssims = []
    c_ssims = []
    l_ssims = []
    psnrs = []
    c_psnrs = []
    l_psnrs = []

    for o, t,p in progress_bar(list(zip(orig, tru,pred))):
        oimg, timg, pimg = [img_as_float32(io.imread(fn)) for fn in [o,t,p]]
        if len(pimg.shape) == 3: pimg = pimg[:,:,0]
        cimg = npzoom(oimg, 4)
        limg = npzoom(oimg, 4, order=1)

        ssims.append(compare_ssim(timg, pimg))
        c_ssims.append(compare_ssim(timg, cimg))
        l_ssims.append(compare_ssim(timg, limg))
        psnrs.append(compare_psnr(timg, pimg))
        c_psnrs.append(compare_psnr(timg, cimg))
        l_psnrs.append(compare_psnr(timg, limg))

    import pandas as pd

    df = pd.DataFrame(dict(ssim=ssims, psnr=psnrs, 
                            bicubic_ssim=c_ssims, bicubic_psnr=c_psnrs,
                            bilinear_ssim=l_ssims, bilinear_psnr=l_psnrs))

    df.describe()
    print(df.describe())
