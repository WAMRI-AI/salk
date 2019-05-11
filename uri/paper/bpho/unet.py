"xres unet building functions"
from fastai import *
from fastai.vision import *
from fastai.vision.models.xresnet import *
from fastai.vision.models.unet import DynamicUnet

__all__ = ['xres_unet_model', 'xres_unet_learner', 'BilinearWrapper']

class BilinearWrapper(nn.Module):
    def __init__(self, model, scale=4, mode='bilinear'):
        super().__init__()
        self.model = model
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        return self.model(F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False))

def xres_unet_model(in_c, out_c, arch, blur=True, blur_final=True, self_attention=True, last_cross=True, bottle=True, norm_type=NormType.Weight, **xres_args):
    body = nn.Sequential(*list(arch(c_in=in_c).children())[:-2])
    print('blur', blur, 'blur_final', blur_final)
    model = DynamicUnet(body,
                        n_classes=out_c,
                        blur=blur,
                        blur_final=blur_final,
                        self_attention=self_attention,
                        norm_type=norm_type,
                        last_cross=last_cross,
                        bottle=bottle, **xres_args)
    return model


def xres_unet_learner(data, arch, in_c=1, out_c=1, xres_args=None, bilinear_upsample=True, **kwargs):
    if xres_args is None: xres_args = {}

    model = xres_unet_model(in_c, out_c, arch, **xres_args)
    if bilinear_upsample:
        model = BilinearWrapper(model)
    learn = Learner(data, model, **kwargs)
    return learn


def image_from_tiles(learn, img, tile_sz=128, scale=4):
    pimg = PIL.Image.fromarray((img * 255).astype(np.uint8),
                               mode='L').convert('RGB')
    cur_size = pimg.size
    new_size = (cur_size[0] * scale, cur_size[1] * scale)
    in_img = Image(
        pil2tensor(pimg.resize(new_size, resample=PIL.Image.BICUBIC),
                   np.float32).div_(255))
    c, w, h = in_img.shape

    in_tile = torch.zeros((c, tile_sz, tile_sz))
    out_img = torch.zeros((c, w, h))

    for x_tile in range(math.ceil(w / tile_sz)):
        for y_tile in range(math.ceil(h / tile_sz)):
            x_start = x_tile

            x_start = x_tile * tile_sz
            x_end = min(x_start + tile_sz, w)
            y_start = y_tile * tile_sz
            y_end = min(y_start + tile_sz, h)

            in_tile[:, 0:(x_end - x_start), 0:(
                y_end -
                y_start)] = in_img.data[:, x_start:x_end, y_start:y_end]

            out_tile, _, _ = learn.predict(Image(in_tile))

            out_x_start = x_start
            out_x_end = x_end
            out_y_start = y_start
            out_y_end = y_end

            in_x_start = 0
            in_y_start = 0
            in_x_end = x_end - x_start
            in_y_end = y_end - y_start

            out_img[:, out_x_start:out_x_end, out_y_start:
                    out_y_end] = out_tile.data[:, in_x_start:in_x_end,
                                               in_y_start:in_y_end]
    return out_img
