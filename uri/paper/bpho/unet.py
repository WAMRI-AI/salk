"xres unet building functions"
from fastai import *
from fastai.vision import *
from fastai.vision.models.xresnet import *
from fastai.vision.models.unet import DynamicUnet

__all__ = ['xres_unet_model', 'xres_unet_learner']

def xres_unet_model(in_c, out_c, arch):
    body = nn.Sequential(*list(arch(c_in=in_c).children())[:-2])
    model = DynamicUnet(
        body, n_classes=out_c,
        blur=True, blur_final=True,
        self_attention=True, norm_type=NormType.Weight, 
        last_cross=True, bottle=True
    )
    return model

def xres_unet_learner(data, arch, in_c=1, out_c=1, **kwargs):
    model = xres_unet_model(in_c, out_c, arch)
    learn = Learner(data, model, **kwargs) 
    return learn

def image_from_tiles(learn, img, tile_sz=128, scale=4):
    pimg = PIL.Image.fromarray((img*255).astype(np.uint8), mode='L').convert('RGB')
    cur_size = pimg.size
    new_size = (cur_size[0]*scale, cur_size[1]*scale)
    in_img = Image(pil2tensor(pimg.resize(new_size, resample=PIL.Image.BICUBIC),np.float32).div_(255))
    c, w, h = in_img.shape
    
    in_tile = torch.zeros((c,tile_sz,tile_sz))
    out_img = torch.zeros((c,w,h))
    
    for x_tile in range(math.ceil(w/tile_sz)):
        for y_tile in range(math.ceil(h/tile_sz)):
            x_start = x_tile

            x_start = x_tile*tile_sz
            x_end = min(x_start+tile_sz, w)
            y_start = y_tile*tile_sz
            y_end = min(y_start+tile_sz, h)
            
            
            in_tile[:,0:(x_end-x_start), 0:(y_end-y_start)] = in_img.data[:,x_start:x_end, y_start:y_end]
            
            out_tile,_,_ = learn.predict(Image(in_tile))

            out_x_start = x_start
            out_x_end = x_end
            out_y_start = y_start
            out_y_end = y_end

            in_x_start = 0
            in_y_start = 0
            in_x_end = x_end-x_start
            in_y_end = y_end-y_start
           
            out_img[:,out_x_start:out_x_end, out_y_start:out_y_end] = out_tile.data[:,
                                                                                  in_x_start:in_x_end, 
                                                                                  in_y_start:in_y_end]
    return out_img