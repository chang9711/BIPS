""" utils.py
"""

import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import time


def adjust_dyn_range(x, drange_in, drange_out):
    if not drange_in == drange_out:
        scale = float(drange_out[1]-drange_out[0])/float(drange_in[1]-drange_in[0])
        bias = drange_out[0]-drange_in[0]*scale
        x = x.mul(scale).add(bias)
    return x


def resize(x, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale(size),
        transforms.ToTensor(),
        ])
    return transform(x)


def make_image_grid(x, ngrid):
    x = x.clone().cpu()
    if pow(ngrid,2) < x.size(0):
        grid = make_grid(x[:ngrid*ngrid], nrow=ngrid, padding=0, normalize=True, scale_each=False)
    else:
        grid = torch.FloatTensor(ngrid*ngrid, x.size(1), x.size(2), x.size(3)).fill_(1)
        grid[:x.size(0)].copy_(x)
        grid = make_grid(grid, nrow=ngrid, padding=0, normalize=True, scale_each=False)
    return grid


def save_image_single(x, path, imsize=256):
    from PIL import Image
    grid = make_image_grid(x, 1)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((imsize*2,imsize), Image.NEAREST)
    im.save(path)


def save_image_grid(x, path, imsize=256, ngrid=4):
    from PIL import Image
    grid = make_image_grid(x, ngrid)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((imsize*2,imsize), Image.NEAREST)
    im.save(path)

def save_image_whole(inp, outp, gt, config, path, max_batch = 4):
    scale_d = config.scale_d

    from PIL import Image
    import matplotlib.pyplot as plt
    bsize = min(outp.shape[0], max_batch)

    cm = plt.get_cmap('jet')

    gt = np.array(gt.cpu().transpose(1,2).transpose(3,2))
    inp = np.array(inp.cpu().transpose(1,2).transpose(3,2))
    outp = np.array(outp.cpu().transpose(1,2).transpose(3,2)) # Batch, Height, Width, Channel
    inp_rgb = (255*(inp[:,:,:,:3]+1)/2).astype(np.uint8)
    if 'D' in config.input_mode or 'L' in config.input_mode:
        inp_d = (((inp[:,:,:,3:])/scale_d))
    else:
        inp_d = (inp_rgb * 0)[:,:,:,:1].astype(np.uint8)

    gt_rgb = (255*(gt[:,:,:,:3]+1)/2).astype(np.uint8)
    gt_d = (((gt[:,:,:,3])/scale_d))
    outp_rgb = (255*(outp[:,:,:,:3]+1)/2).astype(np.uint8)
    outp_d = (((outp[:,:,:,3])/scale_d))

  
    gt_d_max = gt_d.max(-1).max(-1, keepdims=True)
    outp_d_max = outp_d.max(-1).max(-1, keepdims=True)
    d_max = np.concatenate( [gt_d_max, outp_d_max], axis=-1)
    d_max = d_max.max(-1) # To normalize the visualized depth map.

    error_d = (np.abs(gt_d - outp_d))
    error_rgb = np.repeat(np.abs( np.mean(gt_rgb, axis=-1, keepdims=True) - np.mean(outp_rgb, axis=-1, keepdims=True) ), 3, axis=-1)
    img = np.concatenate( [gt_rgb[0], (255*cm(gt_d/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis=-2)
    img = np.concatenate( [img, np.concatenate([inp_rgb[0], (255*cm(inp_d[:,:,:,0]/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis=-2)], axis=-3)
    if len(config.input_mode) == 5:
        img = np.concatenate( [img, np.concatenate([0*inp_rgb[0], (255*cm(inp_d[:,:,:,1]/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis=-2)], axis=-3)
    img = np.concatenate([ img, np.concatenate([outp_rgb[0], (255*cm(outp_d/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)
    img = np.concatenate([ img, np.concatenate([((error_rgb)[0,:,:,:3]).astype(np.uint8), (255*cm(error_d/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)
    for b in range( 1, bsize ):
        img = np.concatenate( [img, np.concatenate([gt_rgb[b], (255*cm(gt_d/d_max[b])[b,:,:,:3]).astype(np.uint8)], axis=-2)], axis=-3)
        img = np.concatenate( [img, np.concatenate([inp_rgb[b], (255*cm(inp_d[:,:,:,0]/d_max[b])[b,:,:,:3]).astype(np.uint8)], axis=-2)], axis=-3)
        if len(config.input_mode) == 5:
            img = np.concatenate( [img, np.concatenate([0*inp_rgb[b], (255*cm(inp_d[:,:,:,1]/d_max[b])[b,:,:,:3]).astype(np.uint8)], axis=-2)], axis=-3)
        img = np.concatenate( [img, np.concatenate([outp_rgb[b], (255*cm(outp_d/d_max[b])[b,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)
        img = np.concatenate([ img, np.concatenate([((error_rgb)[b,:,:,:3]).astype(np.uint8), (255*cm(error_d/d_max[b])[b,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)
    im = Image.fromarray(img)
    im.save(path)

def save_image_whole_test(inp, outp, gt, config, path, max_batch = 1):
    scale_d = config.scale_d
    from PIL import Image
    import matplotlib.pyplot as plt

    cm = plt.get_cmap('jet')

    gt = np.array(gt.cpu().transpose(1,2).transpose(3,2))
    inp = np.array(inp.cpu().transpose(1,2).transpose(3,2))
    outp = np.array(outp.cpu().transpose(1,2).transpose(3,2)) # Batch, Height, Width, Channel

    inp_rgb = (255*(inp[:,:,:,:3]+1)/2).astype(np.uint8)
    if 'D' in config.input_mode or 'L' in config.input_mode:
        inp_d = (((inp[:,:,:,3:])/scale_d))
    else:
        inp_d = (inp_rgb * 0)[:,:,:,:1].astype(np.uint8)

    gt_rgb = (255*(gt[:,:,:,:3]+1)/2).astype(np.uint8)
    gt_d = (((gt[:,:,:,3])/scale_d))
    outp_rgb = (255*(outp[:,:,:,:3]+1)/2).astype(np.uint8)
    outp_d = (((outp[:,:,:,3])/scale_d))

    gt_d_max = gt_d.max(-1).max(-1, keepdims=True)
    outp_d_max = outp_d.max(-1).max(-1, keepdims=True)
    d_max = np.concatenate( [gt_d_max, outp_d_max], axis=-1)
    d_max = d_max.max(-1) # To normalize the visualized depth map.

    error_d = (np.abs(gt_d - outp_d))
    error_rgb = np.repeat(np.abs( np.mean(gt_rgb, axis=-1, keepdims=True) - np.mean(outp_rgb, axis=-1, keepdims=True) ), 3, axis=-1)
    img = np.concatenate( [gt_rgb[0], (255*cm(gt_d/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis=-2)
    img = np.concatenate( [img, np.concatenate([inp_rgb[0], (255*cm(inp_d[:,:,:,0]/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis=-2)], axis=-3)
    
    img = np.concatenate([ img, np.concatenate([outp_rgb[0], (255*cm(outp_d/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)
    img = np.concatenate([ img, np.concatenate([((error_rgb)[0,:,:,:3]).astype(np.uint8), (255*cm(error_d/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)
    im = Image.fromarray(img)
    im.save(path)

def save_image_rgb(outp, config, path):
    from PIL import Image
    import matplotlib.pyplot as plt
    outp = np.array(outp.cpu().transpose(1,2).transpose(3,2)) # Batch, Height, Width, Channel
    img = (255*(outp+1)/2).astype(np.uint8)[0]
    im = Image.fromarray(img)
    im.save(path)

def save_image_d(outp, config, path):
    scale_d = config.scale_d
    from PIL import Image
    import matplotlib.pyplot as plt
    cm = plt.get_cmap('jet')
    outp = np.array(outp.cpu().transpose(1,2).transpose(3,2)) # Batch, Height, Width, Channel
    img = (((2**16-1)*(outp[0])/scale_d)).astype(np.uint16)
    #img = (255*cm(outp[:,:,:,0]/scale_d)).astype(np.uint8)[0,:,:,:3]
    im = Image.fromarray(img[:,:,0]).convert('I')
    #im = Image.fromarray(img)
    im.save(path)

def save_image_schematic(mask, inp, gt, ours, config, path, max_batch = 1):
    scale_d = config.scale_d
    from PIL import Image
    import matplotlib.pyplot as plt

    cm = plt.get_cmap('jet')

    gt = np.array(gt.cpu().transpose(1,2).transpose(3,2))
    inp = np.array(inp.cpu().transpose(1,2).transpose(3,2))
    mask = np.array(mask.cpu().transpose(1,2).transpose(3,2))
    ours = np.array(ours.cpu().transpose(1,2).transpose(3,2)) # Batch, Height, Width, Channel

    inp_rgb = (255*(inp[:,:,:,:3]+1)/2).astype(np.uint8)
    inp_d = (((inp[:,:,:,3:])/scale_d))
    gt_rgb = (255*(gt[:,:,:,:3]+1)/2).astype(np.uint8)
    gt_d = (((gt[:,:,:,3])/scale_d))
    mask_rgb = (255*(mask[:,:,:,:3]+1)/2).astype(np.uint8)
    mask_d = (255*(mask[:,:,:,3:]+1)/2).astype(np.uint8)
    mask_d = np.concatenate([mask_d,mask_d,mask_d], axis=-1)
    ours_rgb = (255*(ours[:,:,:,:3]+1)/2).astype(np.uint8)
    ours_d = (((ours[:,:,:,3])/scale_d))

    gt_d_max = gt_d.max(-1).max(-1, keepdims=True)
    ours_d_max = ours_d.max(-1).max(-1, keepdims=True)

    d_max = np.concatenate( [gt_d_max, ours_d_max], axis=-1)
    d_max = d_max.max() # To normalize the visualized depth map.
    
    white = 255*np.ones( (512, 10, 3) ).astype(np.uint8)

    gt_d = (255*cm(gt_d/d_max)[0,:,:,:3]).astype(np.uint8)
    inp_d = (255*cm(inp_d[:,:,:,0]/d_max)[0,:,:,:3]).astype(np.uint8)
    ours_d = (255*cm(ours_d/d_max)[0,:,:,:3]).astype(np.uint8)
    mask_d = mask_d[0]

    img = np.concatenate( [mask_rgb[0], white, gt_rgb[0], white, inp_rgb[0], white, ours_rgb[0]], axis=-2)
    d =   np.concatenate([mask_d, white, gt_d, white, inp_d, white, ours_d],axis=-2)
    img = np.concatenate( [img, d], axis=-3)

    im = Image.fromarray(img)
    im.save(path)

def save_image_samescene(gt, in1, out1, in2, out2, config, path, max_batch = 1):
    scale_d = config.scale_d
    from PIL import Image
    import matplotlib.pyplot as plt

    cm = plt.get_cmap('jet')

    gt = np.array(gt.cpu().transpose(1,2).transpose(3,2))
    in1 = np.array(in1.cpu().transpose(1,2).transpose(3,2))
    in2 = np.array(in2.cpu().transpose(1,2).transpose(3,2))
    out1 = np.array(out1.cpu().transpose(1,2).transpose(3,2)) # Batch, Height, Width, Channel
    out2 = np.array(out2.cpu().transpose(1,2).transpose(3,2)) # Batch, Height, Width, Channel

    in1_rgb = (255*(in1[:,:,:,:3]+1)/2).astype(np.uint8)
    in1_d = (((in1[:,:,:,3:])/scale_d))
    in2_rgb = (255*(in2[:,:,:,:3]+1)/2).astype(np.uint8)
    in2_d = (((in2[:,:,:,3:])/scale_d))
    gt_rgb = (255*(gt[:,:,:,:3]+1)/2).astype(np.uint8)
    gt_d = (((gt[:,:,:,3])/scale_d))
    out1_rgb = (255*(out1[:,:,:,:3]+1)/2).astype(np.uint8)
    out1_d = (((out1[:,:,:,3])/scale_d))
    out2_rgb = (255*(out2[:,:,:,:3]+1)/2).astype(np.uint8)
    out2_d = (((out2[:,:,:,3])/scale_d))

    gt_d_max = gt_d.max(-1).max(-1, keepdims=True)
    out1_d_max = out1_d.max(-1).max(-1, keepdims=True)
    out2_d_max = out2_d.max(-1).max(-1, keepdims=True)

    d_max = np.concatenate( [gt_d_max, out1_d_max, out2_d_max], axis=-1)
    d_max = d_max.max() # To normalize the visualized depth map.
    
    w = 5 if gt.shape[-3]==256 else 10
    white = 255*np.ones( (gt.shape[-3], w, 3) ).astype(np.uint8)

    gt_d = (255*cm(gt_d/d_max)[0,:,:,:3]).astype(np.uint8)
    in1_d = (255*cm(in1_d[:,:,:,0]/d_max)[0,:,:,:3]).astype(np.uint8)
    in2_d = (255*cm(in2_d[:,:,:,0]/d_max)[0,:,:,:3]).astype(np.uint8)
    out1_d = (255*cm(out1_d/d_max)[0,:,:,:3]).astype(np.uint8)
    out2_d = (255*cm(out2_d/d_max)[0,:,:,:3]).astype(np.uint8)

    img = np.concatenate( [gt_rgb[0], white, in1_rgb[0], white, out1_rgb[0], white, in2_rgb[0], white, out2_rgb[0]], axis=-2)
    d =     np.concatenate([gt_d, white, in1_d, white, out1_d, white, in2_d, white, out2_d],axis=-2)
    img = np.concatenate( [img, d], axis=-3)

    im = Image.fromarray(img)
    im.save(path)


def load_model(net, path):
    net.load_state_dict(torch.load(path))

def save_model(net, path):
    torch.save(net.state_dict(), path)


def make_summary(writer, key, value, step):
    if hasattr(value, '__len__'):
        for idx, img in enumerate(value):
            summary = tf.Summary()
            sio = BytesIO()
            scipy.misc.toimage(img).save(sio, format='png')
            image_summary = tf.Summary.Image(encoded_image_string=sio.getvalue())
            summary.value.add(tag="{}/{}".format(key, idx), image=image_summary)
            writer.add_summary(summary, global_step=step)
    else:
        summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
        writer.add_summary(summary, global_step=step)


def mkdir(path):
    if os.path.isdir(path):
        return
    if os.name == 'nt':
        os.system('mkdir {}'.format(path.replace('/', '\\')))
    else:
        os.system('mkdir {}'.format(path))


import torch
import math
irange = range
def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, t.min(), t.max())

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def save_depth_res(inp, outp, input, gt, config, path, max_batch = 4):
    scale_d = config.scale_d
    from PIL import Image   
    import matplotlib.pyplot as plt
    bsize = min(outp.shape[0], max_batch)
    
    cm = plt.get_cmap('jet')

    input = np.array(input.cpu().transpose(1,2).transpose(3,2))
    inp = np.array(inp.cpu().transpose(1,2).transpose(3,2))
    outp = np.array(outp.cpu().transpose(1,2).transpose(3,2))
    gt = np.array(gt.cpu().transpose(1,2).transpose(3,2))

  

    inp_rgb = (255*(inp[:,:,:,:3]+1)/2).astype(np.uint8)
    inp_d = (((inp[:,:,:,3:])/scale_d))
    
    input_rgb = (255*(input[:,:,:,:3]+1)/2).astype(np.uint8)
    input_d = (((input[:,:,:,3])/scale_d))
    
    
    outp_rgb = (255*(outp[:,:,:,:3]+1)/2).astype(np.uint8)
    outp_d_ini = (((outp[:,:,:,3])/scale_d))
    outp_d_res = (((outp[:,:,:,4]-5)*1.922/scale_d))
    outp_d = outp_d_ini-outp_d_res
    outp_d = np.where(outp_d<0,0,outp_d)

    gt_d = (((gt[:,:,:,3])/scale_d))

    
    input_d_max = input_d.max(-1).max(-1, keepdims=True)
    outp_d_max = outp_d.max(-1).max(-1, keepdims=True)
    d_max = np.concatenate( [input_d_max, outp_d_max], axis=-1)
    d_max = d_max.max(-1)

    img = np.concatenate( [input_rgb[0], (255*cm(input_d/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis=-2)
    img = np.concatenate( [img, np.concatenate([inp_rgb[0], (255*cm(inp_d[:,:,:,0]/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis=-2)], axis=-3)
    img = np.concatenate([ img, np.concatenate([outp_rgb[0], (255*cm(outp_d/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)
    
    img = np.concatenate([ img, np.concatenate([(255*cm(gt_d/d_max[0])[0,:,:,:3]).astype(np.uint8), (255*cm(outp_d_ini/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)
    for b in range( 1, bsize ):
        img = np.concatenate( [img, np.concatenate([input_rgb[b], (255*cm(input_d/d_max[b])[b,:,:,:3]).astype(np.uint8)], axis=-2)], axis=-3)
        img = np.concatenate( [img, np.concatenate([inp_rgb[b], (255*cm(inp_d[:,:,:,0]/d_max[b])[b,:,:,:3]).astype(np.uint8)], axis=-2)], axis=-3)
        img = np.concatenate( [img, np.concatenate([outp_rgb[b], (255*cm(outp_d/d_max[b])[b,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)

        img = np.concatenate( [img, np.concatenate([(255*cm(gt_d/d_max[b])[b,:,:,:3]).astype(np.uint8), (255*cm(outp_d/d_max[0])[b,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)
    im = Image.fromarray(img)
    im.save(path)    

def save_depth_res_ini(inp, outp, input, gt, config, path, max_batch = 4):
    scale_d = config.scale_d
    from PIL import Image   
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    bsize = min(outp.shape[0], max_batch)
    
    cm = plt.get_cmap('jet')
    cm1 = plt.get_cmap('binary')

    cNorm = colors.Normalize(vmin=-5, vmax=5)
    cmid = colors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)

    input = np.array(input.cpu().transpose(1,2).transpose(3,2))
    inp = np.array(inp.cpu().transpose(1,2).transpose(3,2))
    outp = np.array(outp.cpu().transpose(1,2).transpose(3,2))
    gt = np.array(gt.cpu().transpose(1,2).transpose(3,2))


    inp_rgb = (255*(inp[:,:,:,:3]+1)/2).astype(np.uint8)
    inp_d = (((inp[:,:,:,3:])/scale_d))
    
    input_rgb = (255*(input[:,:,:,:3]+1)/2).astype(np.uint8)
    input_d = (((input[:,:,:,3])/scale_d))
    
        
    outp_rgb = (255*(outp[:,:,:,:3]+1)/2).astype(np.uint8)
    outp_d_ini = (((outp[:,:,:,3])/scale_d))
    outp_d_res = (((outp[:,:,:,4]-5)*1.922/scale_d))
    outp_d = outp_d_ini-outp_d_res
    outp_d = np.where(outp_d<0,0,outp_d)

    gt_d = (((gt[:,:,:,3])/scale_d))
    gt_res_d = (gt[:,:,:,4]-5)
    

    res_d= (outp[:,:,:,4]-5)

    input_d_max = input_d.max(-1).max(-1, keepdims=True)
    outp_d_max = outp_d.max(-1).max(-1, keepdims=True)
    d_max = np.concatenate( [input_d_max, outp_d_max], axis=-1)
    d_max = d_max.max(-1)

    img = np.concatenate( [input_rgb[0], (255*cm(input_d/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis=-2)
    img = np.concatenate( [img, np.concatenate([inp_rgb[0], (255*cm(inp_d[:,:,:,0]/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis=-2)], axis=-3)
    img = np.concatenate([ img, np.concatenate([outp_rgb[0], (255*cm(outp_d/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)
    img = np.concatenate([ img, np.concatenate([(255*cm(gt_d/d_max[0])[0,:,:,:3]).astype(np.uint8), (255*cm(outp_d_ini/d_max[0])[0,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)
    img = np.concatenate([ img, np.concatenate([(255*(cm1(gt_res_d))[0,:,:,:3]).astype(np.uint8), (255*(cm1(res_d))[0,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)
    for b in range( 1, bsize ):
        img = np.concatenate( [img, np.concatenate([input_rgb[b], (255*cm(input_d/d_max[b])[b,:,:,:3]).astype(np.uint8)], axis=-2)], axis=-3)
        img = np.concatenate( [img, np.concatenate([inp_rgb[b], (255*cm(inp_d[:,:,:,0]/d_max[b])[b,:,:,:3]).astype(np.uint8)], axis=-2)], axis=-3)
        img = np.concatenate( [img, np.concatenate([outp_rgb[b], (255*cm(outp_d/d_max[b])[b,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)
        img = np.concatenate( [img, np.concatenate([(255*cm(gt_d/d_max[b])[b,:,:,:3]).astype(np.uint8), (255*cm(outp_d/d_max[0])[b,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)
        img = np.concatenate([ img, np.concatenate([(255*(cm1(gt_res_d))[b,:,:,:3]).astype(np.uint8), (255*(cm1(res_d))[b,:,:,:3]).astype(np.uint8)], axis= -2 )], axis=-3)
    im = Image.fromarray(img)
    im.save(path)  

def save_image_rgb_res(outp, config, path):
    from PIL import Image
    import matplotlib.pyplot as plt
    outp = np.array(outp.cpu().transpose(1,2).transpose(3,2)) # Batch, Height, Width, Channel
    outp_rgb = (255*(outp[:,:,:,:3]+1)/2).astype(np.uint8)
    im = Image.fromarray(outp_rgb[0])
    im.save(path)

def save_image_d_res(outp, config, path):
    scale_d = config.scale_d
    from PIL import Image
    import matplotlib.pyplot as plt
    cm = plt.get_cmap('jet')
    outp = np.array(outp.cpu().transpose(1,2).transpose(3,2)) # Batch, Height, Width, Channel
    outp_d_ini = (((outp[:,:,:,3])/scale_d))
    outp_d_res = (((outp[:,:,:,4]-5)*1.922/scale_d))
    outp_d = outp_d_ini-outp_d_res
    outp_d = np.where(outp_d<0,0,outp_d)
    img = (((2**16-1)*outp_d)).astype(np.uint16)
    #img = (255*cm(outp[:,:,:,0]/scale_d)).astype(np.uint8)[0,:,:,:3]
    im = Image.fromarray(img[0]).convert('I')
    #im = Image.fromarray(img)
    im.save(path)
