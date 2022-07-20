import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from custom_layers import *
import copy


def conv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False, In=False, wn=False, pixel=False, gdrop=True, circular=True, only=False):
    if circular:
        layers.append( ERP_padding(pad) )
        pad = 0

    if gdrop:       layers.append(generalized_drop_out(mode='prop', strength=0.0))
    if wn:          layers.append(equalized_conv2d(c_in, c_out, k_size, stride, pad, initializer='kaiming'))
    else:           layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if not only:
        if leaky:   layers.append(nn.LeakyReLU(0.2))
        else:       layers.append(nn.ReLU())
        if bn:      layers.append(nn.BatchNorm2d(c_out))
        if In:      layers.append(nn.InstanceNorm2d(c_out))
        if pixel:   layers.append(pixelwise_norm_layer())
    return layers

def linear(layers, c_in, c_out, sig=True, wn=False):
    layers.append(Flatten())
    if wn:      layers.append(equalized_linear(c_in, c_out, initializer='kaiming'))
    else:       layers.append(nn.Linear(c_in, c_out)) 
    if sig:     layers.append(nn.Sigmoid())
    return layers
    

def get_module_names(model):
    names = []
    for key, val in model.state_dict().items():
        name = key.split('.')[0]
        if not name in names:
            names.append(name)
    return names


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.flag_bn = config.flag_bn
        self.flag_in = config.flag_in
        self.flag_pixelwise = config.flag_pixelwise
        self.flag_wn = config.flag_wn
        self.flag_leaky = config.flag_leaky
        self.flag_tanh = config.flag_tanh
        self.flag_sigmoid_depth = config.flag_sigmoid_depth
        self.flag_norm_latent = config.flag_norm_latent
        self.nc = len(config.input_mode)
        self.outnc = config.outnc
        self.ngf = config.ngf
        self.max_ngf = config.max_ngf
        self.min_ngf = self.ngf
        self.circular = config.flag_circular_pad

        # input branches.
        self.input_rgb = self.input_branch(3)
        if "DL" in self.config.input_mode:
            self.input_d = self.input_branch(2)
        elif "L" in self.config.input_mode or "D" in self.config.input_mode:
            self.input_d = self.input_branch(1)

        # fused data processing layer: U-Net based network.
        self.down1 = self.downsample_block(64)
        self.down1_d = self.downsample_block(64)

        self.down2 = self.downsample_same_block(128)
        self.down2_d = self.downsample_same_block(128)

        self.down3 = self.downsample_same_block(256)
        self.down3_d = self.downsample_same_block(256)

        self.down4 = self.downsample_same_block(256)
        self.down4_d = self.downsample_same_block(256)

        self.mid = self.mid_block(512)
        self.mid_d = self.mid_block(512)

        self.up4 = self.upsample_block(512)
        self.up4_d = self.upsample_block(512)

        self.up3 = self.upsample_block(256)
        self.up3_d = self.upsample_block(256)


        self.up2 = self.upsample_block(256)
        self.up2_d = self.upsample_block(256)

        self.up1 = self.upsample_block(128)
        self.up1_d = self.upsample_block(128)


        # output branches.
        self.output_rgb = self.output_block(3)
        self.output_d_res= self.output_block(1)
        self.output_d_ini = self.output_block(1)

    def input_branch(self, nc):
        layers = []
        ndim = self.ngf

        layers = conv(layers, nc, ndim, 7, 1, 3, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)

        # downsample once.
        if "L" in self.config.input_mode or "D" in self.config.input_mode:
            ndim = min(self.max_ngf, self.ngf * 2)
            layers = conv(layers, ndim//2, ndim, 4, 2, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        else:
            ndim = min(self.max_ngf, self.ngf * 4)
            layers = conv(layers, ndim//4, ndim, 4, 2, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        
        return  nn.Sequential(*layers)



    def downsample_block(self,ngf):
        
        ndim = min(self.max_ngf, ngf * 2)

        layers = []
        #layers.append(nn.Upsample(scale_factor=2, mode='nearest'))       # scale up by factor of 2.0
        
        layers = conv(layers, ngf, ndim, 4, 2, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)

        
        return  nn.Sequential(*layers)
    
    def downsample_same_block(self, ndim):
        

        layers = []
        #layers.append(nn.Upsample(scale_factor=2, mode='nearest'))       # scale up by factor of 2.0
        
        layers = conv(layers, ndim, ndim, 4, 2, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)

        self.ngf = self.ngf * 2
        
        return  nn.Sequential(*layers)
        

    def mid_block(self,ndim):
        
        layers = []
        
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim//2, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        return  nn.Sequential(*layers)

    def upsample_block(self,ngf):
        
        ndim = ngf//2

        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))       # scale up by factor of 2.0
        
        layers = conv(layers, ngf, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)

        
        return  nn.Sequential(*layers)

    def upsample_same_block(self,ndim):
        
        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))       # scale up by factor of 2.0
        
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)

        
        return  nn.Sequential(*layers)

    def output_block(self, outnc):

        ndim = 32

        layers = []
        # upsample once.
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))       # scale up by factor of 2.0
        layers = conv(layers, 64, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)
        layers = conv(layers, ndim, ndim, 3, 1, 1, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, circular=self.circular)

        # map to the target domain.
        layers = conv(layers, ndim, outnc, 1, 1, 0, self.flag_leaky, self.flag_bn, self.flag_in, self.flag_wn, self.flag_pixelwise, only=True, circular=self.circular)
        
        return nn.Sequential(*layers)

    def freeze_layers(self):
        # let's freeze pretrained blocks. (Found freezing layers not helpful, so did not use this func.)
        print('freeze pretrained weights ... ')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        #x_ = self.first_layer(x)
        rgb = x[:,:3]
        rgb = self.input_rgb(rgb)
        
        if "D" in self.config.input_mode or "L" in self.config.input_mode:
            d = x[:,3:]
            d = self.input_d(d)

        x1 = self.down1(rgb)
        x1_d = self.down1_d(d)

        x2 = self.down2(x1)
        x2_d = self.down2_d(x1_d)
        
        x2_cat = torch.cat( [x2, x2_d], dim=1 )

        x3 = self.down3(x2_cat)
        x3_d = self.down3_d(x2_cat)
        
        x4 = self.down4(x3)
        x4_d = self.down4_d(x3_d)
            
        x_ = torch.cat( [x4, x4_d], dim=1 )

        xm_ = self.mid(x_)
        xm_d = self.mid_d(x_)

        xm_cat = torch.cat( [xm_, xm_d], dim=1 )

        x4_ = self.up4(xm_cat + x_)
        x4_d_ = self.up4_d(xm_cat + x_)
        
        x3_ = self.up3(x4_ + x3)
        x3_d_ = self.up3_d(x4_d_ + x3_d)

        x3_cat = torch.cat( [x3_, x3_d_], dim=1 )

        x2_ = self.up2(x3_cat + x2_cat)
        x2_d_ = self.up2_d(x3_cat + x2_cat)

        x1_ = self.up1(x2_ + x1)
        x1_d_ = self.up1_d(x2_d_ + x1_d)


        d_res = self.output_d_res(x1_d_ + d)
        d_ini = self.output_d_ini(x1_d_ + d)
        rgb = self.output_rgb(x1_ + rgb)

        
        rgb = torch.tanh(rgb)
        
        d_res = torch.sigmoid(d_res) * self.config.scale_d / 9.362
        d_ini= torch.sigmoid(d_ini) * self.config.scale_d / 9.362
        
        
        out = torch.cat([rgb,d_ini,d_res], dim=1)
     

        return out


def conv_layer(c_in, c_out, k_size, stride=1, pad=0, leaky=False, norm=False, circular=True):
    layers = []
    if circular:
        layers.append( ERP_padding(pad) )
        pad = 0 
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    #layers.append(equalized_conv2d(c_in, c_out, k_size, stride, pad))
    if norm:    layers.append(nn.InstanceNorm2d(c_out))    
    else:       pass
    if leaky:   layers.append(nn.LeakyReLU(0.2))
    else:       layers.append(nn.ReLU())
    return nn.Sequential( *layers )

class pix2pixHDDiscriminator(nn.Module):
    def __init__(self, config):
        super(pix2pixHDDiscriminator, self).__init__()
        self.config = config
        self.flag_leaky = True
        self.flag_tanh = config.flag_tanh
        self.flag_sigmoid_depth = config.flag_sigmoid_depth
        self.nc = len(config.input_mode)
        self.outnc = config.outnc # 3 for RGB only GAN.
        self.circular = config.flag_circular_pad
        self.conv1 = conv_layer(c_in=5, c_out=64, k_size=4, stride=2, pad=1, leaky=self.flag_leaky, norm=False, circular=self.circular) # first layer does not use instance normalization.
        self.conv2 = conv_layer(c_in=64, c_out=128, k_size=4, stride=2, pad=1, leaky=self.flag_leaky, norm=True, circular=self.circular)
        self.conv3 = conv_layer(c_in=128, c_out=256, k_size=4, stride=2, pad=1, leaky=self.flag_leaky, norm=True, circular=self.circular)
        self.conv4 = conv_layer(c_in=256, c_out=512, k_size=4, stride=2, pad=1, leaky=self.flag_leaky, norm=True, circular=self.circular)
        self.conv5 = nn.Conv2d(512, 1, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = self.conv5(x4)

        if self.config.flag_sigmoid:
            x = torch.sigmoid(x) # do not use sigmoid, because we are using LSGAN.
        return x1, x2, x3, x4, x.mean(-2).mean(-1)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, config):
        super(MultiScaleDiscriminator, self).__init__()
        self.config = config
        self.D0 = pix2pixHDDiscriminator(config)
        self.D1 = pix2pixHDDiscriminator(config)
        self.D2 = pix2pixHDDiscriminator(config)
    
    def forward(self, x):
        # zero_d = torch.tensor([1,1,1,0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).expand_as(x)
        # x = x * zero_d # RGB only GAN loss.

        f01, f02, f03, f04, score0 = self.D0(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        f11, f12, f13, f14, score1 = self.D1(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        f21, f22, f23, f24, score2 = self.D2(x)
        features = [f01, f02, f03, f04, f11, f12, f13, f14, f21, f22, f23, f24]
        return features, torch.cat([score0 , score1 , score2], dim = -1)
