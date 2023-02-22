import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import copy
from torch.nn.init import kaiming_normal, calculate_gain

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class pixelwise_norm_layer(nn.Module):
    def __init__(self):
        super(pixelwise_norm_layer, self).__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x / (torch.mean(x**2, dim=1, keepdim=True) + self.eps) ** 0.5

# for equaliaeed-learning rate.
class equalized_conv2d(nn.Module):
    #def __init__(self, c_in, c_out, k_size, stride, pad, initializer='kaiming', bias=False):
    def __init__(self, c_in, c_out, k_size, stride, pad, initializer=None, bias=False):
        super(equalized_conv2d, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)
        if initializer == 'kaiming':    kaiming_normal(self.conv.weight, a=calculate_gain('conv2d'))
        elif initializer == 'xavier':   xavier_normal(self.conv.weight)
        
        #conv_w = self.conv.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        fan_in = c_in*k_size*k_size
        self.scale = 2/np.sqrt(fan_in)
        #self.scale = (torch.mean(self.conv.weight.data ** 2)) ** 0.5
        self.conv.weight.data.copy_(self.conv.weight.data/self.scale)

    def forward(self, x):
        x = self.conv(x*self.scale)
        return x + self.bias.view(1,-1,1,1).expand_as(x)

class ERP_padding(nn.Module):
    def __init__(self, pad):
        super(ERP_padding, self).__init__()
        self.pad = pad
    def forward(self, x):
        #x = torch.nn.functional.pad(x, (0, 0, self.pad, self.pad), 'constant', 0)
        #x = torch.nn.functional.pad(x, (self.pad, self.pad, 0, 0), 'circular')
        if self.pad == 0:
            return x
        x = torch.nn.functional.pad(x, (self.pad, self.pad, self.pad, self.pad), 'constant', 0)        
        x[:,:,:,0:self.pad] = x[:,:,:,-2*self.pad:-self.pad]
        x[:,:,:,-self.pad:] = x[:,:,:,self.pad:2*self.pad]
        
        return x

class equalized_linear(nn.Module):
    def __init__(self, c_in, c_out, initializer='kaiming'):
        super(equalized_linear, self).__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        if initializer == 'kaiming':    kaiming_normal(self.linear.weight, a=calculate_gain('linear'))
        elif initializer == 'xavier':   torch.nn.init.xavier_normal(self.linear.weight)
        
        linear_w = self.linear.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = (torch.mean(self.linear.weight.data ** 2)) ** 0.5
        self.linear.weight.data.copy_(self.linear.weight.data/self.scale)
        
    def forward(self, x):
        x = self.linear(x.mul(self.scale))
        return x + self.bias.view(1,-1).expand_as(x)


# ref: https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/models/base_model.py
class generalized_drop_out(nn.Module):
    def __init__(self, mode='mul', strength=0.4, axes=(0,1), normalize=False):
        super(generalized_drop_out, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['mul', 'drop', 'prop'], 'Invalid GDropLayer mode'%mode
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)
        self.normalize = normalize
        self.gain = None

    def forward(self, x, deterministic=False):
        if deterministic or not self.strength:
            return x

        rnd_shape = [s if axis in self.axes else 1 for axis, s in enumerate(x.size())]  # [x.size(axis) for axis in self.axes]
        if self.mode == 'drop':
            p = 1 - self.strength
            rnd = np.random.binomial(1, p=p, size=rnd_shape) / p
        elif self.mode == 'mul':
            rnd = (1 + self.strength) ** np.random.normal(size=rnd_shape)
        else:
            coef = self.strength * x.size(1) ** 0.5
            rnd = np.random.normal(size=rnd_shape) * coef + 1

        if self.normalize:
            rnd = rnd / np.linalg.norm(rnd, keepdims=True)
        rnd = Variable(torch.from_numpy(rnd).type(x.data.type()))
        if x.is_cuda:
            rnd = rnd.cuda()
        return x * rnd

    def __repr__(self):
        param_str = '(mode = %s, strength = %s, axes = %s, normalize = %s)' % (self.mode, self.strength, self.axes, self.normalize)
        return self.__class__.__name__ + param_str

def deconv(layers, c_in, c_out, k_size, stride=1, pad=0, leaky=True, bn=False, In=False, wn=False, pixel=False, only=False):
    # This function is not used. Instead, we used nn.Upsample.

    if wn:  layers.append(equalized_deconv2d(c_in, c_out, k_size, stride, pad)) # deconv 로 변경.
    else:   layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad)) # ConvTranspose로 변경.
    if not only:
        if leaky:   layers.append(nn.LeakyReLU(0.2))
        else:       layers.append(nn.ReLU())
        if bn:      layers.append(nn.BatchNorm2d(c_out))
        if In:      layers.append(nn.InstanceNorm2d(c_out))
        if pixel:   layers.append(pixelwise_norm_layer())
    return layers

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

class CircularPadding(nn.Module):
    def __init__(self, pad):
        super(CircularPadding, self).__init__()
        self.pad = pad
    def forward(self, x):
        if self.pad == 0:
            return x
        x = torch.nn.functional.pad(x, (self.pad, self.pad, self.pad, self.pad), 'constant', 0)        
        x[:,:,:,0:self.pad] = x[:,:,:,-2*self.pad:-self.pad]
        x[:,:,:,-self.pad:] = x[:,:,:,self.pad:2*self.pad]
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2d, self).__init__()
        self.pad = CircularPadding(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0)

    def forward(self, x):
        x = self.conv2d( self.pad (x) )
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResBlock,self).__init__()
        
        self.relu = nn.ReLU()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, stride = 1, padding = padding )
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size, stride = 1, padding = padding )
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):

        out = self.relu( self.batchnorm1( self.conv1(x) ) )
        out = self.batchnorm2( self.conv2(out) )    
        out += x    

        return out 

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding ):
        super(ConvBlock,self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, stride, padding )
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):

        x = self.relu( self.batchnorm1( self.conv1( x )))

        return x 


