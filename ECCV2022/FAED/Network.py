import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.upsampling = nn.functional.interpolate

        self.upconv1 = ConvBlock(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding = 1) # 이건 안 쓰인다.

        self.channelAttnetion_rgb = torch.nn.Parameter(torch.randn(128,1,1)) # 이건 안 쓰인다.
        self.channelAttnetion_d = torch.nn.Parameter(torch.randn(128,1,1)) # 이건 안 쓰인다.

        self.upconv2_rgb = ConvBlock(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1) 
        self.upconv3_rgb = ConvBlock(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1) 
        self.upconv4_rgb = ConvBlock(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1) 
        self.upconv5_rgb = ConvBlock(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1) 

        self.upconv2_d = ConvBlock(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1) 
        self.upconv3_d = ConvBlock(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1) 
        self.upconv4_d = ConvBlock(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1) 
        self.upconv5_d = ConvBlock(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1) 

        self.outconv_rgb = Conv2d(in_channels = 32, out_channels = 3 , kernel_size = 9, stride = 1, padding = 4) 
        self.outconv_d = Conv2d(in_channels = 32, out_channels = 1 , kernel_size = 9, stride = 1, padding = 4) 

        self.upres2_rgb = ResBlock(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.upres3_rgb = ResBlock(in_channels = 128, out_channels = 128, kernel_size = 5, padding = 2) 
        self.upres4_rgb = ResBlock(in_channels = 64, out_channels = 64, kernel_size = 7, padding = 3) 
        self.upres5_rgb = ResBlock(in_channels = 32, out_channels = 32, kernel_size = 9, padding = 4) 
        
        self.upres2_d = ResBlock(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.upres3_d = ResBlock(in_channels = 128, out_channels = 128, kernel_size = 5, padding = 2) 
        self.upres4_d = ResBlock(in_channels = 64, out_channels = 64, kernel_size = 7, padding = 3) 
        self.upres5_d = ResBlock(in_channels = 32, out_channels = 32, kernel_size = 9, padding = 4) 

    def forward(self, x):

        #x = self.upconv1(x)
        x = self.upsampling(x, scale_factor = 2 ,mode = 'bilinear', align_corners = False)    
        rgb = x[:,:128]    
        d = x[:,128:]

        #rgb = torch.sigmoid(self.channelAttnetion_rgb).expand_as(x) * x
        #rgb = self.upconv2_rgb(x)
        rgb = self.upconv2_rgb(rgb)
        rgb = self.upres2_rgb(rgb)
        rgb = self.upsampling(rgb, scale_factor = 2 ,mode = 'bilinear', align_corners = False)
        rgb = self.upconv3_rgb(rgb)
        rgb = self.upres3_rgb(rgb)
        rgb = self.upsampling(rgb, scale_factor = 2 ,mode = 'bilinear', align_corners = False)
        rgb = self.upconv4_rgb(rgb)
        rgb = self.upres4_rgb(rgb)
        rgb = self.upsampling(rgb, scale_factor = 2 ,mode = 'bilinear', align_corners = False)
        rgb = self.upconv5_rgb(rgb)
        rgb = self.upres5_rgb(rgb)
        rgb = self.upsampling(rgb, scale_factor = 2 ,mode = 'bilinear', align_corners = False)
        rgb = self.outconv_rgb(rgb)

        #d = torch.sigmoid(self.channelAttnetion_d).expand_as(x) * x
        #d = self.upconv2_d(x)
        d = self.upconv2_d(d)
        d = self.upres2_d(d)
        d = self.upsampling(d, scale_factor = 2 ,mode = 'bilinear', align_corners = False)
        d = self.upconv3_d(d)
        d = self.upres3_d(d)
        d = self.upsampling(d, scale_factor = 2 ,mode = 'bilinear', align_corners = False)
        d = self.upconv4_d(d)
        d = self.upres4_d(d)
        d = self.upsampling(d, scale_factor = 2 ,mode = 'bilinear', align_corners = False)
        d = self.upconv5_d(d)
        d = self.upres5_d(d)
        d = self.upsampling(d, scale_factor = 2 ,mode = 'bilinear', align_corners = False)
        d = self.outconv_d(d)

        x = torch.cat([rgb,d],dim=1)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.downconv1_rgb = Conv2d(in_channels = 3, out_channels = 32, kernel_size = 9, stride = 1, padding = 4)
        self.downconv1_d = Conv2d(in_channels = 1, out_channels = 32, kernel_size = 9, stride = 1, padding = 4)
        
        self.downconv2_rgb = ConvBlock(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 1) 
        self.downconv2_d = ConvBlock(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 1) 

        self.downconv3_rgb = ConvBlock(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.downconv3_d = ConvBlock(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)

        self.downconv4_rgb = ConvBlock(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.downconv4_d = ConvBlock(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)

        self.downconv5_rgb = ConvBlock(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.downconv5_d = ConvBlock(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)

        self.downconv6_rgb = ConvBlock(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        self.downconv6_d = ConvBlock(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)

        self.downres1_rgb = ResBlock(in_channels = 32, out_channels = 32, kernel_size = 9, padding = 4)
        self.downres1_d = ResBlock(in_channels = 32, out_channels = 32, kernel_size = 9, padding = 4)

        self.downres2_rgb = ResBlock(in_channels = 64, out_channels = 64, kernel_size = 7, padding = 3)
        self.downres2_d = ResBlock(in_channels = 64, out_channels = 64, kernel_size = 7, padding = 3)

        self.downres3_rgb = ResBlock(in_channels = 128, out_channels = 128, kernel_size = 5, padding = 2)
        self.downres3_d = ResBlock(in_channels = 128, out_channels = 128, kernel_size = 5, padding = 2)

        self.downres4_rgb = ResBlock(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.downres4_d = ResBlock(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)

        self.downres5_rgb = ResBlock(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.downres5_d = ResBlock(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)

        self.fuse = Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1) 

    def forward(self, x):
        rgb = x[:,:3]
        rgb = self.downconv1_rgb(rgb)
        rgb = self.downres1_rgb(rgb)
        rgb = self.downconv2_rgb(rgb)
        rgb = self.downres2_rgb(rgb)
        rgb = self.downconv3_rgb(rgb)
        rgb = self.downres3_rgb(rgb)
        rgb = self.downconv4_rgb(rgb)
        rgb = self.downres4_rgb(rgb)
        rgb = self.downconv5_rgb(rgb)
        rgb = self.downres5_rgb(rgb)
        rgb = self.downconv6_rgb(rgb)

        d = x[:,3:]
        d = self.downconv1_d(d)
        d = self.downres1_d(d)
        d = self.downconv2_d(d)
        d = self.downres2_d(d)
        d = self.downconv3_d(d)
        d = self.downres3_d(d)
        d = self.downconv4_d(d)
        d = self.downres4_d(d)
        d = self.downconv5_d(d)
        d = self.downres5_d(d)
        d = self.downconv6_d(d)

        x1 = torch.cat([rgb,d], dim=1)

        x = self.fuse(x1)

        return x ,x1

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        x, _ = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    net = Autoencoder()
    net = net.cuda()
    rgb = torch.rand(1,3,512,1024).cuda()
    d = torch.rand(1,1,512,1024).cuda()
    x = torch.cat([rgb,d],dim=1)
    x= net(x)
    print(np.shape(x))
