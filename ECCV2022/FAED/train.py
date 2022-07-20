import DataLoader
import Network as net
from math import floor, ceil
import os, sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = 'RGBD quality Classifier.')

parser.add_argument('--epoch', type=int, default = 60, help = 'epochs to train.')
parser.add_argument('--name', type=str, required = True, help = 'name of the training.')
parser.add_argument('--validate_every', type=int, default = 10, help = 'validate every N iterations.')
parser.add_argument('--train_data_root', type=str, default = '../dataset/train', help = 'train data root')
parser.add_argument('--val_data_root', type=str, default = '../dataset/val', help = 'val data root')
parser.add_argument('--test_data_root', type=str, default = '../dataset/test', help = 'test data root')
parser.add_argument('--imsize', type=int, default = 512, help = 'image size')
parser.add_argument('--scale_d', type=int, default = 93.62, help = 'D_max, 7000 to 10.')
parser.add_argument('--batch_size', type=int, default = 4, help = 'batch size')
parser.add_argument('--lr', type=float, default = 1e-4, help = 'learning rate. default: 1e-4.')

parser.add_argument('--flag_zero_invalid_depth', type=bool, default = True, help = 'if True, zero the invalid depth values (max d values).')

parser.add_argument('--random_seed', type=int, default = 0, help = 'manual random seed.')
parser.add_argument('--resume', type=bool, default = False, help = 'resume flag.')
parser.add_argument('--checkpoint_path', type=str, default = '', help = 'resume checkpoint path.')

config = parser.parse_args()

class Trainer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')
        
        self.lr = config.lr
        self.lr_decay = 0.99
        self.globalIter = 0
        self.globalTick = 0
        self.epoch = config.epoch
        self.name = config.name
        self.loader = DataLoader.RGBQualityDataLoader(config)
                
        # network and cirterion
        self.net = net.AutoEncoder()
        self.loss = nn.L1Loss()

        # optimizer
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr)
        
        torch.cuda.manual_seed(config.random_seed)

        self.net = torch.nn.DataParallel(self.net).cuda()
 
        if self.config.resume:
            checkpoint_path = config.checkpoint_path
            print('[INFO] load checkpoint from ... {}'.format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            self.net.module.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
    def lr_scheduler(self, epoch):
        self.lr *= self.lr_decay
        return
        
    def train(self):
        
        for epoch in range(self.epoch):
            self.lr_scheduler(epoch)

            for iter in tqdm(range(0,ceil(len(self.loader)/(self.config.batch_size)))):
                self.globalIter = self.globalIter+1
                
                # zero gradients.
                self.net.zero_grad()

                # update discriminator.
                rgb, d = self.loader.get_batch(mode='train')
                gt = torch.cat([rgb,d],dim=1)

                output = self.net(gt)
                output_rgb = output[:,:3]
                output_d = output[:,3:]
                output_rgb = torch.tanh(output_rgb)
                output_d = torch.sigmoid(output_d)
                output_d = output_d*self.config.scale_d
                output = torch.cat([output_rgb,output_d], dim=1)
                loss = self.loss(output, gt)
                   
                loss.backward()
                self.optimizer.step()

                # logging.
                if self.globalIter%10 == 0:
                    log_msg = f' [Epoch:{epoch}][Iter:{self.globalIter}][err: {loss.item():.4f}][lr:{self.lr:.5f}]'
                    tqdm.write(log_msg)

                if self.globalIter%config.validate_every == 0:
                    self.save_image(gt, output, f'result/{self.config.name}.png')


            # save model.
            self.save_model('model', epoch)

    def save_model(self, path, epoch):
        if not os.path.exists(path):
            if os.name == 'nt':
                os.system('mkdir {}'.format(path.replace('/', '\\')))
            else:
                os.system('mkdir -p {}'.format(path))
        
        ndis = f'RGBQuality_{self.name}.pth.tar'
    
        save_path = os.path.join(path, ndis)
        
        state_dict = {
                'state_dict' : self.net.module.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
            }
        torch.save( state_dict , save_path)
        print('[INFO] Model Saved: {}'.format(save_path))

    def save_image(self, gt, output, path):
        if not os.path.exists('result'):
            if os.name == 'nt':
                os.system('mkdir {}'.format('result'.replace('/', '\\')))
            else:
                os.system('mkdir -p {}'.format('result'))
    
        scale_d = self.config.scale_d

        cm = plt.get_cmap('jet')

        rgb_input = gt[:,0:3]
        d_input = gt[:,3:]
        rgb_output = output[:,0:3]
        d_output = output[:,3:]

        rgb_input = np.array(rgb_input.cpu().transpose(1,2).transpose(3,2))
        rgb_input = (255*(rgb_input[:,:,:,:3]+1)/2).astype(np.uint8)
        rgb_output = np.array(rgb_output.detach().cpu().transpose(1,2).transpose(3,2))
        rgb_output = (255*(rgb_output[:,:,:,:3]+1)/2).astype(np.uint8)
        img_rgb = np.concatenate([rgb_input[0],rgb_output[0]],axis = -2)

        d_input = np.array(d_input.cpu().transpose(1,2).transpose(3,2))
        d_input = (((d_input[:,:,:,0])/scale_d))
        d_input_max = d_input.max(-1).max(-1)
        d_output = np.array(d_output.detach().cpu().transpose(1,2).transpose(3,2))
        d_output = (((d_output[:,:,:,0])/scale_d))
        d_output_max = d_output.max(-1).max(-1)
        d_max = np.concatenate([np.expand_dims(d_input_max,-1),np.expand_dims(d_output_max,-1)],axis=-1).max(-1)
        img_d = np.concatenate( [(255*cm(d_input[0]/d_max[0])[:,:,:3]).astype(np.uint8),(255*cm(d_output[0]/d_max[0])[:,:,:3]).astype(np.uint8)], axis=-2)
        img = np.concatenate([img_rgb,img_d], axis=-3)
                 
        im = Image.fromarray(img)
        im.save(path)

if __name__ == '__main__':
    print('----------------- configuration -----------------')
    for k, v in vars(config).items():
        print('  {}: {}'.format(k, v))
    print('-------------------------------------------------')
    torch.backends.cudnn.benchmark = True           # boost speed.
    trainer = Trainer(config)
    trainer.train()