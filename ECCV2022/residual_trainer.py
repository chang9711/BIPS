import residual_dataloader as DL
from config import config
import residual_network as net
from math import floor, ceil
import os, sys

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm
import utils as utils
import numpy as np
import mask as mask

class trainer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')
        
        self.optimizer = config.optimizer

        self.resl = 2          
        self.lr = config.lr
        self.globalIter = 0
        self.nc = len(config.input_mode)
        self.epoch = config.epoch
        self.lambda_l1_all = config.lambda_l1_all
        self.lambda_gan = config.lambda_gan
        self.img_name = config.name
        self.global_img_grid = mask.get_global_img_grid(config)

        # network and cirterion
        self.G = net.Generator(config)
        self.D = net.MultiScaleDiscriminator(config)
    
        self.mse = torch.nn.MSELoss()

        if self.use_cuda:
            self.mse = self.mse.cuda()
            torch.cuda.manual_seed(config.random_seed)
            self.G = torch.nn.DataParallel(self.G).cuda()
            self.D = torch.nn.DataParallel(self.D).cuda()
     
        # define tensors, ship model to cuda, and get dataloader.
        self.renew_everything()
 
        if self.config.resume:
            checkpoint_path = config.G_checkpoint_path
            print('[*] load checkpoint from ... {}'.format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            self.G.module.load_state_dict(checkpoint['state_dict'])
            self.opt_g.load_state_dict(checkpoint['optimizer'])

            checkpoint_path = config.D_checkpoint_path
            print('[*] load checkpoint from ... {}'.format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            self.D.module.load_state_dict(checkpoint['state_dict'])
            self.opt_d.load_state_dict(checkpoint['optimizer'])
        
            
    def renew_everything(self):
        # renew dataloader.
        self.loader = DL.dataloader(config)
        
        # define tensors
        self.z = torch.FloatTensor(self.loader.batchsize, self.nc, self.loader.imsize, 2*self.loader.imsize)
        self.x = torch.FloatTensor(self.loader.batchsize, self.nc, self.loader.imsize, 2*self.loader.imsize)
        self.x_tilde = torch.FloatTensor(self.loader.batchsize, self.nc, self.loader.imsize, 2*self.loader.imsize)
        self.real_label = torch.FloatTensor(self.loader.batchsize, 3).fill_(1) # 3 channel, because of multi-scale discriminator. 
        self.fake_label = torch.FloatTensor(self.loader.batchsize, 3).fill_(0)
		
        # enable cuda
        if self.use_cuda:
            self.z = self.z.cuda()
            self.x = self.x.cuda()
            self.x_tilde = self.x.cuda()
            self.real_label = self.real_label.cuda()
            self.fake_label = self.fake_label.cuda()
            torch.cuda.manual_seed(config.random_seed)

        # wrapping autograd Variable.
        self.x = Variable(self.x)
        self.x_tilde = Variable(self.x_tilde)
        self.real_label = Variable(self.real_label)
        self.fake_label = Variable(self.fake_label)
        
        # ship new model to cuda.
        if self.use_cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
        
        # optimizer
        betas = (self.config.beta1, self.config.beta2)
        if self.optimizer == 'adam':
            self.opt_g = Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.lr, betas=betas, weight_decay=0.0)
            self.opt_d = Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.lr, betas=betas, weight_decay=0.0)
    
    def lr_scheduler(self, epoch):
        if epoch < self.config.epoch // 2:
            return

        else:
            delta = self.config.lr / (self.config.epoch // 2)
            self.lr = max(self.lr - delta, delta)
            
            for param_group in self.opt_g.param_groups:
                param_group['lr'] = self.lr
            for param_group in self.opt_d.param_groups:
                param_group['lr'] = self.lr
        
    def train(self):
        
        for epoch in range(self.epoch):
            self.lr_scheduler(epoch)
                                
            # save model.
            self.snapshot('repo/model', epoch)
            for iter in tqdm(range(0,ceil(len(self.loader)/self.loader.batchsize))):
                self.globalIter = self.globalIter+1
                
                # zero gradients.
                self.G.zero_grad()
                self.D.zero_grad()

                # update discriminator.
                rgb_gt, d_gt, d_ini_gt, d_res_gt = self.loader.get_batch(mode='train')

                
                gt = torch.cat( [rgb_gt, d_ini_gt, d_res_gt], dim=1 )


                input = torch.cat( [rgb_gt, d_gt], dim=1 )

                input_mask = mask.generate_input_mask_batch(self.config, self.global_img_grid)
                self.z = input_mask * input
                self.x.data = gt

                self.x_tilde = self.G(self.z) 

                _, self.fx = self.D(self.x)
                _, self.fx_tilde = self.D(self.x_tilde.detach())

            
                loss_d = self.mse(self.fx.squeeze(), self.real_label) + \
                                  self.mse(self.fx_tilde.squeeze(), self.fake_label)

                loss_d.backward()
                self.opt_d.step()

                # update generator.
                features, self.fx = self.D(self.x)
                features_tilde, fx_tilde = self.D(self.x_tilde)


                loss_gan = self.mse(fx_tilde.squeeze(), self.real_label.detach())

                loss_l1_all = ((self.x_tilde - self.x).abs()).mean() 
                
                if self.config.mask_valid_pixel:
                    valid_mask = 1 - ((d_gt >= (self.config.scale_d - 1e-9)) + (d_gt <= 1e-9)).byte()
                else:
                    valid_mask = torch.ones_like(d_gt)
               

                loss_g = self.lambda_gan*loss_gan + self.lambda_l1_all*loss_l1_all 
  
                loss_g.backward()
                self.opt_g.step()

                # logging.
                log_msg = f' [Epoch:{epoch}][Iter:{self.globalIter}] errD: {loss_d.item():.4f} | errG: {loss_g.item():.4f} | [lr:{self.lr:.5f}] | loss_l1_all: {loss_l1_all.item():.4f}'
                tqdm.write(log_msg)
                

                # save image grid.
                if self.globalIter%self.config.save_img_every == 0:

                    utils.mkdir('repo/train')
                    utils.mkdir('repo/val')
                   
                    utils.save_depth_res(self.z.data, self.x_tilde.data, input, self.x.data, self.config, f'repo/train/{self.img_name}.png')
                  
                    with torch.no_grad():
                        rgb_gt, d_gt, d_ini_gt ,d_res_gt = self.loader.get_batch(mode='val')
                        
                        gt = torch.cat( [rgb_gt, d_ini_gt, d_res_gt], dim=1 )

                        input = torch.cat( [rgb_gt, d_gt], dim = 1 )
                        input_mask = mask.generate_input_mask_batch(self.config, self.global_img_grid)
                        in_img =  input * input_mask
                        
                        x_test= self.G( in_img )
                                                
                        utils.save_depth_res(in_img.data, x_test.data, input.data, gt.data, self.config, f'repo/val/{self.img_name}.png')

                # validation.
                # # if self.globalIter%self.config.validate_every == 0:
                # #     all_mask = torch.ones_like(d_gt)
                # #     valid_mask = 1 - ((d_gt >= (self.config.scale_d-1e-9)) + (d_gt <= 1e-9)).byte()
                    
                    
                # #     log_msg = f' [Validation] valid_mae: {val_valid_mae.item():.4f} | valid_rmse: {val_valid_rmse.item():.4f} | lidar_fov_mae_valid: {val_lidar_fov_mae_valid:.4f} | lidar_fov_rmse_valid: {val_lidar_fov_rmse_valid:.4f}'
                # #     tqdm.write(log_msg)

             
    def get_state(self, target):
        if target == 'gen':
            state = {
                'resl' : self.resl,
                'state_dict' : self.G.module.state_dict(),
                'optimizer' : self.opt_g.state_dict(),
            }
            return state
        elif target == 'dis':
            state = {
                'resl' : self.resl,
                'state_dict' : self.D.module.state_dict(),
                'optimizer' : self.opt_d.state_dict(),
            }
            return state

    def snapshot(self, path, epoch):
        if not os.path.exists(path):
            if os.name == 'nt':
                os.system('mkdir {}'.format(path.replace('/', '\\')))
            else:
                os.system('mkdir -p {}'.format(path))
        # save every 100 tick if the network is in stab phase.
        ndis = f'dis_name:{self.img_name}.pth.tar'
        ngen = f'gen_name:{self.img_name}.pth.tar'
    
        save_path = os.path.join(path, ndis)
        #if not os.path.exists(save_path):
        torch.save(self.get_state('dis'), save_path)
        save_path = os.path.join(path, ngen)
        torch.save(self.get_state('gen'), save_path)
        print('[snapshot] model saved @ {}'.format(path))

if __name__ == '__main__':
    ## perform training.
    print('----------------- configuration -----------------')
    for k, v in vars(config).items():
        print('  {}: {}'.format(k, v))
    print('-------------------------------------------------')
    torch.backends.cudnn.benchmark = True           # boost speed.
    trainer = trainer(config)
    trainer.train()