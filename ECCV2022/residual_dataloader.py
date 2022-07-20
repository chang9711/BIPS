import os
from numpy.core.numeric import indices
import torch as torch
import numpy as np
from io import BytesIO
import scipy.misc
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
from PIL import Image
import math
import loss as loss
class dataloader:
    def __init__(self, config):

        print('[*] initializing data loader...')
        self.config = config
        self.root_train = config.train_data_root
        self.root_val = config.val_data_root
        self.root_test = config.test_data_root
        self.batchsize = config.batchsize
        self.imsize = config.imsize
        self.num_workers = 0 
        self.flag_rotate = config.flag_rotate
        self.rotate_val = 0
        self.flag_flip = 0
        if torch.cuda.is_available():
            self.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')

        input_modes = ["RGB","RGBD","RGBL","RGBDL"]
        self.input_mode = config.input_mode
        if self.input_mode not in input_modes:
            raise NotImplementedError(f"input_mode: {self.input_mode} is not implemented.")
        
        self.RGB_folder_train = self.root_train + "/RGB"
        RGB_list_train = os.listdir(self.RGB_folder_train)
        RGB_list_train.sort()
        self.len_data_train = len(RGB_list_train)
        self.RGB_list_train = RGB_list_train

        self.RGB_folder_val = self.root_val + "/RGB"
        RGB_list_val = os.listdir(self.RGB_folder_val)
        RGB_list_val.sort()
        self.len_data_val = len(RGB_list_val)
        self.RGB_list_val = RGB_list_val

        self.RGB_folder_test = self.root_test + "/RGB"
        RGB_list_test = os.listdir(self.RGB_folder_test)
        RGB_list_test.sort()
        self.len_data_test = len(RGB_list_test)
        self.RGB_list_test = RGB_list_test
        
        self.D_folder_train = self.root_train + "/D" # GT depth
        D_list_train = os.listdir(self.D_folder_train)
        D_list_train.sort()
        len_data_train = len(D_list_train)
        self.D_list_train = D_list_train
        assert len_data_train == self.len_data_train, f"number of images for rgb and depth are different (training): {self.len_data_train} for rgb and {len_data_train} for depth."


        self.D_folder_val = self.root_val + "/D"
        D_list_val = os.listdir(self.D_folder_val)
        D_list_val.sort()
        len_data_val = len(D_list_val)
        self.D_list_val = D_list_val
        assert len_data_val == self.len_data_val, f"number of images for rgb and depth are different (val): {self.len_data_val} for rgb and {len_data_val} for depth."
    
        
        self.D_folder_test = self.root_test + "/D"
        D_list_test = os.listdir(self.D_folder_test)
        D_list_test.sort()
        len_data_test = len(D_list_test)
        self.D_list_test = D_list_test
        assert len_data_test == self.len_data_test, f"number of images for rgb and depth are different (test): {self.len_data_test} for rgb and {len_data_test} for depth."
    
        self.D_ini_folder_train = self.root_train + "/D_ini"
        D_ini_list_train = os.listdir(self.D_ini_folder_train)
        D_ini_list_train.sort()
        self.len_data_train = len(D_ini_list_train)
        self.D_ini_list_train = D_ini_list_train

        self.D_ini_folder_val = self.root_val + "/D_ini"
        D_ini_list_val = os.listdir(self.D_ini_folder_val)
        D_ini_list_val.sort()
        self.len_data_val = len(D_ini_list_val)
        self.D_ini_list_val = D_ini_list_val

        self.D_ini_folder_test = self.root_test + "/D_ini"
        D_ini_list_test = os.listdir(self.D_ini_folder_test)
        D_ini_list_test.sort()
        self.len_data_test = len(D_ini_list_test)
        self.D_ini_list_test = D_ini_list_test
      
        print('[*] data loader initialized.')

    def read_input(self, mode='train'):
        modes = ['train', 'val']
        assert mode in modes, f"given mode: '{mode}' is not implemented in datalodaer.read_input()."

        if mode == 'train':
            indices = np.random.randint(self.len_data_train, size = self.batchsize)
            rgb_path = self.RGB_folder_train
            rgb_list = self.RGB_list_train
            d_path = self.D_folder_train
            d_list = self.D_list_train
            d_ini_path = self.D_ini_folder_train
            d_ini_list = self.D_ini_list_train

        elif mode == 'val':
            indices = np.random.randint(self.len_data_val, size = self.batchsize)
            rgb_path = self.RGB_folder_val
            rgb_list = self.RGB_list_val
            d_path = self.D_folder_val
            d_list = self.D_list_val
            d_ini_path = self.D_ini_folder_val
            d_ini_list = self.D_ini_list_val

        rgb = torch.zeros(self.config.batchsize, 3, self.imsize, 2*self.imsize)
        d = torch.zeros(self.config.batchsize, 1, self.imsize, 2*self.imsize)
        d_ini = torch.zeros(self.config.batchsize, 1, self.imsize, 2*self.imsize)
        d_res = torch.zeros(self.config.batchsize, 1, self.imsize, 2*self.imsize)
        
        for i, idx in enumerate(indices):
            # color image loading.
            rgb_ = Image.open(rgb_path + "/" + rgb_list[idx], 'r')
            rgb_ = np.asarray(rgb_.resize((2*self.imsize, self.imsize), Image.BILINEAR))
            rgb_ = rgb_[:,:,:3] # in case alpha channel exists.
            import pdb
            # color jitter
            if self.config.flag_color_jitter:
                cj = np.random.randn(3)*0.1 + 1
                rgb_ = np.clip(rgb_ * cj, 0, 255)

            # transform pixel space into [-1, 1]
            rgb_ = (rgb_/255) * 2 - 1

            # transform into pytorch tensor
            rgb_ = torch.tensor(rgb_).transpose(2,1).transpose(1,0)

            # flip the image
            self.flag_flip = np.random.randint(2) # outputs 0 or 1.
            if self.flag_flip:
                rgb_ = rgb_.flip([-1]) # rangom horizontal flip on-the-fly.

            # rotate the image
            if self.flag_rotate:
                self.rotate_val = np.random.randint(self.imsize*2)
                rgb_ = rgb_.roll(self.rotate_val,-1)

            # unsqueeze
            rgb_ =  rgb_.float()
            rgb[i] = rgb_

            d_ = Image.open(d_path + "/" + d_list[idx], 'r')
            d_ = np.asarray(d_.resize((2*self.imsize, self.imsize), Image.NEAREST))
               
            d_ini_ = Image.open(d_ini_path + "/" + d_ini_list[idx], 'r')
            d_ini_ = np.asarray(d_ini_.resize((2*self.imsize, self.imsize), Image.BILINEAR))
           
            d_res_ = d_ini_ - d_
            d_res_ = np.where(((d_res_<20)&(d_res_>-20)),0,d_res_)
            d_res_ = np.where(d_res_<-6727,-6727,d_res_)

            d_ = (d_ / (2**16-1)) * self.config.scale_d
                
            d_ini_ = (d_ini_ / (2**16-1)) * self.config.scale_d 
            d_res_ = (d_res_ / (2**16-1)) * self.config.scale_d /1.922 + 5
           
            #transform into pytorch tensor
            d_ = torch.tensor(d_).unsqueeze(0)
          

            # flip the image
            if self.flag_flip:
                d_ = d_.flip([-1]) # rangom horizontal flip on-the-fly.

            # rotate the image
            if self.flag_rotate:
                d_ = d_.roll(self.rotate_val,-1)
   
            # zero the invalid depth value
            if self.config.flag_zero_invalid_depth:
                d_max_mask = (d_ > 0.85*(self.config.scale_d)).float()
                d_ = d_ * (1-d_max_mask)

            d_ini_ = torch.tensor(d_ini_).unsqueeze(0)

            if self.flag_flip:
                d_ini_ = d_ini_.flip([-1]) # rangom horizontal flip on-the-fly.

            if self.flag_rotate:
                d_ini_ = d_ini_.roll(self.rotate_val,-1)
            
            d_res_ = torch.tensor(d_res_).unsqueeze(0)

            if self.flag_flip:
                d_res_ = d_res_.flip([-1]) # rangom horizontal flip on-the-fly.

            if self.flag_rotate:
                d_res_ = d_res_.roll(self.rotate_val,-1)

            d_ = d_.float()
            d_ini_ = d_ini_.float()
            d_res_ = d_res_.float()

            d[i] = d_
            d_ini[i] = d_ini_
            d_res[i] = d_res_
        # return
        return rgb, d, d_ini, d_res

    def get_batch(self, mode='train'):
        # read input
        rgb_gt, d_gt, d_ini_gt, d_res_gt = self.read_input(mode=mode)

        if self.use_cuda:
            rgb_gt = rgb_gt.cuda()
            d_gt = d_gt.cuda()
            d_ini_gt = d_ini_gt.cuda()
            d_res_gt = d_res_gt.cuda()
        return rgb_gt, d_gt, d_ini_gt, d_res_gt

    def read_input_test(self, idx):
        rgb_path = self.RGB_folder_test
        rgb_list = self.RGB_list_test
        d_path = self.D_folder_test
        d_list = self.D_list_test
        d_ini_path = self.D_ini_folder_test
        d_ini_list = self.D_ini_list_test

        rgb = torch.zeros(1, 3, self.imsize, 2*self.imsize)
        d = torch.zeros(1, 1, self.imsize, 2*self.imsize)
        d_ini = torch.zeros(1, 1, self.imsize, 2*self.imsize)
        d_res = torch.zeros(self.config.batchsize, 1, self.imsize, 2*self.imsize)

        # color image loading.
        rgb_ = Image.open(rgb_path + "/" + rgb_list[idx], 'r')
        rgb_ = np.asarray(rgb_.resize((2*self.imsize, self.imsize), Image.BILINEAR))

        rgb_ = (rgb_[:,:,:3])

        # transform pixel space into [-1, 1]
        rgb_ = (rgb_/255) * 2 - 1

        # transform into pytorch tensor
        rgb_ = torch.tensor(rgb_).transpose(2,1).transpose(1,0)

        # unsqueeze
        rgb_ =  rgb_.float()
        rgb[0] = rgb_

        # depth image loading.
        
        d_ = Image.open(d_path + "/" + d_list[idx], 'r')
        d_ = np.asarray(d_.resize((2*self.imsize, self.imsize), Image.NEAREST))
        
        d_ini_ = Image.open(d_ini_path + "/" + d_ini_list[idx], 'r')
        d_ini_ = np.asarray(d_ini_.resize((2*self.imsize, self.imsize), Image.NEAREST))

        d_res_ = d_ini_ - d_
        d_res_ = np.where(((d_res_<20)&(d_res_>-20)),0,d_res_)
        d_res_ = np.where(d_res_<-6727,-6727,d_res_)

        d_ = (d_ / (2**16-1)) * self.config.scale_d
            
        d_ini_ = (d_ini_ / (2**16-1)) * self.config.scale_d 
        d_res_ = (d_res_ / (2**16-1)) * self.config.scale_d /1.922 + 5        
        # transform depth space into [0, scale_d] (depth data is in uint16)
       
      
        #transform into pytorch tensor
        d_ = torch.tensor(d_).unsqueeze(0)
        d_ini_ = torch.tensor(d_ini_).unsqueeze(0)
        d_res_ = torch.tensor(d_res_).unsqueeze(0)

           
        
        # zero the invalid depth value
        if self.config.flag_zero_invalid_depth:
            d_max_mask = (d_ > 0.85*(self.config.scale_d)).float()
            d_ = d_ * (1-d_max_mask)
        # float
        d_ = d_.float()
        d_ini_ = d_ini_.float()
        d_res_ = d_res_.float()

        d[0] = d_
        d_ini[0] = d_ini_
        d_res[0] = d_res_
        # return
        return rgb, d, d_ini, d_res

    def get_batch_test(self, idx):
        # read input
        rgb_gt, d_gt, d_ini_gt, d_res_gt = self.read_input_test(idx)
        
        if self.use_cuda:
            rgb_gt = rgb_gt.cuda()
            d_gt = d_gt.cuda()
            d_ini_gt = d_ini_gt.cuda()
            d_res_gt = d_res_gt.cuda()
        return rgb_gt, d_gt, d_ini_gt, d_res_gt

    def __len__(self):
        return self.len_data_train

if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    depth = Image.open("", 'r')
    depth = np.asarray(depth)
    depth = (depth / (2**16-1)) * 10
    import torch
    import matplotlib.pyplot as plt
    import time
    from config import config
    cm = plt.get_cmap('jet')
    depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0).cuda().float()
    obj = dataloader(config)
    start = time.time()
    depth_diff = obj.simulate_occlusion(depth)
    depth_diff = obj.global_lidar_mask
    
    end = time.time()
    print(end - start)

    depth_diff = np.array(depth_diff.cpu().transpose(1,2).transpose(3,2).squeeze())
    depth_diff = (cm((depth_diff)/10)*255).astype(np.uint8)
    print(depth_diff.shape)
    im = Image.fromarray(depth_diff)
    im.save("./temp.png")

