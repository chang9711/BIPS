import os
import torch as torch
import numpy as np
from io import BytesIO
import scipy.misc
import torch    
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
from PIL import Image
import math


class RGBQualityDataLoader:
    def __init__(self, config):

        print('[INFO] initializing data loader...')
        self.config = config
        self.root_train = config.train_data_root
        self.root_val = config.val_data_root
        self.root_test = config.test_data_root
        self.batch_size = config.batch_size 
        self.imsize = config.imsize
        self.num_workers = 0
        if torch.cuda.is_available():
            self.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')
        self.test_idx = 0

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

        print('[INFO] data loader initialized.')

    def read_input(self, mode='train'):
        modes = ['train', 'val', 'test']
        assert mode in modes, f"given mode: '{mode}' is not implemented in datalodaer.read_input()."

        if mode == 'train':
            indices = np.random.randint(self.len_data_train, size = self.batch_size) 
            rgb_path = self.RGB_folder_train
            rgb_list = self.RGB_list_train
            d_path = self.D_folder_train
            d_list = self.D_list_train

        elif mode == 'val':
            indices = np.random.randint(self.len_data_val, size = self.batch_size) 
            rgb_path = self.RGB_folder_val
            rgb_list = self.RGB_list_val
            d_path = self.D_folder_val
            d_list = self.D_list_val

        elif mode == 'test':
            indices = [ self.test_idx ]
            #indices = np.random.randint(self.len_data_test, size = self.batch_size)
            rgb_path = self.RGB_folder_test
            rgb_list = self.RGB_list_test
            d_path = self.D_folder_test
            d_list = self.D_list_test

        rgb = torch.zeros(self.batch_size, 3, self.imsize, 2*self.imsize) 
        d = torch.zeros(self.batch_size, 1, self.imsize, 2*self.imsize) 
        
        for i, idx in enumerate(indices): # load two positive cases.
            # color image loading.
            rgb_ = Image.open(rgb_path + "/" + rgb_list[idx], 'r')
            rgb_ = np.asarray(rgb_.resize((2*self.imsize, self.imsize), Image.BILINEAR))
            rgb_ = rgb_[:,:,:3] # in case alpha channel exists.
            # transform pixel space into [-1, 1]
            rgb_ = (rgb_/255) * 2 - 1
    
           
            # transform into pytorch tensor
            rgb_ = torch.tensor(rgb_).transpose(2,1).transpose(1,0)
            # flip the image
            self.flag_flip = np.random.randint(2) # outputs 0 or 1.
            if self.flag_flip and mode != 'test':
                rgb_ = rgb_.flip([-1]) # rangom horizontal flip on-the-fly.

            # unsqueeze
            rgb_ =  rgb_.float()
            rgb[i] = rgb_
            
            d_ = Image.open(d_path + "/" + d_list[idx], 'r')
            d_ = np.asarray(d_.resize((2*self.imsize, self.imsize), Image.NEAREST))
            scale_d = self.config.scale_d
     
            # transform depth space into [0, scale_d] (depth data is in uint16)
            d_ = (d_ / (2**16-1)) * scale_d

            #transform into pytorch tensor
            d_ = torch.tensor(d_).unsqueeze(0)

            # flip the image
            if self.flag_flip:
                d_ = d_.flip([-1]) # rangom horizontal flip on-the-fly.

            # zero the invalid depth value
            if self.config.flag_zero_invalid_depth:
                d_max_mask = (d_ > 0.85*(scale_d)).float()
                d_ = d_ * (1-d_max_mask)

            d_ = d_.float()
            d[i] = d_

            if mode == 'test':
                self.test_idx += 1
                return rgb[:1], d[:1]

        return rgb, d

    def get_batch(self, mode='train'):
        # read input
        rgb_gt, d_gt = self.read_input(mode=mode)

        if self.use_cuda:
            rgb_gt = rgb_gt.cuda()
            d_gt = d_gt.cuda()
        return rgb_gt, d_gt

    def __len__(self):
        return self.len_data_train


