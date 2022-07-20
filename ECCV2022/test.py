import residual_dataloader as DL
from config import config
import residual_network as network
from math import floor, ceil
import os, sys
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm
import utils as utils
import numpy as np
import loss as loss
import FAED.Network as AEnet
import torch
import random
import math
import numpy as np

flag_imsave = False
flag_stat = True

def generate_input_mask_batch(config, img_grid):
    mask = generate_input_mask(img_grid).unsqueeze(0)
    for batch in range(1,config.batchsize):
        mask = torch.cat([mask, generate_input_mask(img_grid).unsqueeze(0)], dim=0)
    return mask

def generate_input_mask(img_grid):
    
    # this function generates the final input 
    
    input_mode = random.random() # switch btw RGB, D, RGBD input modes. sample from 0~1 uniform distribution.
    #input_mode = 0.9 # RGBD case

    if input_mode < 1/3: # RGB only case
        input_mask = generate_rgb_only_mask(img_grid)
        return input_mask
        
    elif 1/3 <= input_mode < 2/3: # D only case
        d_mode = random.random() # switch btw nfov and lidar modes. sample from 0~1 uniform distribution.
        if d_mode < 1/2: # nfov D case
            input_mask = generate_d_only_nfov_mask(img_grid)
        else: # lidar D case
            input_mask = generate_d_only_lidar_mask(img_grid)
        return input_mask

    else: # RGBD case
        d_mode = random.random() # switch btw nfov and lidar modes. sample from 0~1 uniform distribution.
        #d_mode = 0.1 # RGBD cam
        if d_mode < 1/2:
            input_mask = generate_rgbd_nfov_mask(img_grid)
        else:
            input_mask = generate_rgbd_lidar_mask(img_grid)
        return input_mask

def generate_rgbd_lidar_mask(img_grid):
    lidar_mask = generate_LIDAR_mask(img_grid)
    nfov_mask = generate_nfov_mask(img_grid)

    nfov_mask = nfov_mask.unsqueeze(0)
    lidar_mask = lidar_mask.unsqueeze(0)
    mask = torch.cat( [nfov_mask, nfov_mask, nfov_mask, lidar_mask], dim=0 )
    return mask

def generate_d_only_lidar_mask(img_grid):
    lidar_mask = generate_LIDAR_mask(img_grid)

    lidar_mask = lidar_mask.unsqueeze(0)
    nfov_mask = torch.zeros_like(lidar_mask)
    mask = torch.cat( [nfov_mask, nfov_mask, nfov_mask, lidar_mask], dim=0 )
    return mask

def generate_rgbd_nfov_mask(img_grid):
    nfov_mask = generate_nfov_mask(img_grid)

    nfov_mask = nfov_mask.unsqueeze(0)
    mask = torch.cat( [nfov_mask, nfov_mask, nfov_mask, nfov_mask], dim=0 )
    return mask

def generate_rgb_only_mask(img_grid):
    nfov_mask = generate_nfov_mask(img_grid)

    nfov_mask = nfov_mask.unsqueeze(0)
    lidar_mask = torch.zeros_like(nfov_mask)
    mask = torch.cat( [nfov_mask, nfov_mask, nfov_mask, lidar_mask], dim=0 )
    return mask

def generate_d_only_nfov_mask(img_grid):
    nfov_mask = generate_nfov_mask(img_grid)

    nfov_mask = nfov_mask.unsqueeze(0)
    rgb_mask = torch.zeros_like(nfov_mask)
    mask = torch.cat( [rgb_mask, rgb_mask, rgb_mask, nfov_mask], dim=0 )
    return mask

def get_global_img_grid(config):
    # mesh grid generation.
    x = np.linspace(-math.pi , math.pi , 2*config.imsize)
    y = np.linspace(math.pi/2 , -math.pi/2 , config.imsize)
    theta, phi = np.meshgrid(x, y)
    theta = torch.tensor(theta).cuda()
    phi = torch.tensor(phi).cuda()

    # get the position along the principal axis(z-axis) on the unit sphere, in erp image coordinate.
    z_erp = torch.cos(phi) * torch.cos(theta)
    x_erp = torch.cos(phi) * torch.sin(theta)
    y_erp = torch.sin(phi)

    return [theta, phi, x_erp, y_erp, z_erp]

def generate_nfov_mask(img_grid):
    # this function generates mask of nfov images of multiple cams on the ERP image grid using pitch angle.
    # in current implementation, the yaw angles between the cameras are set to have the same angular intervals.
    # this function takes approximately 1ms using torch cuda tensor operations.
    
    hor_fov = float(random.randint(60,90))
    ver_fov = float(random.randint(60,90))
    pitch = float(random.randint(0, 90))
    n_cam = random.randint(1,int(max(min((90/(abs(pitch)+1e-9)),4),1))) # n_cam is adaptive to pitch angle (if pitch=90, n_cam is always 1.)
    flip = random.random()

    assert hor_fov >=0 and ver_fov >= 0 and hor_fov <= 180 and ver_fov <= 180, "fov of nvof image should be in [0, 180]."
    assert -90 <= pitch <= 90, "pitch angle should lie in [-90,90]."

    theta = img_grid[0]
    phi = img_grid[1]
    x_erp = img_grid[2]
    y_erp = img_grid[3]
    z_erp = img_grid[4]

    pitch = pitch * math.pi / 180
    yaw = math.pi * 2 / n_cam

    # max x and y value in the NFoV image plane.  
    max_y = math.tan( (ver_fov+2*pitch)*math.pi/(2*180))
    max_x = math.tan( hor_fov*math.pi/(2*180))
    min_y = math.tan( (-ver_fov+2*pitch)*math.pi/(2*180))
    min_x = math.tan( -hor_fov*math.pi/(2*180))

    # tilt the coordinate along x-axis using the pitch angle to get the nfov image coordinate.
    x_nfov = x_erp
    z_nfov = z_erp*math.cos(pitch) + y_erp*math.sin(pitch)
    y_nfov = -z_erp*math.sin(pitch) + y_erp*math.cos(pitch)

    mask = torch.zeros_like(phi) # initializing mask var.

    for cam in range(n_cam):
        # tilt the coordinate along y-axis using the pitch angle to get the nfov image coordinate.
        tilt_angle = yaw * cam

        y_nfov_ = y_nfov
        z_nfov_ = z_nfov*math.cos(tilt_angle) + x_nfov*math.sin(tilt_angle)
        x_nfov_ = -z_nfov*math.sin(tilt_angle) + x_nfov*math.cos(tilt_angle)

        # shoot the ray until z_nfov=1. We set the NFoV image plane is z_nfov=1 plane. (in other words, z axis is the principal axis)
        y = y_nfov_ / (z_nfov_ + 1e-9)
        x = x_nfov_ / (z_nfov_ + 1e-9)
        z = z_nfov_
        
        # masking the values that have appropriate FoV with positive z_nfov position.
        mask += (y > min_y) * (y < max_y) * (x > min_x) * (x < max_x) * (z >= 0)

    if flip < 1/2:
        mask = torch.flip(mask, [1])
    mask = mask > 0
    
    return torch.tensor(mask).float()

def generate_LIDAR_mask(img_grid): 
    theta = img_grid[0]
    phi = img_grid[1]
    x_erp = img_grid[2]
    y_erp = img_grid[3]
    z_erp = img_grid[4]

    channel = 2 ** random.randint(1,4)
    vert_fov_down = float(random.randint(1, 3) * channel )
    vert_fov_up = float(random.randint(1, 3) * channel )
    pitch = float(random.randint(-90, 90))
    yaw = random.randint(0,360)

    vert_fov_down = (-vert_fov_down) * math.pi / 180  # convert: degree -> radian [pi/2, -pi/2]
    vert_fov_up = (vert_fov_up) * math.pi / 180 # convert: degree -> radian [pi/2, -pi/2]

    pitch = pitch * math.pi / 180 # in radian
    yaw = int(yaw * img_grid[0].shape[1] / 360) # in pixel.

    # tilt the coordinate along x-axis using the pitch angle to get the lidar.
    x_lidar = x_erp
    z_lidar = z_erp*math.cos(pitch) + y_erp*math.sin(pitch)
    y_lidar = -z_erp*math.sin(pitch) + y_erp*math.cos(pitch)

    # sample lidar points using y value.
    vert_interval = (vert_fov_up - vert_fov_down) / channel

    mask = torch.zeros_like(phi)
    margin = math.pi / (phi.shape[0]) # margin is a hyper-parameter.
    for c in range(channel):
        angle = vert_fov_up - c * vert_interval
        mask += ((angle - margin < y_lidar) * (y_lidar < angle + margin) )

    mask = torch.roll(mask, yaw, dims=1)
    mask = mask > 0

    return torch.tensor(mask).float()

def get_activation(x, net, config):

    x = net(x)

    mean_feature = torch.mean( x, dim = 3, dtype = float )

    weight = torch.cos( torch.linspace(math.pi/2 , -math.pi/2 , mean_feature.shape[-1]) ).unsqueeze(0).unsqueeze(0).expand_as(mean_feature)
    mean_feature = weight * mean_feature

    mean_vector = mean_feature.view( -1, config.feature_size )
    
    return mean_vector

class tester:
    def __init__(self, config):
        self.config = config
        self.config.batchsize = 1
        
        self.global_img_grid = get_global_img_grid(config)
        
        # load trained model.
        
        self.G = network.Generator(config)
        checkpoint_path = config.G_checkpoint_path
        checkpoint = torch.load(checkpoint_path)
        self.G.load_state_dict(checkpoint['state_dict'])

        self.name = self.config.name #checkpoint_path[20:-8]
        print('name:',self.name)

        if torch.cuda.is_available():
            self.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')
        
        if self.use_cuda:
            torch.cuda.manual_seed(config.random_seed)
            self.G = torch.nn.DataParallel(self.G).cuda(device=0)
        
        # dataloader.
        self.loader = DL.dataloader(config)

        
    def test(self):
        if flag_imsave:
            utils.mkdir('repo/test')
            utils.mkdir(f'repo/test/{self.name}')
            utils.mkdir(f'repo/test/{self.name}/RGB')
            utils.mkdir(f'repo/test/{self.name}/D')

        # Encoder.
        A = AEnet.Encoder()
        A = torch.nn.DataParallel(A).cuda()
        A.eval()

        checkpoint_path = config.E_checkpoint_path
        print("[INFO] load Encoder checkpoint from: '{}'".format(checkpoint_path))  
        checkpoint = torch.load(checkpoint_path)
        
        checkpoint = dict( (key[8:], value) for (key, value) in checkpoint['state_dict'].items() if 'encoder' in key )
        A.module.load_state_dict(checkpoint, strict=False) 

        num_data = len(self.loader)

        data = np.zeros([1, config.feature_size])
        mean = np.zeros([config.feature_size])
        
        for iter in tqdm(range(0,self.loader.len_data_test)):
            with torch.no_grad():                    
                rgb_gt, d_gt, d_ini_gt, d_res_gt = self.loader.get_batch_test(idx = iter)

                input_mask = generate_input_mask_batch(self.config, self.global_img_grid)

                input = torch.cat( [rgb_gt, d_gt], dim=1 )

                in_img = input_mask * input


                x_test = self.G(in_img)
                
                outp_rgb = x_test[:,:3,:,:]

                scale_d = 93.62
                
                outp_d_ini = x_test[:,3:4,:,:]/scale_d
                outp_d_res = (x_test[:,-1:,:,:]-5)*1.922/scale_d
                
                outp_d = outp_d_ini-outp_d_res
                outp_d = torch.where(outp_d<0,torch.tensor(0,dtype=torch.float),outp_d)
                outp_d = outp_d * scale_d

                x_test = torch.cat( [outp_rgb, outp_d], dim=1 )

                if flag_stat:
                    act = get_activation(x_test, A, config)

                    act = act.detach().cpu().data.numpy()
                    data = np.append(data, act, axis=0)
                    mean += act.sum(axis=0)

                    d_name = self.loader.D_list_test[iter]
                    rgb_name = self.loader.RGB_list_test[iter]

                    rgb_out = x_test[:,:3,:,:]
                    d_out = x_test[:,-1:,:,:]    


        if flag_stat:
        
            mean /= num_data
            data = np.array(data[1:])

            mean = np.mean(data, axis=0)
            cov = np.cov(data, rowvar = False)

            path = 'stats'
            if not os.path.exists(path):
                if os.name == 'nt':
                    os.system('mkdir {}'.format(path.replace('/', '\\')))
                else:
                    os.system('mkdir -p {}'.format(path))

            np.savez(f'{path}/{config.name}', mean=mean, cov=cov)
            print(f"[INFO] statistics of the data: has been saved to: '{path}/'{config.name}'.")
    
if __name__ == '__main__':
    ## perform testing.
    print('----------------- configuration -----------------')
    for k, v in vars(config).items():
        print('  {}: {}'.format(k, v))
    print('-------------------------------------------------')
    torch.backends.cudnn.benchmark = True           # boost speed.
    tester = tester(config)
    tester.test()