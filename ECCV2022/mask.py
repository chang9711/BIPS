import torch
import random
import math
import numpy as np

def generate_input_mask_batch(config, img_grid):
    mask = generate_input_mask(img_grid).unsqueeze(0)
    for batch in range(1,config.batchsize):
        mask = torch.cat([mask, generate_input_mask(img_grid).unsqueeze(0)], dim=0)
    return mask

def generate_input_mask(img_grid):
    
    # this function generates the final input mask.
    input_mode = random.random() # switch btw RGB, D, RGBD input modes. sample from 0~1 uniform distribution.

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
    pitch = float(random.randint(-90, 90))
    n_cam = random.randint(1,int(max(min((90/(abs(pitch)+1e-9)),4),1))) # n_cam is adaptive to pitch angle (if pitch=90, n_cam is always 1.)
    #print(f'hor_fov:{hor_fov}, ver_fov:{ver_fov}, pitch:{pitch}, n_cam:{n_cam}')
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

if __name__ == "__main__":
    # save image to check the masks..
    import utils
    class Config():
        def __init__(self, x):
            self.imsize = x

    config = Config(512)

    img_grid = get_global_img_grid(config)
    rgb_out = generate_LIDAR_mask(img_grid)
    print(rgb_out.shape)
    rgb_out = rgb_out.unsqueeze(0).unsqueeze(0)
    rgb_out = torch.cat([rgb_out, rgb_out, rgb_out], dim=1)
    print(rgb_out.shape)
    utils.save_image_rgb(rgb_out.data, config, f'temp.png')