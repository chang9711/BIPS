import argparse
import time
import numpy as np

parser = argparse.ArgumentParser('360 RGBD Scene Generation 2022 ECCV ver.')

## general settings.
parser.add_argument('--name', type=str, required=False, default='test')          # name of the training attempt.
parser.add_argument('--train_data_root', type=str, default='./dataset/train')
parser.add_argument('--val_data_root', type=str, default='./dataset/val')
parser.add_argument('--test_data_root', type=str, default='/content/drive/MyDrive/KCVS/TA_session/test')
parser.add_argument('--random_seed', type=int, default=int(time.time()))
parser.add_argument('--input_mode', type=str, default='RGBD')    # input modes 
## training parameters.
parser.add_argument('--lr', type=float, default=0.001)          # learning rate.
parser.add_argument('--lr_decay', type=float, default=0.995)    # learning rate decay at every resolution transition. currently not using this value.
parser.add_argument('--outnc', type=int, default=5)             # number of output channel. e.g., RGB + D: 4
parser.add_argument('--ngf', type=int, default=32)              # feature dimension of first layer of generator.
parser.add_argument('--max_ngf', type=int, default=512)         # maximum feature dimension of generator.
parser.add_argument('--epoch', type=int, default=100)        # epochs to train.
parser.add_argument('--batchsize', type=int, default=2)         # batch size.
parser.add_argument('--imsize', type=int, default=512)          # image size (height). 512 means 512x1024 resolution.
parser.add_argument('--lambda_l1_all', type=int, default=20)    # lambda: weight of l1 loss_all.
parser.add_argument('--lambda_gan', type=int, default=1)        # lambda: weight of gan loss.
parser.add_argument('--scale_d', type=float, default=1/700)         # depth scaling factor in the network. d value will lie in [0, 10]. 10/7000 (we set 7,000 to d_max=10)
parser.add_argument('--d_res_hmax', type=float, default=1.922)
parser.add_argument('--depth_thres', type=float, default=6727.)
parser.add_argument('--residual_thres', type=float, default=20.)

parser.add_argument('--resume', type=bool, default=False)       # resume training flag.
parser.add_argument('--mask_valid_pixel', type=bool, default=False)# Whether not to train on invliad pixels or not.
parser.add_argument('--feature_size', type=int, default = 4096, help = 'feature size of auto encoder.')

## testing parameters.
parser.add_argument('--G_checkpoint_path', type=str, default='/content/drive/MyDrive/KCVS/TA_session/gen_name_residual_branch_two.pth')
parser.add_argument('--D_checkpoint_path', type=str, default='')
parser.add_argument('--E_checkpoint_path', type=str, default='/content/drive/MyDrive/KCVS/TA_session/RGBQuality_resume_Structured3d_mod.pth.tar')

parser.add_argument('--gt_stat_path', type=str, default='')

## network structure.
parser.add_argument('--flag_wn', type=bool, default=True)           # use of equalized-learning rate.
parser.add_argument('--flag_bn', type=bool, default=False)          # use of batch-normalization. (not recommended)
parser.add_argument('--flag_in', type=bool, default=False)           # use of instance-normalization.
parser.add_argument('--flag_pixelwise', type=bool, default=False)   # use of pixelwise normalization for generator.
parser.add_argument('--flag_gdrop', type=bool, default=False)       # use of generalized dropout layer for discriminator.
parser.add_argument('--flag_leaky', type=bool, default=True)        # use of leaky relu instead of relu.
parser.add_argument('--flag_tanh', type=bool, default=True)         # use of tanh at the end of the generator.
parser.add_argument('--flag_sigmoid', type=bool, default=False)     # use of sigmoid at the end of the discriminator.
parser.add_argument('--flag_sigmoid_depth', type=bool, default=True)# use of sigmoid at the end of the generator (for depth).
parser.add_argument('--flag_add_noise', type=bool, default=False)   # add noise to the real image(x)
parser.add_argument('--flag_norm_latent', type=bool, default=False) # pixelwise normalization of latent vector (z)
parser.add_argument('--flag_rotate', type=bool, default=True)       # rotate the images when loading images.
parser.add_argument('--flag_circular_pad', type=bool, default=True) # ERP circular padding.
parser.add_argument('--flag_color_jitter', type=bool, default=False) # color jitter flag while training.
parser.add_argument('--flag_zero_invalid_depth', type=bool, default=True) # zero the maximum (invalid) depth value.

## optimizer setting.
parser.add_argument('--optimizer', type=str, default='adam')        # optimizer type.
parser.add_argument('--beta1', type=float, default=0.9)             # beta1 for adam.
parser.add_argument('--beta2', type=float, default=0.999)            # beta2 for adam.

## display and save setting.
parser.add_argument('--save_img_every', type=int, default=100)      # save images every specified iteration.
parser.add_argument('--validate_every', type=int, default=np.inf)       # validation every specified iteration.
parser.add_argument('--display_tb_every', type=int, default=30)    # display progress every specified iteration.

## parse and save config.
config, _ = parser.parse_known_args()
