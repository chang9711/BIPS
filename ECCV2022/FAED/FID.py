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
from scipy import linalg
import time

parser = argparse.ArgumentParser(description = 'Frechet Distance.')
parser.add_argument('--real_path', type=str, default = '', help = 'real stats path.')
parser.add_argument('--fake_path', type=str, default = '', help = 'fake stats path.')

config = parser.parse_args()

### https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
def sqrt_newton_schulz(A, numIters, dtype):
    batchSize = A.shape[0]
    dim = A.shape[1]
    A = A.type(dtype)
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt().type(dtype)
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A)).type(dtype)
    I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    Z = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    for i in range(numIters):
        T = 0.5*(3.0*I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    error = compute_error(A, sA)
    
    print(f'[INFO] matrix sqrt calculation error: {error}.')
    return sA, error

def compute_error(A, sA):
    normA = torch.sqrt(torch.sum(torch.sum(A * A, dim=1),dim=1))
    error = A - torch.bmm(sA, sA)
    error = torch.sqrt((error * error).sum(dim=1).sum(dim=1)) / normA
    return torch.mean(error.abs())

def sqrtm_positivesemidefinite(A):
    # since we are dealing with covariance matrices, it is always diagonalizable.
    # in this function, we first diagonalize the matrix using torch.symeig: A = V @ D @ VT.
    # then, we can get sqrt(A) = V @ sqrt(D) @ VT.

    print(f"[INFO] calculating square root of matrix...")
    start = time.time()
    A = A.cuda()
    print((A.transpose(-1,-2)-A).abs().max())
    w, v = torch.eig(A, eigenvectors=True) # eigenvalue decomposition.
    print(w.unique().shape, w.reshape(-1).shape)
    print(w.unique().shape == w.reshape(-1).shape)
    
    print(torch.diag(v.transpose(0,1).matmul(A).matmul(v)).abs().min())
    print((v.matmul(A).matmul(v.transpose(0,1)) - A).abs().mean())

    diag = (v.transpose(0,1).matmul(A).matmul(v)).clamp(min=0, max=np.inf) # due to numerical error, there exists small negative components.
    
    sqrtdiag = torch.diag(torch.sqrt(torch.diag(diag)))
    sqrtm = v.matmul(sqrtdiag).matmul((v.transpose(0,1)))
    error = (sqrtm.matmul(sqrtm) - A).abs().mean()
    print(f"[INFO] square root of matrix calculated. mean error: {error:.3f}, time took: {time.time()-start:.3f}s.")
    return sqrtm

def sqrtm(A):
    # Product might be almost singular
    start = time.time()
    covmean, _ = linalg.sqrtm(A, disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print('Imaginary component {}'.format(m))
        covmean = covmean.real
    covmean = torch.tensor(covmean).cuda()
    error = (covmean.matmul(covmean) - torch.tensor(A).cuda()).abs().mean()
    print(f"[INFO] square root of matrix calculated. mean error: {error:.3f}, time took: {time.time()-start:.3f}s.")
    return covmean

def calculate_Frechet_distance(mu1, sigma1, mu2, sigma2):

    diff = (mu1 - mu2).squeeze()
    feature_size = mu1.shape[-1]
    #sqrt_cov, _ = linalg.sqrtm(sigma1.matmul(sigma2).cpu().detach().squeeze(), disp=False)
    #sqrt_cov, _ = sqrt_newton_schulz(sigma1.bmm(sigma2), 10, 'torch.cuda.FloatTensor')
    
    mat = sigma1.bmm(sigma2).squeeze()
    sqrt_cov = torch.tensor(sqrtm(mat.squeeze().cpu().numpy()))
    # sqrt_cov = torch.tensor(sqrtm_positivesemidefinite(mat.squeeze())) 
    tr_sqrt_cov = torch.trace(sqrt_cov.squeeze())

    return (diff.dot(diff) + torch.trace(sigma1.squeeze()) + torch.trace(sigma2.squeeze()) - 2 * tr_sqrt_cov)#.clamp(min=0, max=np.inf)

class MatrixSQRT(nn.Module):
    def __init__(self, feature_size):
        super(MatrixSQRT, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(1, feature_size, feature_size, requires_grad=True))
        self.feature_size = feature_size
        
    def forward(self):
        #x = torch.ones_like(self.weight.weight.data)
        
        return self.weight.bmm(self.weight)

if __name__ == '__main__':
    print('----------------- configuration -----------------')
    for k, v in vars(config).items():
        print('  {}: {}'.format(k, v))
    print('-------------------------------------------------')
    torch.backends.cudnn.benchmark = True           # boost speed.
    
    print(f"[INFO] loading real data statistics from: '{config.real_path}'.")
    real_data = np.load(config.real_path)
    real_mean = real_data['mean']
    real_cov = real_data['cov']
    print(f"[INFO] loading fake data statistics from: '{config.fake_path}'.")
    fake_data = np.load(config.fake_path)
    fake_mean = fake_data['mean']
    fake_cov = fake_data['cov']

    print('[INFO] calculating Frechet distance.')
    mu1 = torch.tensor(real_mean).unsqueeze(0)
    mu2 = torch.tensor(fake_mean).unsqueeze(0)
    sigma1 = torch.tensor(real_cov).unsqueeze(0)
    sigma2 = torch.tensor(fake_cov).unsqueeze(0)

    fid_score = calculate_Frechet_distance(mu1,sigma1,mu2,sigma2)
    print(f'[INFO] Frechet distance is: {fid_score}.')

    import pandas as pd
    import os

    if not os.path.isfile('./stats/experiments.xlsx'):
        df = pd.DataFrame({'dataset':['BlenderDataset_train'], 'real':[f'{config.real_path}'], 'fake':[f'{config.fake_path}'], 'FD':[f'{fid_score.cpu().numpy()}']})
        df.to_excel('./stats/experiments.xlsx')
    else:
        df = pd.read_excel('./stats/experiments.xlsx', index_col=0)
        df = df.append({'dataset':'BlenderDataset_train', 'real':f'{config.real_path}', 'fake':f'{config.fake_path}', 'FD':f'{fid_score.cpu().numpy()}'}, ignore_index=True)
        df.to_excel('./stats/experiments.xlsx')