import numpy as np
import matplotlib
matplotlib.use('Agg') # don't generate display windows for plt
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from cs_exp import NICE_ICS, CSDataset
from util.visualizer import Visualizer

import time
import os
import argparse
import math, random

from options import CSTestOptions
import glob


if __name__ == '__main__':
    # Get default options
    opt = CSTestOptions().parse(['--gpu_ids','0','--mute'])
    
    # Get experiment Paths
    checkpoint_paths = [os.path.split(path)[1] for path in glob.glob('./checkpoints/nice_ics_*')]# if path.find('l1e-1') > 0 or path.find('l1e0') > 0]
    checkpoint_paths.sort()
    #paths = paths[:-1]

    tested = [os.path.split(path)[1][:-4] for path in glob.glob('./results/*.png')]
    ignore = ['nice_ics_jacvec1e-2'] 
    #ignore = ['nice_cs_nl3_jacvece-3_lr2e-4']

    paths = [path for path in checkpoint_paths if path not in tested and path not in ignore]
    #print(checkpoint_paths)
    for path in paths:
      print(path)
    for name in paths:
        # extract options
        #_, name = os.path.split(path)
        nl = 3#int(name[name.find('nl')+2])
        
        print(name)

        use_batch_norm = True#len(name)==11
        
        opt.name = name
        opt.num_layers = nl
        opt.use_batch_norm = use_batch_norm
        mse = nn.MSELoss()

        
        epoch_nums=np.arange(500,500*len(glob.glob(os.path.join('./checkpoints/',name,'*00_*')))+1,500)
        
        # Load training data
        dataset = CSDataset('datasets/cs_f50_data.npz')
        data_loader = DataLoader(dataset, batch_size=10000,shuffle=False)
        
        forward_mses = np.zeros((len(epoch_nums),len(data_loader)))
        reverse_mses = np.zeros((len(epoch_nums),len(data_loader)))
        for i,epoch in enumerate(epoch_nums):
            opt.epoch = epoch
            model = NICE_ICS(opt).eval()
            for j,data in enumerate(data_loader):
                model.set_input(data)
                model.reverse()
                with torch.no_grad():
                    model.forward()
                    forward_mses[i,j] = mse(model.yhats,model.ys).item()
                    reverse_mses[i,j] = mse(model.xhats,model.xs).item()
            del model
            torch.cuda.empty_cache()

        # Load testing data
        dataset = CSDataset('datasets/cs_f50_test_data.npz')
        data_loader = DataLoader(dataset, batch_size=10000,shuffle=False)
        
        test_forward_mses = np.zeros((len(epoch_nums),len(data_loader)))
        test_reverse_mses = np.zeros((len(epoch_nums),len(data_loader)))
        for i,epoch in enumerate(epoch_nums):
            opt.epoch = epoch
            model = NICE_ICS(opt).eval()
            for j,data in enumerate(data_loader):
                model.set_input(data)
                model.reverse()
                with torch.no_grad():
                    model.forward()
                    test_forward_mses[i,j] = mse(model.yhats,model.ys).item()
                    test_reverse_mses[i,j] = mse(model.xhats,model.xs).item()

            del model
            torch.cuda.empty_cache()
            
        
        # Save Results
        outpath = os.path.join('./results',name+'.png')
        plt.figure(figsize=(16,8))
        plt.subplot(121)
        plt.plot(epoch_nums,forward_mses.mean(axis=1),label='Training')
        plt.plot(epoch_nums,test_forward_mses.mean(axis=1),label='Test')
        plt.legend(loc=0)
        plt.title(opt.name + ' Forward')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')

        plt.subplot(122)
        plt.plot(epoch_nums,reverse_mses.mean(axis=1),label='Training')
        plt.plot(epoch_nums,test_reverse_mses.mean(axis=1),label='Test')
        plt.legend(loc=0)
        plt.title(opt.name + ' Reverse')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')

        plt.savefig(outpath,bbox_inches='tight')



