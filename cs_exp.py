import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset
import torch.autograd as autograd


from nice.models import NICEAdditiveModel as NICEModel

import time
import os
import argparse
import math, random

# Define Options
from util import util
from collections import OrderedDict




class CSOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', type=str, default='cs_data.npz', help='path to data')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        
        # model parameters
        parser.add_argument('--input_dim', type=int, default=500, help='size of input vector')
        parser.add_argument('--hidden_dim', type=int, default=1000, help='size of hidden dimension for NICE model')
        parser.add_argument('--num_layers', type=int, default=3, help='# of Invertible Modules')
   
        
        # Visdom related stuff
        parser.add_argument('--display_freq', type=int, default=50000, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=3, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=5500, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=100000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=50000, help='frequency of showing training results on console')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        
        # Training Parameters
        parser.add_argument('--batch_size', type=int, default=10000, help='input batch size')

        parser.add_argument("--nEpochs", type=int, default=10000, help="number of epochs to train for")
        parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
        parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
        parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
        parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
        parser.add_argument("--isTrain",action="store_true",help="Train mode")
        parser.add_argument('--save_latest_freq', type=int, default=100000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=500, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.isTrain = True

        
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

    
# Dataset
class CSDataset(Dataset):
    def __init__(self,path):
        super().__init__()
        data = np.load(path)
        self.Xs = torch.Tensor(data['Xs'])
        self.Ys = torch.Tensor(data['Ys'])
    
    def __len__(self):
        return self.Xs.shape[0]
    
    def __getitem__(self,idx):
        return (self.Xs[idx],self.Ys[idx])

class BASEModel(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu') 
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        
        
        # Create Networks
        self.netF = NICEModel(opt.input_dim, opt.hidden_dim, opt.num_layers,opt.use_batch_norm).to(self.device)
        
        self.model_names = ['F']
        
        if not opt.isTrain:
            # Load networks
            self.load_networks(opt.epoch)
    
    def set_input(self,input):
        xs,ys = input
        self.xs = xs.to(self.device)
        self.ys = ys.to(self.device)
        
    def forward(self):
        self.yhats = self.netF(self.xs)
        
    def reverse(self):
        self.xhats = self.netF.inverse(self.ys,train=self.isTrain)
    
    def compute_jacvec(self):
        grad_outputs = torch.randn(self.yhats.size()).to(self.device)
        J = autograd.grad(outputs=self.yhats, inputs=self.xs,
                 grad_outputs=grad_outputs,
                 create_graph=True, retain_graph=True, only_inputs=True)[0]
        gt = torch.matmul(grad_outputs,self.AtA)
        
        return self.criterionMSE(J,gt)
    
    def backward_F(self):
        # Content Loss
        pass
        
    def optimize_parameters(self):
        # forward
        pass 
    
    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.to(self.device)
                else:
                    torch.save(net.cpu().state_dict(), save_path)
                    
    def load_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)
    
    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret     

# Define Model
class NICE_CS(BASEModel):
    def __init__(self,opt):
        super().__init__(opt)
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu') 
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        
        
        # Create Networks
        self.netF = NICEModel(opt.input_dim, opt.hidden_dim, opt.num_layers,opt.use_batch_norm).to(self.device)
        
        self.model_names = ['F']
        
        if self.isTrain:
            # Define Losses
            self.criterionMSE = torch.nn.MSELoss()
            self.loss_names = ['mse']
            
            # Define Optimizers
            self.optimizer = torch.optim.Adam(self.netF.parameters(), lr=opt.lr)
            
            if opt.l1 > 0:
                self.criterionL1 = torch.nn.L1Loss()
                self.loss_names.append('l1')
            if opt.fb > 0:
                self.loss_names.append('imse')
                if opt.fb_l1:
                    self.criterionFB = torch.nn.L1Loss()
                else:
                    self.criterionFB = torch.nn.MSELoss()
            if opt.jacvec > 0:
                self.loss_names.append('jacvec')
                self.AtA = torch.Tensor(np.load(opt.AtA_loc)).to(self.device)
        
        if not opt.isTrain:
            # Load networks
            self.load_networks(opt.epoch)
    
    
    def backward_F(self):
        # Content Loss
        self.loss_mse = self.criterionMSE(self.yhats,self.ys)#*self.opt.lambda_l1
        self.loss = self.loss_mse

        if self.opt.l1 > 0 or self.opt.fb > 0:
            self.reverse()
        if self.opt.l1 > 0:
            self.loss_l1 = self.criterionL1(self.xhats, torch.zeros_like(self.xhats).to(self.device))*self.opt.l1
            self.loss += self.loss_l1
        if self.opt.fb > 0:
            self.loss_imse = self.criterionFB(self.xhats,self.xs)*self.opt.fb
            self.loss += self.loss_imse
        if self.opt.jacvec > 0:
            self.loss_jacvec = self.compute_jacvec()*self.opt.jacvec
            self.loss += self.loss_jacvec
        # Calculate Gradients
        self.loss.backward()
        
    def optimize_parameters(self):
        # forward
        if self.opt.jacvec > 0:
            self.xs.requires_grad_(True)
        self.forward()
        
        # G
        self.optimizer.zero_grad()
        self.backward_F()
        self.optimizer.step()
        

# Define Model
class NICE_ICS(BASEModel):
    def __init__(self,opt):
        super().__init__(opt)
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu') 
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        
        
        # Create Networks
        self.netF = NICEModel(opt.input_dim, opt.hidden_dim, opt.num_layers,opt.use_batch_norm).to(self.device)
        
        self.model_names = ['F']
        
        if self.isTrain:
            # Define Losses
            if opt.use_l1:
                self.criterionINV = torch.nn.L1Loss()
            else:
                self.criterionINV = torch.nn.MSELoss()
            self.loss_names = ['cs']
            
            # Define Optimizers
            self.optimizer = torch.optim.Adam(self.netF.parameters(), lr=opt.lr)
            
            if opt.jacvec > 0:
                self.criterionJACVEC = torch.nn.MSELoss()
                self.loss_names.append('jacvec')
                self.AtA = torch.Tensor(np.load(opt.AtA_loc)).to(self.device)
        
        if not opt.isTrain:
            # Load networks
            self.load_networks(opt.epoch)
    
    
    def compute_jacvec(self):
        grad_outputs = torch.randn(self.yhats.size()).to(self.device)
        J = autograd.grad(outputs=self.yhats, inputs=self.xs,
                 grad_outputs=grad_outputs,
                 create_graph=True, retain_graph=True, only_inputs=True)[0]
        gt = torch.matmul(grad_outputs,self.AtA)
        
        return self.criterionJACVEC(J,gt)
    
    def backward_F(self):
        # Content Loss
        self.reverse()
        self.loss_cs = self.criterionINV(self.xhats,self.xs)#*self.opt.lambda_l1
        self.loss = self.loss_cs

        if self.opt.jacvec > 0:
            self.forward()
            self.loss_jacvec = self.compute_jacvec()*self.opt.jacvec
            self.loss += self.loss_jacvec
        # Calculate Gradients
        self.loss.backward()
        
    def optimize_parameters(self):
        # forward
        if self.opt.jacvec > 0:
            self.xs.requires_grad_(True)
        
        # G
        self.optimizer.zero_grad()
        self.backward_F()
        self.optimizer.step()
        
# Define Model
class FB_NICE_CS(NICE_CS):
    def __init__(self,opt):
        super().__init__(opt)
        if self.isTrain:
            self.loss_names.append('imse')
    
    def backward_F(self):
        # Content Loss
        self.loss_mse = self.criterionMSE(self.yhats,self.ys)#*self.opt.lambda_l1
        
        # Backward Loss
        self.loss_imse = self.criterionMSE(self.xhats,self.xs)
        
        # Calculate Gradients
        self.loss = self.loss_mse + self.loss_imse
        self.loss.backward()
        
    def optimize_parameters(self):
        # forward
        self.forward()
        self.reverse()
        
        # G
        self.optimizer.zero_grad()
        self.backward_F()
        self.optimizer.step()
        
        
