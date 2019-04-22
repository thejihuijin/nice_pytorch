import argparse
import os
from util import util
import torch

class BaseOptions():
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
        parser.add_argument('--dataroot', type=str, default='datasets/cs_data.npz', help='path to data')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--mute', action='store_true', help='if specified, dont print option settings')



        # Model Parameters
        parser.add_argument('--input_dim', type=int, default=500, help='size of input vector')
        parser.add_argument('--hidden_dim', type=int, default=1000, help='size of hidden dimension for NICE model')
        parser.add_argument('--num_layers', type=int, default=3, help='# of Invertible Modules')
        parser.add_argument('--use_batch_norm', action='store_false', help='Turn off batch norm in NICE model')
   
        
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.initialized = True
        self.isTrain = False
        return parser

    def gather_options(self,args):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                                             conflict_handler='resolve')
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        if args is not None:
            return parser.parse_known_args(args)[0]
        
        return parser.parse_known_args()[0]

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

    def parse(self,args=None):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options(args)
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if not opt.mute:
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

class TrainOptions(BaseOptions):
    def initialize(self,parser):
        parser = super().initialize(parser)
        
        parser.add_argument("--batch_size", type=int, default=10000, help="training batch size")
        parser.add_argument("--nEpochs", type=int, default=5000, help="number of epochs to train for")
        parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
        parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
        #parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
        parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
        parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
        parser.add_argument('--save_latest_freq', type=int, default=100000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=500, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')


        # Visdom related stuff
        parser.add_argument('--display_freq', type=int, default=50000, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=3, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=-1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=5500, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=100000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=50000, help='frequency of showing training results on console')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
               
        
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        self.isTrain = True
        

        return parser
class ICSTrainOptions(TrainOptions):
    def initialize(self,parser):
        parser = super().initialize(parser)
        parser.add_argument('--dataroot', type=str, default='datasets/cs_f50_data.npz', help='path to data')
        parser.add_argument("--batch_size", type=int, default=5000, help="training batch size")

        
        parser.add_argument("--use_l1", action='store_true', help='Use L1 instead of MSE')
        
        # Regularizers
        parser.add_argument("--jacvec", type=float,default=0, help="Train with projected jacobian information")
        parser.add_argument("--AtA_loc",type=str,default='AtA.npy', help='Location of AtA matrix')
        
        return parser

        

class CSTrainOptions(TrainOptions):
    def initialize(self,parser):
        parser = super().initialize(parser)

        # Regularizers
        parser.add_argument("--l1", type=float, default=0, help="L1 sparsity regularizer on inverse mode. Default=0")
        parser.add_argument("--fb", type=float, default=0, help="Train on inverse mode. Default=0")
        parser.add_argument("--fb_l1", action='store_true', help="use L1 norm on imse")
        parser.add_argument("--jacvec", type=float,default=0, help="Train with projected jacobian information")
        parser.add_argument("--AtA_loc",type=str,default='AtA.npy', help='Location of AtA matrix')

        
        return parser    
class CSTestOptions(BaseOptions):
    def initialize(self,parser):
        parser = super().initialize(parser)
        
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        self.isTrain = False
        
        return parser
