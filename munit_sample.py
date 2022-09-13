from torch import nn
from scipy.stats import entropy
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import sys
import torch
import os
import torch.distributed as dist

from Data_Utils.data_utils import *
from Sampling import sample_images
from Training.trainer import *

from tqdm import tqdm

def parse_args():
    """the argument parser"""
    desc = "A MUNIT Implementation Designed to run on multiple GPU's"

    #input and output args
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--base_data_dir', type=str,
        default="Data/", help='The directory that holds the image data')

    parser.add_argument('--input_data_dir', type=str,
        default="test_A/", help='The directory for input data')

    parser.add_argument('--style_data_dir', type=str,
        default="test_B/", help='The directory for the style data')

    parser.add_argument("--output_dir", type=str,
        default="base_results_dir/", help="The directory for all the output results.")

    parser.add_argument("--experiment_name", type=str,
        default="munit", help='Identifying str to label the experiment')
    
    parser.add_argument('--output_images_path', type=str, 
        default="Sampled_images", help='Folder name for save images during training')

    parser.add_argument('--output_images_subfolder_path', type=str, 
        default="", help='Subfolder name for save images during training')

    #args for input data
    parser.add_argument('--img_width', type=int, default=1024, help='The width of the image')

    parser.add_argument('--img_height', type=int, default=1024, help='The width of the image')

    parser.add_argument('--crop_size', type=int, default=1024, help='The width of the image')

    parser.add_argument('--num_workers', type=int,
        default=8, help='number of workers when data processing')

    #args for saving
    parser.add_argument('--print_freq', type=int,
        default=10, help='How often is the status printed')

    parser.add_argument('--save_freq', type=int, default=500, help='How often is the model saved')

    parser.add_argument('--display_size', type=int,
        default=16, help="how many images to display when printing results")

    #gen architecture hyper parameters
    parser.add_argument('--num_bottom_filters', type=int,
        default=64, help='Num of filters in last layer')
    
    parser.add_argument('--mlp_dims', type=int,
        default=256, help='Num of filters in mlp')
    
    parser.add_argument('--num_down_sample_layers', type=int,
        default=2, help='The number of down sample layers')
    
    parser.add_argument('--num_res_blocks', type=int,
        default=4, help='The number of residual blocks')

    #training options
    parser.add_argument('--max_iter', type=int, default=20, help='The number of epochs')

    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch')

    parser.add_argument('--lr_policy_step_size', type=int, default=100000, help='how often to decay the lr')

    #loss function related arguments
    parser.add_argument('--gan_w', type=float,
        default=1, help='how much to weight the gan adversarial loss')

    parser.add_argument('--recon_x_w',type=float,
        default=10, help='how much to weight the image reconstruction loss')

    parser.add_argument('--recon_s_w', type=float,
        default=1, help='how much to weight the style reconstruction loss')

    parser.add_argument('--recon_c_w', type=float,
        default=1, help='how much to weight the content reconstruction loss')

    parser.add_argument('--style_img',type=str, default="")
    
    # parser.add_argument('--checkpoint',type=str, default="Saved_Models/gen_00033500.pt")
 
    return parser.parse_args()


def main():
    """The main entrance to the training loops"""

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    
    args = parse_args()

    args.experiment_name = args.experiment_name
    
    args.base_results_dir = os.path.join(args.output_dir,args.experiment_name)
    args.output_images_path = os.path.join(args.base_results_dir, args.output_images_path, args.output_images_subfolder_path)

    # Output path dir
    print('creating directories in ' + args.base_results_dir)
    os.makedirs(args.base_results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_images_path), exist_ok=True)

    args.saved_model_dir = os.path.join(args.base_results_dir,"Saved_Models")

    print('Recovering model from ' + args.saved_model_dir)
    
    sample_images(args) 


    # print("done training")


if __name__ == '__main__':
    main()
