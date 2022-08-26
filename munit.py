"""
munit.py
The entry path for the neural network training.
"""
import argparse
import os
from datetime import datetime
from Training.training import train
import torch.distributed as dist
import torch.multiprocessing as mp



def parse_args():
    """the argument parser"""
    desc = "A MUNIT Implementation Designed to run on multiple GPU's"

    #input and output args
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--base_data_dir', type=str,
        default="Data/", help='The directory that holds the image data')

    parser.add_argument('--input_data_dir', type=str,
        default="train_A/", help='The directory for input data')

    parser.add_argument('--style_data_dir', type=str,
        default="train_B/", help='The directory for the style data')

    parser.add_argument("--output_dir", type=str,
        default="base_results_dir/", help="The directory for all the output results.")

    parser.add_argument("--experiment_name", type=str,
        default="munit", help='Identifying str to label the experiment')

    #args for input data
    parser.add_argument('--img_width', type=int, default=512, help='The width of the image')

    parser.add_argument('--img_height', type=int, default=512, help='The width of the image')

    parser.add_argument('--crop_size', type=int, default=256, help='The width of the image')

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
        default=2, help='The number of epochs')
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

    #continue training from a previous experiment
    parser.add_argument('--continue_training', action='store_true')

    #distributed training    
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='Number of nodes')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='Number of GPUs per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='Ranking within the nodes')
    parser.add_argument('--world_size', default=0, type=int, help='world size')
    parser.add_argument('--rank', default=0, type=int, help='rank')
    if dist.is_available():
        parser.add_argument('--backend', type=str, help='distributed backend',
                        choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                        default=dist.Backend.NCCL)


    return parser.parse_args()

def main():
    """The main entrance to the training loops"""

    args = parse_args()

    args.experiment_name = datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + args.experiment_name

    args.base_results_dir = os.path.join(args.output_dir,args.experiment_name)

    if not os.path.exists(args.base_results_dir):
        os.makedirs(args.base_results_dir)
        os.makedirs(os.path.join(args.base_results_dir,"Generated_Images"))
        os.makedirs(os.path.join(args.base_results_dir,"Loss"))
        os.makedirs(os.path.join(args.base_results_dir,"Saved_Models"))


    args.image_results_dir = os.path.join(args.base_results_dir, "Generated_Images")

    args.saved_model_dir = os.path.join(args.base_results_dir,"Saved_Models")

    # Multiprocessing
    if args.gpus > 1:
        args.world_size = args.gpus * args.nodes
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    else:
        train(0,args)


    # print("done training")


if __name__ == '__main__':
    main()
