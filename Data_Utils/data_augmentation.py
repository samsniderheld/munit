###file for generating augmented data###
import os
import glob
import re
import argparse
import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm



def parse_args():
    """the argument parser"""
    desc = "A MUNIT Implementation Designed to run on multiple GPU's"

    #input and output args
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--base_data_dir', type=str,
        default="../../datasets/munit_barbarian/", help='The directory that holds the image data')

    parser.add_argument('--input_data_dir', type=str,
        default="train_A/", help='The directory for input data')

    parser.add_argument('--style_data_dir', type=str,
        default="train_B/", help='The directory for the style data')

    parser.add_argument("--output_dir", type=str,
        default="../../datasets/barbarian_unpaired", help="The directory for all the output results.")
    
    parser.add_argument('--num_samples', type=int,
        default=1000, help='number of samples to generate')


    return parser.parse_args()


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def transform(files,name,min_shear,max_shear):
    
    idx = np.random.randint(len(files))

    rand_x = np.random.random()
    rand_y = np.random.random()
    rand_trans = np.random.random()
    rand_shear = np.random.random() 
    # rand_contrast = np.random.random()
    # rand_brightness = np.random.random()
    # rand_saturation = np.random.random()

    rand_rot = np.random.randint(-365,365)
    random_x_trans = np.random.randint(-50,50)
    random_y_trans = np.random.randint(-50,50)
    shear_val = np.random.randint(min_shear,max_shear)
    # contrast_level = np.random.uniform(.5,1.5)
    # brightness_level = np.random.uniform(.5,1.5)
    # saturation_level = np.random.uniform(.5,1.5)

    replace_val = [0,255,0]



    img_raw = Image.open(files[idx]).convert(mode='RGB')


    img_a = img_raw.resize((1024,1024))
    img_a = T.functional.rotate(img_a, rand_rot,fill=replace_val)


    if rand_x > .5:
        img_a = T.functional.hflip(img_a)

    if rand_y > .5:
        img_a = T.functional.vflip(img_a)

    if rand_shear > .5:
        img_a = T.functional.affine(img_a, 0, [0,0],1.0,shear_val,fill=replace_val)

    if rand_trans > .5:
        img_a = T.functional.affine(img_a, 0, [random_x_trans,random_y_trans],1.0,0,fill=replace_val)


    return img_a


def main():

    args = parse_args()

    input_path = os.path.join(args.base_data_dir,args.input_data_dir,"*")
    style_path = os.path.join(args.base_data_dir,args.style_data_dir,"*")

    print(input_path)
    print(style_path)

    output_input_path = os.path.join(args.output_dir,"train_A")
    output_style_path = os.path.join(args.output_dir,"train_B")

    print(output_input_path)
    print(output_style_path)

    files_A = sorted(glob.glob(input_path), key=natural_keys)
    files_B = sorted(glob.glob(style_path), key=natural_keys)



    for i in tqdm(range(0,args.num_samples)):
        img_a = transform(files_A,i,50,70)
        img_b = transform(files_B,i,50,70)

        img_a_name = os.path.join(output_input_path,f'{i:04d}.jpg')
        img_b_name = os.path.join(output_style_path,f'{i:04d}.jpg')

        img_a.save(img_a_name)
        img_b.save(img_b_name)

if __name__ == '__main__':
    main()
