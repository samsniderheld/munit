"""file to hold image and loss reporting"""
import cv2
import torch
import torchvision.utils as vutils


def write_loss(iterations, trainer, train_writer):
    """writes losses to tensorboard file"""
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)

def write_images(image_outputs, display_image_num, file_name):
    """writes images to file in a grid format"""
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data,
        nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_to_images(image_outputs, display_image_num, image_directory, postfix):
    """wrapper function for writing images"""
    num_images = len(image_outputs)
    write_images(image_outputs[0:num_images//2], display_image_num, f'{image_directory}/gen_a2b_{postfix}.jpg')
    write_images(image_outputs[num_images//2:num_images], display_image_num, f'{image_directory}/gen_b2a_{postfix}.jpg')