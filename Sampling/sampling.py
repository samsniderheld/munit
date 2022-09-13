import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from Data_Utils.data_utils import *
from Utils.util_functions import save_sampled_images
import torchvision.utils as vutils
from Training.trainer import *

def sample_images(args):
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.gpus = args.gpu = 0

    #setup data
    print("loading data")
    data_loader = get_data_loader_folder(args, os.path.join(args.base_data_dir, args.input_data_dir),
        args.batch_size, False, args.img_width, args.crop_size, args.crop_size)
    image_names = ImageFolder(os.path.join(args.base_data_dir, args.input_data_dir), transform=None, return_paths=True)


    # load model
    print("loading model")
    trainer = MUNIT_Trainer(args)
    trainer.load_pretrained_gen(args.saved_model_dir)
    trainer.to(args.device)
    trainer.eval()

    encode = trainer.generator_a.encode  # encode function
    decode = trainer.generator_b.decode  # decode function
    style_encode = trainer.generator_b.encode  # style encode function
    style_dim = trainer.style_dim

    # style
    s = torch.randn(1, style_dim, 1, 1).to(args.device)
    transform = transforms.Compose([transforms.Resize(1024),
                                transforms.ToTensor(),
                                # transforms.GaussianBlur(kernel_size=(9, 9), sigma=(5.0, 5.0)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    if args.style_img != '':
        style_image = Variable(transform(Image.open(args.style_img).convert('RGB')).unsqueeze(0).cuda()) 
        _, style = style_encode(style_image)
    else:
        style = Variable(torch.randn(1, 8, 1, 1).cuda())

    del style_encode

    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        images = Variable(images.to(args.device))
        content, _ = encode(images) 
        outputs = decode(content, style)
        outputs = (outputs + 1) / 2.
        path = os.path.join(args.output_images_path, 'output{:09d}.{}'.format(i,"png"))
        vutils.save_image(outputs.data, path, padding=0, normalize=True)
        del images
        del content
        del outputs
        #save_sampled_images(image_outputs.to(images_a.dtype), args.output_images_path, file_name='test')

    return print("done prediction")
