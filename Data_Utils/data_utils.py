"""file that handles all of the data loading and support functions"""
import os
import os.path
import torch.utils.data as data
import torchvision.transforms as transforms
from torch import nn

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from Utils.util_functions import is_distributed

from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def default_loader(path):
    """PIL image ope interface"""
    return Image.open(path).convert('RGB')

def is_image_file(filename):
    """make sure that file is an image"""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    """open all images in path and output them into a list"""
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def isDDP(model):
    return isinstance(model, nn.parallel.DistributedDataParallel)


class ImageFolder(data.Dataset):
    """Simple class for representing all files in an image folder"""
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

def get_data_loader_folder(args,input_folder, batch_size, train,  new_size=None,
                           height=256, width=256, num_workers=0, crop=True, world_size=1, rank=1 ):
    """open folder, import images and apply data transforms"""
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)

    if(args.gpus>1):

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=False) if is_distributed() else None

        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
            drop_last=True, num_workers=num_workers, pin_memory=True, sampler=sampler)
    else:

        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
            drop_last=True, num_workers=num_workers)
    
    return loader

