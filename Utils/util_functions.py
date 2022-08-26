"""file that contains etc util functions"""
import math
import os
import time
import re
from torch.optim import lr_scheduler
import torch.nn.init as init


def get_model_list(dirname, key):
    """used for resuming training, get's all models in base dir"""
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

def weights_init(init_type='gaussian'):
    """returns a basic weight init function"""
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    return init_fun

def get_scheduler(optimizer,args, iterations=-1):
    """create a pytorch scheduler"""
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_policy_step_size,
                                        gamma=.5, last_epoch=iterations)
    return scheduler


class Timer:
    """simple class to reporting iteration timings"""
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))
        