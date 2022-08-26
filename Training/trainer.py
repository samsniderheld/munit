"""The base trainer class the loads the images,
does the forward pass, calculates losses, and
the backward pass"""
import os

import torch
from torch import nn
from torch.autograd import Variable

from Model.ada_in_ae import AdaINGen
from Model.ms_img_disc import MsImageDis

from Utils.util_functions import weights_init, get_scheduler, get_model_list

# Parse torch version for autocast
# ######################################################
version = torch.__version__
version = tuple(int(n) for n in version.split('.')[:-1])
#has_autocast = version >= (1, 6)
has_autocast = False
# ######################################################

class MUNIT_Trainer(nn.Module):
    """See above"""
    def __init__(self, args):
        super(MUNIT_Trainer, self).__init__()

        self.args = args
        
        lr = 0.0001

        # Initiate the networks
        self.generator_a = AdaINGen(3, args)  # auto-encoder for domain a
        self.generator_b = AdaINGen(3, args)  # auto-encoder for domain b
        self.dicrimininator_a = MsImageDis(3, args)  # discriminator for domain a
        self.discriminator_b = MsImageDis(3, args)  # discriminator for domain b
        self.instance_norm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = 8

        # fix the noise used in sampling
        display_size = int(args.display_size)
        self.style_vector_a = torch.randn(display_size, self.style_dim, 1, 1).cuda(self.args.gpu)
        self.style_vector_b = torch.randn(display_size, self.style_dim, 1, 1).cuda(self.args.gpu)

        # Setup the optimizers
        beta1 = .5
        beta2 = .999
        weight_decay = 0.0001
        discriminator_parameters = list(self.dicrimininator_a.parameters()) + list(self.discriminator_b.parameters())
        generator_parameters = list(self.generator_a.parameters()) + list(self.generator_b.parameters())
        self.discrimator_optimizer = torch.optim.Adam([p for p in discriminator_parameters if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        self.generator_optimizer = torch.optim.Adam([p for p in generator_parameters if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        self.discirimiator_scheduler = get_scheduler(self.discrimator_optimizer, args)
        self.generator_scheduler = get_scheduler(self.generator_optimizer, args)

        # Network weight initialization
        self.apply(weights_init('kaiming'))
        self.dicrimininator_a.apply(weights_init('gaussian'))
        self.discriminator_b.apply(weights_init('gaussian'))

    def recon_criterion(self, x, target):
        """mean aboslute error"""
        return torch.mean(torch.abs(x - target))

    def forward(self, x_a, x_b):
        """definine the forward pass"""
        self.eval()
        style_vector_a = Variable(self.style_vector_a)
        style_vector_b = Variable(self.style_vector_b)
        content_a, style_a_fake = self.generator_a.encode(x_a)
        content_b, style_b_fake = self.generator_b.encode(x_b)
        x_b_2_a = self.generator_a.decode(content_b, style_vector_a)
        x_a_2_b = self.generator_b.decode(content_a,style_vector_b)
        self.train()
        return x_a_2_b, x_b_2_a


    def __aux_gen_update(self, x_a, x_b, args):
        """forward pass and loss estimation for the generator - function to allow fp16"""
        self.generator_optimizer.zero_grad()
        style_vector_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda(self.args.gpu))
        style_vector_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda(self.args.gpu))
        # encode
        content_a, style_vector_a_prime = self.generator_a.encode(x_a)
        content_b,style_vector_b_prime = self.generator_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.generator_a.decode(content_a, style_vector_a_prime)
        x_b_recon = self.generator_b.decode(content_b,style_vector_b_prime)
        # decode (cross domain)
        x_b_2_a = self.generator_a.decode(content_b, style_vector_a)
        x_a_2_b = self.generator_b.decode(content_a,style_vector_b)
        # encode again
        content_b_recon, style_vector_a_recon = self.generator_a.encode(x_b_2_a)
        content_a_recon,style_vector_b_recon = self.generator_b.encode(x_a_2_b)

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_style_vector_a = self.recon_criterion(style_vector_a_recon, style_vector_a)
        self.loss_gen_recon_style_vector_b = self.recon_criterion(style_vector_b_recon,style_vector_b)
        self.loss_gen_recon_content_a = self.recon_criterion(content_a_recon, content_a)
        self.loss_gen_recon_content_b = self.recon_criterion(content_b_recon, content_b)
        
        # GAN loss
        self.loss_generator_adv_a = self.dicrimininator_a.calc_gen_loss(x_b_2_a)
        self.loss_generator_adv_b = self.discriminator_b.calc_gen_loss(x_a_2_b)
        
        # total loss
        loss_gen_total =  args.gan_w * self.loss_generator_adv_a + \
                              args.gan_w * self.loss_generator_adv_b + \
                              args.recon_x_w * self.loss_gen_recon_x_a + \
                              args.recon_s_w * self.loss_gen_recon_style_vector_a + \
                              args.recon_c_w * self.loss_gen_recon_content_a + \
                              args.recon_x_w * self.loss_gen_recon_x_b + \
                              args.recon_s_w * self.loss_gen_recon_style_vector_b + \
                              args.recon_c_w * self.loss_gen_recon_content_b
        
        return loss_gen_total


    def gen_update(self, x_a, x_b, args):
        """forward and backward pass for the generator"""
        if False: # has_autocast: # NEED TO CHECK: gen autocast is not working!  RuntimeError: Expected running_mean to have type Half but got Float
            with torch.cuda.amp.autocast(enabled=True):
                self.loss_gen_total = self.__aux_gen_update(x_a, x_b, args)
        else:
            self.loss_gen_total = self.__aux_gen_update(x_a, x_b, args)

        self.loss_gen_total.backward()
        self.generator_optimizer.step()

    def sample(self, x_a, x_b):
        """get samples from all networks"""
        self.eval()
        style_vector_a1 = Variable(self.style_vector_a)
        style_vector_b1 = Variable(self.style_vector_b)
        style_vector_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda(self.args.gpu))
        style_vector_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda(self.args.gpu))
        x_a_recon, x_b_recon, x_b_2_a1, x_b_2_a2, x_a_2_b1, x_a_2_b2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            content_a, style_a_fake = self.generator_a.encode(x_a[i].unsqueeze(0))
            content_b, style_b_fake = self.generator_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.generator_a.decode(content_a, style_a_fake))
            x_b_recon.append(self.generator_b.decode(content_b, style_b_fake))
            x_b_2_a1.append(self.generator_a.decode(content_b, style_vector_a1[i].unsqueeze(0)))
            x_b_2_a2.append(self.generator_a.decode(content_b, style_vector_a2[i].unsqueeze(0)))
            x_a_2_b1.append(self.generator_b.decode(content_a,style_vector_b1[i].unsqueeze(0)))
            x_a_2_b2.append(self.generator_b.decode(content_a,style_vector_b2[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_b_2_a1, x_b_2_a2 = torch.cat(x_b_2_a1), torch.cat(x_b_2_a2)
        x_a_2_b1, x_a_2_b2 = torch.cat(x_a_2_b1), torch.cat(x_a_2_b2)
        self.train()
        return x_a, x_a_recon, x_a_2_b1, x_a_2_b2, x_b, x_b_recon, x_b_2_a1, x_b_2_a2

    def __aux_dis_update(self, x_a, x_b, args):
        """forward pass and loss estimation for the generator - function to allow fp16"""
        self.discrimator_optimizer.zero_grad()
        style_vector_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda(args.gpu))
        style_vector_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda(args.gpu))
        # encode
        content_a, _ = self.generator_a.encode(x_a)
        content_b, _ = self.generator_b.encode(x_b)
        # decode (cross domain)
        x_b_2_a = self.generator_a.decode(content_b, style_vector_a)
        x_a_2_b = self.generator_b.decode(content_a,style_vector_b)
        # D loss
        self.loss_dicrimininator_a = self.dicrimininator_a.calc_dis_loss(x_b_2_a.detach(), x_a)
        self.loss_discriminator_b = self.discriminator_b.calc_dis_loss(x_a_2_b.detach(), x_b)
        
        return args.gan_w * self.loss_dicrimininator_a + args.gan_w * self.loss_discriminator_b

    def dis_update(self, x_a, x_b, args):
        """forward and backward pass for the discriminator"""
        if has_autocast:
            with torch.cuda.amp.autocast(enabled=True):
                self.loss_dis_total = self.__aux_dis_update(x_a, x_b, args)
        else:
            self.loss_dis_total = self.__aux_dis_update(x_a, x_b, args)
        self.loss_dis_total.backward()
        self.discrimator_optimizer.step()

    def update_learning_rate(self):
        """update the learning rates"""
        if self.discirimiator_scheduler is not None:
            self.discirimiator_scheduler.step()
        if self.generator_scheduler is not None:
            self.generator_scheduler.step()

    def resume(self, checkpoint_dir, args):
        """function to resume training"""
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.generator_a.load_state_dict(state_dict['a'])
        self.generator_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dicrimininator_a.load_state_dict(state_dict['a'])
        self.discriminator_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.discrimator_optimizer.load_state_dict(state_dict['dis'])
        self.generator_optimizer.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.discirimiator_scheduler = get_scheduler(self.discrimator_optimizer, args, iterations)
        self.generator_scheduler = get_scheduler(self.generator_optimizer, args, iterations)
        print(f'Resume from iteration {iterations:08d}')
        return iterations

    def save(self, snapshot_dir, iterations):
        """save the models"""
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, f'gen_{(iterations + 1):08d}.pt')
        dis_name = os.path.join(snapshot_dir, f'dis__{(iterations + 1):08d}.pt')
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.generator_a.state_dict(), 'b': self.generator_b.state_dict()}, gen_name)
        torch.save({'a': self.dicrimininator_a.state_dict(), 'b': self.discriminator_b.state_dict()}, dis_name)
        torch.save({'gen': self.generator_optimizer.state_dict(), 'dis': self.discrimator_optimizer.state_dict()}, opt_name)
