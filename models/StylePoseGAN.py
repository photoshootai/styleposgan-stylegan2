import os
import torch
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


from models import ANet, PNet, GNet

#import helpers
# from stylegan2 import exists, null_context, combine_contexts, default, cast_list, is_empty, raise_if_nan
# #import training helpers
# from stylegan2 import gradient_accumulate_contexts, loss_backwards, gradient_penalty, calc_pl_lengths
# #import noise-related helper functions
# from stylegan2 import noise, noise_list, 
from stylegan2 import *
#Import losses


class StylePoseGAN(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.a_net = ANet()
        self.p_net = PNet()
        self.g_net = GNet() #Contains g_net.G, g_net.D, g_net.D_aug, g_net.S


        
        self.g_lr = 1e-3
        self.d_lr = 1e-2
        self.ttur_mult = 2

        #Disabling Pytorch lightning's default optimizer
        self.automatic_optimization = False

    def forward(self, pose_map, texture_map):
        # in lightning, forward defines the prediction/inference actions
        E = self.PNet(pose_map)
        z = self.ANet(texture_map)

        gen_I = self.g_net.G(E, z)
        return gen_I #Forward pass returns the generated image
    

    # training_step defined the train loop.
    # It is independent of forward
    def training_step(self, batch, batch_idx):
        #Get optimizers
        D_opt, G_opt = self.optimizers()

        
        (S_pose_map, S_texture_map, source_I), (T_pose_map, T_texture_map, target_I) = batch #x, y = batch, so x is  the tuple, and y is the triplet

        #PNet
        S_E = self.PNet(S_pose_map)
        T_E = self.PNet(T_pose_map)

        #ANet
        S_z = self.ANet(S_texture_map)
        T_z = self.ANet(T_texture_map)


        input_noise = None #TODO: make it same as in the lucid rains repo
        I_s = self.g_net.G(z_s, input_noise, E_s) #G(E_s, z_s)            
        generated = self.g_net.G(z_s, input_noise, E_t)

        loss = get_total_loss(I_t, generated)
        
        


        

        #Training Discriminator

        D_opt.zero_grad()

        fake_output, fake_q_loss = self.g_net.D_aug(generated.clone().detach(), detach = True)
        real_output, real_q_loss = D_aug(image_batch, **aug_kwargs)

        real_output_loss = real_output
        fake_output_loss = fake_output

        if self.rel_disc_loss:
            real_output_loss = real_output_loss - fake_output.mean()
            fake_output_loss = fake_output_loss - real_output.mean()

        #What loss function to use here?
        divergence = D_loss_fn(real_output_loss, fake_output_loss)
        disc_loss = divergence

        if self.has_fq:
            quantize_loss = (fake_q_loss + real_q_loss).mean()
            self.q_loss = float(quantize_loss.detach().item())

            disc_loss = disc_loss + quantize_loss

        #TODO: Check everything below?
        if apply_gradient_penalty:
            gp = gradient_penalty(image_batch, real_output)
            self.last_gp_loss = gp.clone().detach().item()
            self.track(self.last_gp_loss, 'GP')
            disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt, loss_id = 1)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)

        self.track(self.d_loss, 'D')
        
        self.manual_backward(-1*self.d_loss)#TODO
        self.GAN.D_opt.step()        

        #Training the Generato
        G_opt.zero_grad()

        fake_output, _ = D_aug(generated_images, **aug_kwargs)
        fake_output_loss = fake_output

        real_output = None
        if G_requires_reals:
            image_batch = next(self.loader).cuda(self.rank)
            real_output, _ = D_aug(image_batch, detach = True, **aug_kwargs)
            real_output = real_output.detach()

        if self.top_k_training:
            epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
            k_frac = max(self.generator_top_k_gamma ** epochs, self.generator_top_k_frac)
            k = math.ceil(batch_size * k_frac)

            if k != batch_size:
                fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)

        loss = G_loss_fn(fake_output_loss, real_output)
        gen_loss = loss

        if apply_path_penalty:
            pl_lengths = calc_pl_lengths(w_styles, generated_images)
            avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

            if not is_empty(self.pl_mean):
                pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                if not torch.isnan(pl_loss):
                    gen_loss = gen_loss + pl_loss

        gen_loss = gen_loss / self.gradient_accumulate_every
        gen_loss.register_hook(raise_if_nan)
        backwards(gen_loss, self.GAN.G_opt, loss_id = 2)

        total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.track(self.g_loss, 'G')

        self.GAN.G_opt.step()       



        loss.backward()
        optimizer.step()
        
        self.log_dict({'g_loss': errG, 'd_loss': errD}, prog_bar=True)
        return  [self.d_loss, self.g_loss]

    #You can customize any part of training (such as the backward pass) by overriding any of the 20+ hooks found in Available Callback hooks
    # def backward(self, loss, optimizer, optimizer_idx):
    #     loss.backward()
    def training_step_end(self, losses):
        return s


    #TODO check this
    def validation_step(self):
        return super().validation_step()

    def configure_optimizers(self):
        
        # init optimizers
        generator_params = list(self.g_net.G.parameters()) + list(self.g_net.S.parameters())
        G_opt = Adam(generator_params, lr=self.g_lr, betas=(0.5, 0.9))
        D_opt = Adam(self.g_net.D.parameters(), lr=self.d_lr * self.ttur_mult, betas=(0.5, 0.9))


        #Can also do learning rate scheduling:
        #optimizers = [G_opt, D_opt]
        #lr_schedulers = {'scheduler': ReduceLROnPlateau(G_opt, ...), 'monitor': 'metric_to_track'}
        #return optimizers, lr_schedulers

        return G_opt, D_opt
