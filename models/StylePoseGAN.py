from losses.loss import gan_d_loss, get_face_id_loss, get_l1_loss, get_perceptual_vgg_loss
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
from losses import gan_g_loss

class StylePoseGAN(pl.LightningModule):

    def __init__(self, image_size, g_lr=2e-3, d_lr=2e-3, ttur_mult=2, latent_dim=2048, network_capacity=16, attn_layers=[1, 2, 3, 4]):
        super().__init__()
        self.a_net = ANet()
        self.p_net = PNet()
        self.g_net = GNet(image_size=image_size, latent_dim=latent_dim ) #Contains g_net.G, g_net.D, g_net.D_aug, g_net.S

        self.d_patch = None #Implement D_Patch

        self.d_lr = d_lr
        self.g_lr = g_lr
        self.image_size = image_size

        #Disabling Pytorch lightning's default optimizer
        self.automatic_optimization = False

    def forward(self, pose_map, texture_map):
        # in lightning, forward defines the prediction/inference actions
        E = self.PNet(pose_map)
        z = self.ANet(texture_map)

        gen_I = self.g_net.G(E, z)
        return gen_I #Forward pass returns the generated image
    

    # training_step defined the train loop. # It is independent of forward
    def training_step(self, batch, batch_idx):
        apply_path_penalty=False
        avg_pl_length = 0

        #Weights
        weight_l1 =1
        weight_vgg = 1
        weight_face = 1
        weight_gan = 1

        #Get optimizers
        min_opt, max_opt = self.optimizers()

        min_opt.zero_grad()
        max_opt.zero_grad()

        (I_s, S_pose_map, S_texture_map, ), (I_t, T_pose_map, T_texture_map) = batch #x, y = batch, so x is  the tuple, and y is the triplet

        #PNet
        E_s = self.PNet(S_pose_map)
        E_t = self.PNet(T_pose_map)

        #ANet
        z_s = self.ANet(S_texture_map)
        z_t = self.ANet(T_texture_map)


        input_noise = torch.FloatTensor(batch.size()[0], self.image_size, self.image_size, 1).uniform_(0., 1.).cuda(device)
        I_dash_s = self.g_net.G(z_s, input_noise, E_s) #G(E_s, z_s)            
        I_dash_s_to_t = self.g_net.G(z_s, input_noise, E_t)
        
        #Need to detach at the top level 
        rec_loss_1 =  weight_l1 * get_l1_loss(I_dash_s, I_s) + \
                      weight_vgg * get_perceptual_vgg_loss(I_dash_s, I_s) + \
                      weight_face * get_face_id_loss(I_dash_s, I_s)
                                
        rec_loss_2 =  weight_l1 * get_l1_loss(I_dash_s_to_t ,I_t) + \
                      weight_vgg * get_perceptual_vgg_loss(I_dash_s_to_t, I_t) + \
                      weight_face * get_face_id_loss(I_dash_s_to_t, I_t) 

        gan_loss_1_g = gan_g_loss(I_dash_s, I_s, self.g_net.G, self.g_net.D, self.g_net.D_aug)
        gan_loss_2_g = gan_g_loss(I_dash_s_to_t, I_t, self.g_net.G, self.g_net.D, self.g_net.D_aug)

        gan_loss_1_d = gan_d_loss(I_dash_s.clone().detach(), I_s, self.g_net.G, self.g_net.D, self.g_net.D_aug)
        gan_loss_2_d = gan_d_loss(I_dash_s_to_t.clone().detach(), I_t, self.g_net.G, self.g_net.D, self.g_net.D_aug)

        """
        L_GAN has two parts: 
        1. the D_loss=log(D(x)) + log(1-D(G(z))) which needs to maximized
        2. the G_loss=log(D(G(z))) obtained from the -log(D) trick, which also needs to be maximized

        The make two cases of the Loss_total:
        One contains L_GAN = D_loss and that needs to be maximized w.r.t D and DPath
        One contains L_GAN = G_loss that needs to be maximized w.r.t ANet, PNet and GNet.G

        So basically: 
        D and DPatch maximize the total loss but with L_GAN as D_loss
        ANet, PNet, and GNet.G minimize the total loss but with L_GAN as G_loss

        This makes StylePoseGAN conform to the traditional def. of a GAN 
        with the {ANet, PNet, GNet.G} being the "G" being minimized, 
        and {D, DPatch} being the "D" being maximized
        """

        #This is the total loss that needs to be minimized. The only GAN loss here is -log(D(G(z)) times two for the two reconstruction losses
        l_total_to_min = rec_loss_1 + rec_loss_2 + gan_loss_1_g + gan_loss_2_g 
        
       #Total Loss that needs to be maximized. The only GAN loss here is -[log(D(x)) + log(1-D(G(z)))] for the respective args
        l_total_to_max = (-1)*rec_loss_1 + (-1)*rec_loss_2 + gan_loss_1_d + gan_loss_2_d
        
        min_opt.zero_grad()
        l_total_to_min.backward()
        min_opt.step()
                    
        max_opt.zero_grad()
        l_total_to_max.backward()
        max_opt.step()
        
        # Calculate Moving Averages
        if apply_path_penalty and not np.isnan(avg_pl_length):
                    self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
                    self.track(self.pl_mean, 'PL')

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (l_total_to_min, l_total_to_max)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException
        
        
        self.log_dict({'generation_loss': l_total_to_min, 'disc_loss': l_total_to_max}, prog_bar=True)
        return  {'l_total_to_min': l_total_to_min, 'l_total_to_max': l_total_to_max}


    #You can customize any part of training (such as the backward pass) by overriding any of the 20+ hooks found in Available Callback hooks
    # def backward(self, loss, optimizer, optimizer_idx):
    #     loss.backward()
    # def training_step_end(self, losses):
    #     return


    #TODO check this
    def validation_step(self, batch, batch_idx):

        weight_l1 =1
        weight_vgg = 1
        weight_face = 1
        weight_gan = 1

        (S_pose_map, S_texture_map, I_s), (T_pose_map, T_texture_map, I_t) = batch #x, y = batch, so x is  the tuple, and y is the triplet 

        #PNet
        E_s = self.PNet(S_pose_map)
        E_t = self.PNet(T_pose_map)

        #ANet
        z_s = self.ANet(S_texture_map)
        z_t = self.ANet(T_texture_map)


        input_noise = None #TODO: make it same as in the lucid rains repo
        I_dash_s = self.g_net.G(z_s, input_noise, E_s) #G(E_s, z_s)            
        I_dash_s_to_t = self.g_net.G(z_s, input_noise, E_t)


        #Need to detach at the top level 
        rec_loss_1 =  weight_l1 * get_l1_loss(I_dash_s, I_s) + \
                      weight_vgg * get_perceptual_vgg_loss(I_dash_s, I_s) + \
                      weight_face * get_face_id_loss(I_dash_s, I_s)
                                
        rec_loss_2 =  weight_l1 * get_l1_loss(I_dash_s_to_t ,I_t) + \
                      weight_vgg * get_perceptual_vgg_loss(I_dash_s_to_t, I_t) + \
                      weight_face * get_face_id_loss(I_dash_s_to_t, I_t) 

        gan_loss_1_g = gan_g_loss(I_dash_s, I_s, self.g_net.G, self.g_net.D, self.g_net.D_aug)
        gan_loss_2_g = gan_g_loss(I_dash_s_to_t, I_t, self.g_net.G, self.g_net.D, self.g_net.D_aug)

        gan_loss_1_d = gan_d_loss(I_dash_s.clone().detach(), I_s, self.g_net.G, self.g_net.D, self.g_net.D_aug)
        gan_loss_2_d = gan_d_loss(I_dash_s_to_t.clone().detach(), I_t, self.g_net.G, self.g_net.D, self.g_net.D_aug)

        """
        L_GAN has two parts: 
        1. the D_loss=log(D(x)) + log(1-D(G(z))) which needs to maximized
        2. the G_loss=log(D(G(z))) obtained from the -log(D) trick, which also needs to be maximized

        The make two cases of the Loss_total:
        One contains L_GAN = D_loss and that needs to be maximized w.r.t D and DPath
        One contains L_GAN = G_loss that needs to be maximized w.r.t ANet, PNet and GNet.G

        So basically: 
        D and DPatch maximize the total loss but with L_GAN as D_loss
        ANet, PNet, and GNet.G minimize the total loss but with L_GAN as G_loss

        This makes StylePoseGAN conform to the traditional def. of a GAN 
        with the {ANet, PNet, GNet.G} being the "G" being minimized, 
        and {D, DPatch} being the "D" being maximized
        """

        #This is the total loss that needs to be minimized. The only GAN loss here is -log(D(G(z)) times two for the two reconstruction losses
        l_total_to_min = rec_loss_1 + rec_loss_2 + gan_loss_1_g + gan_loss_2_g 
        
       #Total Loss that needs to be maximized. The only GAN loss here is -[log(D(x)) + log(1-D(G(z)))] for the respective args
        l_total_to_max = (-1)*rec_loss_1 + (-1)*rec_loss_2 + gan_loss_1_d + gan_loss_2_d
        
        return  {'l_total_to_min': l_total_to_min, 'l_total_to_max': l_total_to_max}

    def configure_optimizers(self):
        
        # init optimizers
        # G_opt = Adam(self.g_net.G.parameters(), lr=self.g_lr, betas=(0.5, 0.9))
        # D_opt = Adam(self.g_net.D.parameters(), lr=self.d_lr * self.ttur_mult, betas=(0.5, 0.9))

        param_to_min = list(self.a_net.parameters()) + list(self.p_net.parameters()) + list(self.g_net.G.parameters())
        param_to_max = list(self.g_net.D.parameters()) #+ list(self.d_patch.parameters())
        min_opt = Adam(param_to_min, lr=self.g_lr, betas=(0.5, 0.9))
        max_opt = Adam(param_to_max, lr=self.d_lr, betas=(0.5, 0.9))
        
        #Can also do learning rate scheduling:
        #optimizers = [G_opt, D_opt]
        #lr_schedulers = {'scheduler': ReduceLROnPlateau(G_opt, ...), 'monitor': 'metric_to_track'}
        #return optimizers, lr_schedulers

        return min_opt, max_opt

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("StylePoseGAN")
        parser.add_argument("--latent_dim", type=int, default=2048)
        parser.add_argument("--network_capacity", type=int, default=16)
        parser.add_argument("--attn_layers", type=list, default=[1,2, 3, 4])

        return parent_parser