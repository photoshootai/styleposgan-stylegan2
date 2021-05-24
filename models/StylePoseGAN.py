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
from losses import get_total_loss

class StylePoseGAN(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.a_net = ANet()
        self.p_net = PNet()
        self.g_net = GNet() #Contains g_net.G, g_net.D, g_net.D_aug, g_net.S


        
        self.g_lr = 1e-3
        self.d_lr = 1e-2

        self.main_lr = 2e-3 #TODO: Check this
        self.ttur_mult = 2

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
        I_dash_s = self.g_net.G(z_s, input_noise, E_s) #G(E_s, z_s)            
        I_dash_s_to_t = self.g_net.G(z_s, input_noise, E_t)

                
        #Backward pass on different losses
        #Training Discriminator

        D_opt.zero_grad()


        #Opt step for g_opt and d_opt


        # Calculate Moving Averages

        

       
        
        
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
        G_opt = Adam(self.g_net.G.parameters(), lr=self.g_lr, betas=(0.5, 0.9))
        D_opt = Adam(self.g_net.D.parameters(), lr=self.d_lr * self.ttur_mult, betas=(0.5, 0.9))

        all_params = list(self.a_net.parameters() + list(self.p_net.parameters()) + list(self.g_net.parameters()))
        Main_opt = Adam(all_params,
        #Can also do learning rate scheduling:
        #optimizers = [G_opt, D_opt]
        #lr_schedulers = {'scheduler': ReduceLROnPlateau(G_opt, ...), 'monitor': 'metric_to_track'}
        #return optimizers, lr_schedulers

        return G_opt, D_opt
