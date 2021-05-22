import os
import torch
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


#Import losses


class StylePoseGAN(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.a_net = ANet()
        self.p_net = PNet()
        self.g_net = GNet() #Contains g_net.G, g_net.D, g_net.D_aug
    

    def forward(self, pose_map, texture_map):
        # in lightning, forward defines the prediction/inference actions
        E = self.PNet(pose_map)
        z = self.ANet(texture_map)

        gen_I = self.g_net.G(E, z)
        return gen_I #Forward pass returns the generated image
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        
        (S_pose_map, S_texture_map, source_I), (T_pose_map, T_texture_map, target_I) = batch #x, y = batch, so x is  the tuple, and y is the triplet
        # x = x.view(x.size(0), -1) 
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        
        
        #PNet
        S_E = self.PNet(S_pose_map)
        T_E = self.PNet(T_pose_map)

        #ANet
        S_z = self.ANet(S_texture_map)
        T_z = self.ANet(T_texture_map)


        I_s = G(E_s, z_s)            
        generated = G(E_t, z_s)

        loss = get_total_loss(I_t, generated)
        
        
        #Training Discriminator

        #Training the Generator



        loss.backward()
        optimizer.step()
        
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    #You can customize any part of training (such as the backward pass) by overriding any of the 20+ hooks found in Available Callback hooks
    # def backward(self, loss, optimizer, optimizer_idx):
    #     loss.backward()

    #TODO check this
    def validation_step(self):
        return super().validation_step()

    def configure_optimizers(self):
        #TODO
        pass
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # return optimizer

        #Need to enable custom optimizer to get custom loss???