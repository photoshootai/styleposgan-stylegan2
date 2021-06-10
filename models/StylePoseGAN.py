from losses.FaceIDLoss import FaceIDLoss
from losses.loss import get_patch_loss
import torch
import torch
import pytorch_lightning as pl


from models import ANet, PNet, GNet
from stylegan2 import *


from losses import get_face_id_loss, gan_g_loss, gan_d_loss, get_l1_loss, get_perceptual_vgg_loss, VGG16Perceptual, DPatch

class StylePoseGAN(pl.LightningModule):

    def __init__(self, image_size, batch_size, g_lr=2e-3, d_lr=2e-3, ttur_mult=2, latent_dim=2048,
                 network_capacity=16, attn_layers=(1, 2, 3, 4), mtcnn_crop_size=160, pl_reg=True):  # changed attention from list to tuple since mutable default args cause problems
        super().__init__()

        #Attributes
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.ttur_mult = ttur_mult
        self.image_size = image_size
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.network_capacity = network_capacity
        self.attn_layers = attn_layers
        self.mtcnn_crop_size = mtcnn_crop_size
    
        self.a_net = ANet(im_chan=3).train()
        self.p_net = PNet(im_chan=3).train()
        self.g_net = GNet(image_size=image_size, latent_dim=latent_dim).train() #Contains g_net.G, g_net.D, g_net.D_aug, g_net.S

        #Loss calculation models
        self.vgg16_perceptual_model = VGG16Perceptual(requires_grad=False).eval()
        self.face_id_loss = FaceIDLoss(self.mtcnn_crop_size, requires_grad = False, device=self.device).eval()
        self.d_patch = DPatch() # Needs to be on same device as data!


        #Log hyperparameters
        self.save_hyperparameters()

        #print('Device Rank', self.global_rank)  # should be 0 on main, > 0 on other gpus/tpus/cpus
        print("StylePoseGAN module initialized with: ", {"image_size": self.image_size, "batch_size": self.batch_size, "latent_dim": self.latent_dim} )

    
    def training_step(self, batch, batch_idx, optimizer_idx):
     
        #Total Loss that needs to be maximized. The only GAN loss here is -[log(D(x)) + log(1-D(G(z)))] for the respective args
        # Weights
        weight_l1 =1
        weight_vgg = 1
        weight_face = 1
        weight_gan = 1
        weight_patch = 1

        (I_s, S_pose_map, S_texture_map), (I_t, T_pose_map) = batch #x, y = batch, so x is  the tuple, and y is the triplet
        
        # PNet
        E_s = self.p_net(S_pose_map)
        E_t = self.p_net(T_pose_map)

        # ANet
        z_s = self.a_net(S_texture_map)  # needs norm

        #GNet        
        input_noise = torch.randn(I_s.shape[0], self.image_size, self.image_size, 1).type_as(z_s)
        # print("GNet Forward Pass details: ", {"z_s_repeated": z_s.repeat(1, 5, 1).size(), "E_s": E_s.size(), "E_t": E_t.size(), "input_noise": input_noise.size()})
        
        #  Repeat z num_layer times
        I_dash_s = self.g_net.G(z_s.repeat(1, 5, 1), input_noise, E_s) #G(E_s, z_s)
        I_dash_s_to_t = self.g_net.G(z_s.repeat(1, 5, 1), input_noise, E_t)


        ########
        #Optimizing D, DPatch with max_opt
        ######
        if optimizer_idx == 0:
            
            #Detaching generated when passing to Discriminators inside gan_d_loss
            gan_loss_1_d = weight_gan * gan_d_loss(I_dash_s, I_s, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device)
            gan_loss_2_d = weight_gan * gan_d_loss(I_dash_s_to_t, I_t, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device)
            patch_loss = weight_patch * get_patch_loss(I_dash_s_to_t, I_t, self.d_patch)

            l_total = gan_loss_1_d + gan_loss_2_d + (-1)*patch_loss #patch loss * (-1) because it needs to be maximized
            self.log_dict({"disc_loss_total": l_total, "patch_loss_d": patch_loss, "d_loss_1": gan_loss_1_d, "d_loss_2": gan_loss_2_d }, on_epoch=True)
            l_result =  l_total

        ########
        #Optimizing G, A, P
        #######
        if optimizer_idx == 1:

            rec_loss_1 =  weight_l1 * get_l1_loss(I_dash_s, I_s) + \
                            weight_vgg * get_perceptual_vgg_loss(self.vgg16_perceptual_model, I_dash_s, I_s) + \
                            weight_face * get_face_id_loss(I_dash_s, I_s, self.face_id_loss, crop_size=self.mtcnn_crop_size)

            rec_loss_2 =  weight_l1 * get_l1_loss(I_dash_s_to_t ,I_t) + \
                            weight_vgg * get_perceptual_vgg_loss(self.vgg16_perceptual_model,I_dash_s_to_t, I_t) + \
                            weight_face * get_face_id_loss(I_dash_s_to_t, I_t, self.face_id_loss, crop_size=self.mtcnn_crop_size)

        
            gan_loss_1_g = weight_gan * gan_g_loss(I_dash_s, I_s, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device )
            gan_loss_2_g = weight_gan * gan_g_loss(I_dash_s_to_t, I_t, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device)
            patch_loss = weight_patch * get_patch_loss(I_dash_s_to_t, I_t, self.d_patch)

            #This is the total loss that needs to be minimized. The only GAN loss here is -log(D(G(z)) times two for the two reconstruction losses
            l_total = rec_loss_1 + rec_loss_2 + gan_loss_1_g + gan_loss_2_g + patch_loss
            self.log_dict({"gen_loss_total": l_total, "rec_loss_1": rec_loss_1, "rec_loss_2": rec_loss_2, "g_loss_1": gan_loss_1_g, "g_loss_2": gan_loss_2_g, "patch_loss_g": patch_loss }, on_epoch=True)
            l_result =  l_total

        return l_result
          
    def configure_optimizers(self):

        param_to_min = list(self.a_net.parameters()) + list(self.p_net.parameters()) + list(self.g_net.G.parameters())
        param_to_max = list(self.g_net.D.parameters()) + list(self.d_patch.parameters())
        min_opt = torch.optim.Adam(param_to_min, lr=self.g_lr, betas=(0.0, 0.99))
        max_opt = torch.optim.Adam(param_to_max, lr=self.d_lr, betas=(0.0, 0.99))

        #Can also do learning rate scheduling:
        #optimizers = [G_opt, D_opt]
        #lr_schedulers = {'scheduler': ReduceLROnPlateau(G_opt, ...), 'monitor': 'metric_to_track'}
        #return optimizers, lr_schedulers

        return [max_opt, min_opt], []

   