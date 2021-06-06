from torch._C import device
from losses.FaceIDLoss import FaceIDLoss
from losses.loss import get_patch_loss
import os
import torch
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import pytorch_lightning as pl


from models import ANet, PNet, GNet
from stylegan2 import *


from losses import get_face_id_loss, gan_g_loss, gan_d_loss, get_l1_loss, get_perceptual_vgg_loss, VGG16Perceptual, DPatch

class StylePoseGAN(pl.LightningModule):

    def __init__(self, image_size, batch_size, g_lr=2e-3, d_lr=2e-3, ttur_mult=2, latent_dim=2048,
                 network_capacity=16, attn_layers=(1, 2, 3, 4), mtcnn_crop_size=160, steps=0, pl_reg=True):  # changed attention from list to tuple since mutable default args cause problems
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

        #TODO: FEATURE
        #Path Penalty related attrs.  
        # self.steps = steps
        # self.pl_reg = pl_reg
        # self.pl_mean = None #TODO: what should this be set to if at all
        # self.pl_length_ma = EMA(0.99)

        #Loss calculation models
        self.vgg16_perceptual_model = VGG16Perceptual(requires_grad=False).eval()
        self.face_id_loss = FaceIDLoss(self.mtcnn_crop_size, requires_grad = False, device=self.device).eval()
        self.d_patch = DPatch() # Needs to be on same device as data!

        #Disabling Pytorch lightning's default optimizer
        self.automatic_optimization = False

<<<<<<< HEAD
        # print('Device Rank', self.global_rank)  # should be 0 on main, > 0 on other gpus/tpus/cpus
=======
        #Log hyperparameters
        self.save_hyperparameters()

        print('Device Rank', self.global_rank)  # should be 0 on main, > 0 on other gpus/tpus/cpus
>>>>>>> dev
        print("StylePoseGAN module initialized with: ", {"image_size": self.image_size, "batch_size": self.batch_size, "latent_dim": self.latent_dim} )

    def get_forward_results(self, batch):
        """
        Separate repeated code from validation and training steps into different function for easier update/debugging
        """


        # normalize = lambda t: (t - torch.min(t)) / (torch.max(t) - torch.min(t))

        (I_s, S_pose_map, S_texture_map), (I_t, T_pose_map, _) = batch #x, y = batch, so x is  the tuple, and y is the triplet

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



        return (I_dash_s, I_dash_s_to_t, I_s, I_t)
       

    def training_step(self, batch, batch_idx):
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
        #Total Loss that needs to be maximized. The only GAN loss here is -[log(D(x)) + log(1-D(G(z)))] for the respective args
        # Weights
        weight_l1 =1
        weight_vgg = 1
        weight_face = 1
        weight_gan = 1
        weight_patch = 1

        (I_dash_s, I_dash_s_to_t, I_s, I_t) = self.get_forward_results(batch)

        # Get optimizers
        steps = batch_idx
        min_opt, max_opt = self.optimizers()



        #Loss not dependant on D and G, can be obtained before hand
        rec_loss_1 =  weight_l1 * get_l1_loss(I_dash_s, I_s) + \
                      weight_vgg * get_perceptual_vgg_loss(self.vgg16_perceptual_model, I_dash_s, I_s) + \
                      weight_face * get_face_id_loss(I_dash_s, I_s, self.face_id_loss, crop_size=self.mtcnn_crop_size)

        rec_loss_2 =  weight_l1 * get_l1_loss(I_dash_s_to_t ,I_t) + \
                      weight_vgg * get_perceptual_vgg_loss(self.vgg16_perceptual_model,I_dash_s_to_t, I_t) + \
                      weight_face * get_face_id_loss(I_dash_s_to_t, I_t, self.face_id_loss, crop_size=self.mtcnn_crop_size)
        

        ########
        #Optimizing D, DPatch
        #######

        #Detaching generated when passing to Discriminators inside gan_d_loss
        gan_loss_1_d = weight_gan * gan_d_loss(I_dash_s, I_s, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device)
        gan_loss_2_d = weight_gan * gan_d_loss(I_dash_s_to_t, I_t, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device)
        patch_loss = weight_patch * get_patch_loss(I_dash_s_to_t, I_t, self.d_patch)

        l_total_to_max = (-1)*rec_loss_1 + (-1)*rec_loss_2 + gan_loss_1_d + gan_loss_2_d + (-1)*patch_loss

        max_opt.zero_grad()
        self.manual_backward(l_total_to_max, retain_graph=True)
        max_opt.step()


        ########
        #Optimizing G, A, P
        #######
        gan_loss_1_g = weight_gan * gan_g_loss(I_dash_s, I_s, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device )
        gan_loss_2_g = weight_gan * gan_g_loss(I_dash_s_to_t, I_t, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device)
        patch_loss = weight_patch * get_patch_loss(I_dash_s_to_t, I_t, self.d_patch)

        #This is the total loss that needs to be minimized. The only GAN loss here is -log(D(G(z)) times two for the two reconstruction losses
        l_total_to_min = rec_loss_1 + rec_loss_2 + gan_loss_1_g + gan_loss_2_g + patch_loss

        min_opt.zero_grad()
        self.manual_backward(l_total_to_min)
        min_opt.step()
       
        named_losses = {
            'rl1': rec_loss_1,
            'rl2': rec_loss_2,
            'gl1d': gan_loss_1_d,  # huge numbers, grows very fast, needs scaling(?)
            'gl2d': gan_loss_2_d,  # ^
            'gl1g': gan_loss_1_g,  # either very large numbers or 0
            'gl2g': gan_loss_2_g,  # ^
            # 'pl': patch_loss
        }

        # print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in named_losses.items()) + "}")
        
        # if steps % 10 == 0 and steps > 20000:
        #     self.g_net.EMA()

        #Parameter Averaging
        # if steps <= 25000 and steps % 1000 == 2:
        #     self.g_net.reset_parameter_averaging()

        # save from NaN errors
        if any(torch.isnan(l) for l in (l_total_to_min, l_total_to_max)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        self.log_dict({'gen_loss': l_total_to_min, 'disc_loss': l_total_to_max, **named_losses}, prog_bar=True, on_epoch=True)
        #Commented out 'I_dash_s': I_dash_s, 'I_dash_s_to_t': I_dash_s_to_t} from the above returned dictionary because:
        # "If you are returning the batch and predictions from training_step (or validation_step) they will be accumulated to be passed to training_step_end and validation_step_end respectively, 
        # which could be causing the OOM errors"

        return  {'gen_loss': l_total_to_min, 'disc_loss': l_total_to_max}
          
    # def training_epoch_end(self, outputs):
    #     steps = len(outputs)
    #     generated_s_to_t = outputs[-1]['I_dash_s_to_t']
    #     generated_s_dash = outputs[-1]['I_dash_s']

    #     if not (os.path.isdir('./test_ims')):
    #         os.mkdir('./test_ims')
    #     save_image(generated_s_to_t, f'./test_ims/s_to_t_step_{steps}.jpg')
    #     save_image(generated_s_dash, f'./test_ims/s_dash_{steps}.jpg')
        # print(f'Saved I_dash_s_to_t at step {steps} to test_ims dir.')
    
    #TODO check this
    def validation_step(self, batch, batch_idx):
          # Weights
        weight_l1 =1
        weight_vgg = 1
        weight_face = 1
        weight_gan = 1
        weight_patch = 1


        (I_dash_s, I_dash_s_to_t, I_s, I_t) = self.get_forward_results(batch)

        # Get optimizers
        steps = batch_idx
        min_opt, max_opt = self.optimizers()



        #Loss not dependant on D and G, can be obtained before hand
        rec_loss_1 =  weight_l1 * get_l1_loss(I_dash_s, I_s) #+ \
                      #weight_vgg * get_perceptual_vgg_loss(self.vgg16_perceptual_model, I_dash_s, I_s) + \
                      #weight_face * get_face_id_loss(I_dash_s, I_s, self.face_id_loss, crop_size=self.mtcnn_crop_size)

        rec_loss_2 =  weight_l1 * get_l1_loss(I_dash_s_to_t ,I_t)# + \
                    #   weight_vgg * get_perceptual_vgg_loss(self.vgg16_perceptual_model,I_dash_s_to_t, I_t) + \
                    #   weight_face * get_face_id_loss(I_dash_s_to_t, I_t, self.face_id_loss, crop_size=self.mtcnn_crop_size)
        

        #Detaching generated when passing to Discriminators inside gan_d_loss
        gan_loss_1_d = weight_gan * gan_d_loss(I_dash_s, I_s, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device)
        gan_loss_2_d = weight_gan * gan_d_loss(I_dash_s_to_t, I_t, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device)
        #patch_loss = weight_patch * get_patch_loss(I_dash_s_to_t, I_t, self.d_patch)

        l_total_to_max = (-1)*rec_loss_1 + (-1)*rec_loss_2 + gan_loss_1_d + gan_loss_2_d# + (-1)*patch_loss

        gan_loss_1_g = weight_gan * gan_g_loss(I_dash_s, I_s, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device )
        gan_loss_2_g = weight_gan * gan_g_loss(I_dash_s_to_t, I_t, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device)
        #patch_loss = weight_patch * get_patch_loss(I_dash_s_to_t, I_t, self.d_patch)

        #This is the total loss that needs to be minimized. The only GAN loss here is -log(D(G(z)) times two for the two reconstruction losses
        l_total_to_min = rec_loss_1 + rec_loss_2 + gan_loss_1_g + gan_loss_2_g #+ patch_loss

        named_losses = {
            'rl1': rec_loss_1,
            'rl2': rec_loss_2,
            'gl1d': gan_loss_1_d,  # huge numbers, grows very fast, needs scaling(?)
            'gl2d': gan_loss_2_d,  # ^
            'gl1g': gan_loss_1_g,  # either very large numbers or 0
            'gl2g': gan_loss_2_g,  # ^
            # 'pl': patch_loss
        }
        
        self.log_dict({'gen_loss': l_total_to_min, 'disc_loss': l_total_to_max, **named_losses}, prog_bar=True, on_epoch=True)
        return  {'gen_loss': l_total_to_min, 'disc_loss': l_total_to_max}

    def configure_optimizers(self):


        param_to_min = list(self.a_net.parameters()) + list(self.p_net.parameters()) + list(self.g_net.G.parameters())
        param_to_max = list(self.g_net.D.parameters()) #+ list(self.d_patch.parameters())
        min_opt = torch.optim.Adam(param_to_min, lr=self.g_lr, betas=(0.0, 0.99))
        max_opt = torch.optim.Adam(param_to_max, lr=self.d_lr, betas=(0.0, 0.99))

        #Can also do learning rate scheduling:
        #optimizers = [G_opt, D_opt]
        #lr_schedulers = {'scheduler': ReduceLROnPlateau(G_opt, ...), 'monitor': 'metric_to_track'}
        #return optimizers, lr_schedulers

        return min_opt, max_opt
