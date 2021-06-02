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

from losses import get_face_id_loss, gan_g_loss, gan_d_loss, get_l1_loss, get_perceptual_vgg_loss, VGG16Perceptual, DPatch

class StylePoseGAN(pl.LightningModule):

    def __init__(self, image_size, batch_size, g_lr=2e-3, d_lr=2e-3, ttur_mult=2, latent_dim=2048,
                 network_capacity=16, attn_layers=(1, 2, 3, 4), mtcnn_crop_size=160, steps=0, pl_reg=True):  # changed attention to tuple since mutable default args cause problems
        super().__init__()
        self.a_net = ANet()
        self.p_net = PNet()
        self.g_net = GNet(image_size=image_size, latent_dim=latent_dim ) #Contains g_net.G, g_net.D, g_net.D_aug, g_net.S

        self.d_patch = DPatch() # Needs to be on same device as data!

        self.d_lr = d_lr
        self.g_lr = g_lr
        self.image_size = image_size
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.mtcnn_crop_size = mtcnn_crop_size

        self.steps = steps
        self.pl_reg = pl_reg
        self.pl_mean = None
        self.pl_length_ma = EMA(0.99)

        #TODO: This is wrong
        #Noise Tensor
        #self.register_buffer("input_noise", torch.randn(batch_size, self.image_size, self.image_size, 1))

        # self.input_noise = , device= self.device)
        # self.input_noise = torch.FloatTensor(batch_size, self.image_size, self.image_size, 1, device=self.device).uniform_(0., 1.) #TODO: Fix to generalized case

        #Disabling Pytorch lightning's default optimizer

        #Loss calculation models
        self.vgg16_perceptual_model = VGG16Perceptual(requires_grad=False)
        self.face_id_loss = FaceIDLoss(self.mtcnn_crop_size, requires_grad = False, device=self.device)
        self.automatic_optimization = False

        print('Device Rank', self.global_rank)  # should be 0 on main, > 0 on other gpus/tpus/cpus
        print("StylePoseGAN module initialized with: ", {"image_size": self.image_size, "batch_size": self.batch_size, "latent_dim": self.latent_dim} )


    def compute_loss_components(self, batch):
        """
        Separate repeated code from validation and training steps into different function for easier update/debugging
        """

        # Weights
        weight_l1 =1
        weight_vgg = 1
        weight_face = 1
        weight_gan = 1
        weight_patch = 1

        normalize = lambda t: (t - torch.min(t)) / (torch.max(t) - torch.min(t))

        (I_s, S_pose_map, S_texture_map), (I_t, T_pose_map, _) = batch #x, y = batch, so x is  the tuple, and y is the triplet

        # PNet
        E_s = self.p_net(S_pose_map)
        E_t = self.p_net(T_pose_map)


        # ANet
        z_s = self.a_net(S_texture_map)  # needs norm
        # z_t = self.a_net(T_texture_map)

        # # Create model

        # """
        # Generator Forward Pass
        # """
        
        # input_noise = torch.randn(I_s.shape[0], self.image_size, self.image_size, 1).type_as(z_s)
        # print("Forward Pass details: ", {"z_s_repeated": z_s.repeat(1, 5, 1).size(), "E_s": E_s.size(), "E_t": E_t.size(), "input_noise": input_noise.size()})
        # Repeat z num_layer times
        # I_dash_s = self.g_net.G(z_s.repeat(1, 5, 1), input_noise, E_s) #G(E_s, z_s)
        # I_dash_s_to_t = self.g_net.G(z_s.repeat(1, 5, 1), input_noise, E_t)
        
        # if not (self.steps % 750):
        #     if not (os.path.isdir('./test_ims')):
        #         os.mkdir('./test_ims')
        #     save_image(normalize(I_dash_s_to_t), f'./test_ims/step_{self.steps}.jpg')

        # # Consider normalizing:
        # # I_s = normalize(I_s)
        # # I_t = normalize(I_t)
        # # E_s = normalize(E_s)
        # # E_t = normalize(E_t)
        # # z_s = normalize(z_s)
        # # I_dash_s = normalize(I_dash_s)
        # # I_dash_s_to_t = normalize(I_dash_s_to_t)

        # #Need to detach at the top level
        # rec_loss_1 =  weight_l1 * get_l1_loss(I_dash_s, I_s) + \
        #               weight_vgg * get_perceptual_vgg_loss(self.vgg16_perceptual_model, I_dash_s, I_s) + \
        #               weight_face * get_face_id_loss(I_dash_s, I_s, self.face_id_loss, crop_size=self.mtcnn_crop_size)

        # rec_loss_2 =  weight_l1 * get_l1_loss(I_dash_s_to_t ,I_t) + \
        #               weight_vgg * get_perceptual_vgg_loss(self.vgg16_perceptual_model,I_dash_s_to_t, I_t) + \
        #               weight_face * get_face_id_loss(I_dash_s_to_t, I_t, self.face_id_loss, crop_size=self.mtcnn_crop_size)


        # gan_loss_1_d = weight_gan * gan_d_loss(I_dash_s.detach(), I_s, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device)
        # gan_loss_2_d = weight_gan * gan_d_loss(I_dash_s_to_t.detach(), I_t, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device)

        # gan_loss_1_g = weight_gan * gan_g_loss(I_dash_s, I_s, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device )
        # gan_loss_2_g = weight_gan * gan_g_loss(I_dash_s_to_t, I_t, self.g_net.G, self.g_net.D, self.g_net.D_aug, self.device)

        # patch_loss = weight_patch * get_patch_loss(I_dash_s_to_t, I_t, self.d_patch)

        # # return (rec_loss_1, rec_loss_2,
        # #         gan_loss_1_d, gan_loss_2_d,  # skyrockets, maybe we need to regulate?
        # #         gan_loss_1_g, gan_loss_2_g,  # flips between 0, and very large numbers
        # #         patch_loss,
        # #         I_dash_s, I_dash_s_to_t, z_s)

        return (1,1, 1,1, 1, 1, 1, I_s, I_t, z_s)
    # training_step defined the train loop. # It is independent of forward
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
        #Get optimizers
        # min_opt, max_opt = self.optimizers()

        # min_opt.zero_grad()
        # max_opt.zero_grad()

        (rec_loss_1, rec_loss_2, gan_loss_1_d, gan_loss_2_d, gan_loss_1_g,
         gan_loss_2_g, patch_loss, _ , I_dash_s_to_t, z_s) = self.compute_loss_components(batch)

        #Total Loss that needs to be maximized. The only GAN loss here is -[log(D(x)) + log(1-D(G(z)))] for the respective args
        l_total_to_max = (-1)*rec_loss_1 + (-1)*rec_loss_2 + gan_loss_1_d + gan_loss_2_d + patch_loss

        # min_opt.zero_grad()
        # l_total_to_min.backward(retain_graph=True)
        # min_opt.step()

        # max_opt.zero_grad()
        # self.manual_backward(l_total_to_max, retain_graph=True)
        # max_opt.step()
        # gan_loss_1_g = gan_g_loss(I_dash_s, I_s, self.g_net.G, self.g_net.D, self.g_net.D_aug)
        # gan_loss_2_g = gan_g_loss(I_dash_s_to_t, I_t, self.g_net.G, self.g_net.D, self.g_net.D_aug)


        #This is the total loss that needs to be minimized. The only GAN loss here is -log(D(G(z)) times two for the two reconstruction losses
        # need to merge losses?
        l_total_to_min = rec_loss_1 + rec_loss_2 + gan_loss_1_g + gan_loss_2_g


        named_losses = {
            'rl1': rec_loss_1,
            'rl2': rec_loss_2,
            'gl1d': gan_loss_1_d,  # huge numbers, grows very fast, needs scaling(?)
            'gl2d': gan_loss_2_d,  # ^
            'gl1g': gan_loss_1_g,  # either very large numbers or 0
            'gl2g': gan_loss_2_g,  # ^
            'pl': patch_loss
        }
        # print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in named_losses.items()) + "}")
        
        # min_opt.zero_grad()
        # self.manual_backward(l_total_to_min)
        # min_opt.step()

        # Calculate Moving Averages
        # avg_pl_length = self.pl_mean
        # is_main = self.global_rank == 0
        # apply_path_penalty = self.pl_reg and self.steps > 5000 and self.steps % 32 == 0
        # gen_loss = l_total_to_min # assuming this is the generator loss (?)

        # # assuming the loss is merged here atuomatically?
        # if apply_path_penalty:
        #     # pl_lengths = calc_pl_lengths(w_styles, generated_images)
        #     pl_lengths = calc_pl_lengths(w_styles, I_dash_s)
        #     avg_pl_length = torch.mean(pl_lengths.detach())  # scalar (rank-0 tensor)

        #     if not is_empty(self.pl_mean):  # checks if tensor is 0
        #         pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
        #         if not torch.isnan(pl_loss):
        #             gen_loss = gen_loss + pl_loss
        # if apply_path_penalty and not np.isnan(avg_pl_length):
        #     self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
        #     self.track(self.pl_mean, 'PL')

        # #EMA if on main thread
        # if is_main:
        #     if self.steps % 10 == 0 and self.steps > 20000:
        #         self.g_net.EMA()

        #     #Parameter Averaging
        #     if self.steps <= 25000 and self.steps % 1000 == 2:
        #         self.g_net.reset_parameter_averaging()

        # # save from NaN errors
        # if any(torch.isnan(l) for l in (l_total_to_min, l_total_to_max)):
        #     print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
        #     self.load(self.checkpoint_num)
        #     raise NanException

        # self.steps += 1

        self.log_dict({'generation_loss': l_total_to_min, 'disc_loss': l_total_to_max}, prog_bar=True)
        return  {'l_total_to_min': l_total_to_min, 'l_total_to_max': l_total_to_max, 'z_s': z_s, 'I_dash_s_to_t': I_dash_s_to_t}

    def training_step_end(self, *losses, **kwargs):
        # args = ({l_min, l_max, z_s, I_dash_s_to_t}, ...)
        # do we need to combine and backward in here?
        N = len(losses)
        min_opt, max_opt = self.optimizers()

        min_opt.zero_grad()
        max_opt.zero_grad()

        # Moving Averages
        avg_pl_length = self.pl_mean
        is_main = self.global_rank == 0
        apply_path_penalty = self.pl_reg and self.steps > 5000 and self.steps % 32 == 0  # TODO: change condition to > 5000

        total_gen_loss = torch.zeros((1), device=self.device) 
        total_disc_loss = torch.zeros((1), device=self.device)

        for loss_dict in losses:
            gen_loss = loss_dict['l_total_to_min'] # assuming this is the generator loss (?)
            disc_loss = loss_dict['l_total_to_max']
            z_s = loss_dict['z_s']
            I_dash_s_to_t = loss_dict['I_dash_s_to_t']

            if apply_path_penalty:
                # pl_lengths = calc_pl_lengths(w_styles, generated_images)
                pl_lengths = calc_pl_lengths(z_s, I_dash_s_to_t)
                avg_pl_length = torch.mean(pl_lengths.detach())  # scalar (rank-0 tensor)

                if not is_empty(self.pl_mean):  # checks if tensor is 0
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            total_gen_loss += gen_loss
            total_disc_loss += disc_loss
            self.manual_backward(disc_loss, retain_graph=True)
            self.manual_backward(gen_loss)

        total_gen_loss = total_gen_loss / N
        total_disc_loss = total_disc_loss / N

        #TODO: I think this is incorrect: -Madhav.
        max_opt.step()
        min_opt.step()  # in lucidrains, optstep is after ema

        if apply_path_penalty and not torch.isnan(avg_pl_length):
            # print('Applying path penalty')
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
            # self.track(self.pl_mean, 'PL')

        #EMA if on main thread
        if is_main:
            if self.steps % 10 == 0 and self.steps > 20000:
                self.g_net.EMA()

            #Parameter Averaging
            if self.steps <= 25000 and self.steps % 1000 == 2:
                self.g_net.reset_parameter_averaging()

        # save from NaN errors
        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        default_out = super().validation_step_end(*losses, **kwargs)
        # print(default_out)
        self.steps += 1
        return {'total_gen_loss': total_gen_loss, 'total_disc_loss': total_disc_loss}

    #TODO check this
    def validation_step(self, batch, batch_idx):
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

        # print("DEVICE IN VAL STEP IS: ", self.device) #Assuming that by the time training_step is called, self.device points to the correct device
        (rec_loss_1, rec_loss_2, gan_loss_1_d, gan_loss_2_d, gan_loss_1_g,
         gan_loss_2_g, patch_loss, I_dash_s, I_dash_s_to_t, z_s) = self.compute_loss_components(batch)


        #This is the total loss that needs to be minimized. The only GAN loss here is -log(D(G(z)) times two for the two reconstruction losses
        l_total_to_min = rec_loss_1 + rec_loss_2 + gan_loss_1_g + gan_loss_2_g

       #Total Loss that needs to be maximized. The only GAN loss here is -[log(D(x)) + log(1-D(G(z)))] for the respective args
        l_total_to_max = (-1)*rec_loss_1 + (-1)*rec_loss_2 + gan_loss_1_d + gan_loss_2_d + patch_loss

        return  {'l_total_to_min': l_total_to_min, 'l_total_to_max': l_total_to_max}

    def configure_optimizers(self):

        # init optimizers
        # G_opt = Adam(self.g_net.G.parameters(), lr=self.g_lr, betas=(0.5, 0.9))
        # D_opt = Adam(self.g_net.D.parameters(), lr=self.d_lr * self.ttur_mult, betas=(0.5, 0.9))

        param_to_min = list(self.a_net.parameters()) + list(self.p_net.parameters()) + list(self.g_net.G.parameters())
        param_to_max = list(self.g_net.D.parameters()) #+ list(self.d_patch.parameters())
        min_opt = torch.optim.Adam(param_to_min, lr=self.g_lr, betas=(0.5, 0.9))
        max_opt = torch.optim.Adam(param_to_max, lr=self.d_lr, betas=(0.5, 0.9))

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