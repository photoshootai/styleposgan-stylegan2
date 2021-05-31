from losses.FaceIDLoss import FaceIDLoss
import torch
import torch.nn as nn
from torchvision import models
from stylegan2 import hinge_loss, gen_hinge_loss
import numpy as np

#Facenet
#TODO: evaluate other options along 2 dimensions of trade off, prediction quality and inference time
from facenet_pytorch import MTCNN, InceptionResnetV1


    
#Eq 4 in paper
def get_l1_loss(I_gen, I_gt, reduction='mean'):
    l1_loss = nn.L1Loss(reduction=reduction)
    return l1_loss(I_gen, I_gt)

def get_perceptual_vgg_loss(vgg_perceptual_model, I_gen, I_gt):
    
    gen_tups = vgg_perceptual_model(I_gen)
    gt_tups  = vgg_perceptual_model(I_gt)
    vgg_loss = calcaluate_l_vgg(gen_tups, gt_tups)
    return vgg_loss

#Eq 5
def calcaluate_l_vgg(gen_tuples, gt_tuples):
    #TODO: Vectorization?
    total = 0
    for i in range(len(gen_tuples)):
        total += (1/len(gen_tuples[i])) * get_l1_loss(gen_tuples[i], gt_tuples[i])
    return total

def get_face_id_loss(generated, gt, face_id_loss_model, crop_size):
    return face_id_loss_model(generated, gt, crop_size)


"""
Don't know if we need these two below, but this basically define G and D's losses using BCE seperately, following the usual Pytorch tutorial
"""
def gan_d_loss(generated, real, G, D, D_aug, device, detach=True):

    #Training Discriminator
    G_requires_reals = False
    criterion = nn.BCEWithLogitsLoss()

    fake_output, fake_q_loss = D_aug(generated, detach = detach)
    real_output, real_q_loss = D_aug(real)

    batch_size = generated.shape[0]

    real_label = torch.ones((batch_size, 1), device=device)
    fake_label = torch.zeros((batch_size, 1), device=device)

    disc_loss_real = criterion(real_output, real_label)
    disc_loss_fake = criterion(fake_output, fake_label)

    total_disc_loss = disc_loss_real + disc_loss_fake 
    #TODO: Check this
    # if args["epoch_id"] % 4== 0:
    #     gp = gradient_penalty(image_batch, real_output)
    #     last_gp_loss = gp.clone().detach().item()
    #     #track(last_gp_loss, 'GP')
    #     total_disc_loss = total_disc_loss + gd

    return total_disc_loss


def gan_g_loss(generated, real, G, D, D_aug, device):
    criterion = nn.BCEWithLogitsLoss()
    fake_output, _ = D_aug(generated)

    batch_size = generated.shape[0]
    real_label = torch.ones((batch_size, 1), device=device)
    # print("real_label", real_label)
    # print("fake_output", fake_output)
    g_loss = criterion(fake_output, real_label) #-1 * log(D(G(z))

    return g_loss
    
    #Experimental features for contrastive loss and top-k training

    # G_requires_reals = False
    # real_output = None
    # if G_requires_reals:
    #     image_batch = next(self.loader).cuda(self.rank)
    #     real_output, _ = D_aug(image_batch, detach = True, **aug_kwargs)
    #     real_output = real_output.detach()

    # if self.top_k_training:
    #     epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
    #     k_frac = max(self.generator_top_k_gamma ** epochs, self.generator_top_k_frac)
    #     k = math.ceil(batch_size * k_frac)

    #     if k != batch_size:
    #         fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)
    


    # Path penalty feature

    # if apply_path_penalty:
    # pl_lengths = calc_pl_lengths(w_styles, generated_images)
    # avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

    # if not is_empty(self.pl_mean):
    #     pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
    #     if not torch.isnan(pl_loss):
    #         gen_loss = gen_loss + pl_loss


def get_patch_loss(generated: torch.Tensor, real: torch.Tensor,
                   DPatch: torch.nn.Module) -> torch.Tensor:
    """
    Produce patch loss betweeen generated and real image using DPatch patch discriminator

    Arguments:
        generated [torch.Tensor (batch, C, H, W)]: generated image batch
        real [torch.Tensor (batch, C, H, W)]: real image batch
        DPatch [torch.nn.Module]: Patch discrimator with forward
    
    Returns:
        loss: [torch.Tensor (0)]: rank 0 tensor representing a scalar loss value

    Side Effects:
        None    
    """
    d_x = DPatch(real)
    d_g_z = DPatch(generated)
    loss = get_l1_loss(d_g_z, d_x, reduction='mean')
    return loss


def _disc_loss_function(real, fake):
    pass


def _gen_loss_function(real, fake):
    pass

