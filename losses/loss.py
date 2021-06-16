import torch
import torch.nn as nn

    

#Eq 4 in paper
def get_l1_loss(I_gen, I_gt, reduction='mean'):
    l1_loss = nn.L1Loss(reduction=reduction) # reconsider init module every call
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



def gan_d_loss(generated, real, G, D, D_aug, device, detach=True):

    criterion = nn.BCEWithLogitsLoss()

    fake_output, fake_q_loss = D_aug(generated.detach(), detach = detach)
    real_output, real_q_loss = D_aug(real) #D(X)

    batch_size = generated.shape[0]

    real_label = torch.ones((batch_size, 1), device=device)
    fake_label = torch.zeros((batch_size, 1), device=device)

    disc_loss_real = criterion(real_output, real_label)
    disc_loss_fake = criterion(fake_output, fake_label)

    total_disc_loss = disc_loss_real + disc_loss_fake 

    return total_disc_loss


def gan_g_loss(generated, real, G, D, D_aug, device):
    criterion = nn.BCEWithLogitsLoss()
    fake_output, _ = D_aug(generated)

    batch_size = generated.shape[0]
    real_label = torch.ones((batch_size, 1), device=device)

    g_loss = criterion(fake_output, real_label) #-1 * log(D(G(z))

    return g_loss


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
    # print('Shape of DPatch', d_x.shape)
    loss = get_l1_loss(d_g_z, d_x, reduction='mean')
    return loss


