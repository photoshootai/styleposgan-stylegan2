from losses.VGGPerceptual import VGG16
import torch.nn as nn
from torchvision import models

def get_total_loss(I_s, I_dash_s, I_t, I_gen):
    #get_reconstruction_loss()
    pass

#Helper function to get reconstruction_loss multiple times as used in total_loss
def get_reconstruction_loss(I_a, I_b):
    #
    pass
    
def get_l1_loss(I_gen, I_gt):
    l1_loss = nn.L1Loss()
    return l1_loss(I_gen, I_gt)

def get_perceptual_vgg_loss(I_gen, I_gt):
    vgg = VGG16()
    gen_tups = vgg.forward(I_gen)
    gt_tups  = vgg.forward(I_gt)
    vgg_loss = calcaluate_l_vgg(gen_tups, gt_tups)
    return vgg_loss

def calcaluate_l_vgg(gen_tuples, gt_tuples):
    #TODO: Vectorization
    total = 0
    for i in range(len(gen_tuples)):
        total += (1/len(gen_tuples[i])) * get_l1_loss(gen_tuples[i], gt_tuples[i])
    return total

def face_id_loss():
    pass

def style_gan_loss():
    pass

def patch_loss():
    pass