from torch.nn.functional import embedding
from losses.VGGPerceptual import VGG16
import torch.nn as nn
from torchvision import models
from stylegan2 import hinge_loss, gen_hinge_loss


#Facenet
#TODO: evaluate other options along 2 dimensions of trade off, prediction quality and inference time
from facenet_pytorch import MTCNN, InceptionResnetV1

#Helper function to get reconstruction_loss multiple times as used in total_loss
def get_reconstruction_loss(generated, real, G, D, D_aug, loss_weights, args): 
    #Eq 3 in paper
    rec_loss = loss_weights["l1"]*get_l1_loss(generated, real) + 0#gan_loss(generated, real, G, D)
    return rec_loss
    
#Eq 4 in paper
def get_l1_loss(I_gen, I_gt):
    l1_loss = nn.L1Loss(reduction="sum")
    return l1_loss(I_gen, I_gt)

def get_perceptual_vgg_loss(I_gen, I_gt):
    vgg = VGG16().cuda()
    gen_tups = vgg.forward(I_gen)
    gt_tups  = vgg.forward(I_gt)
    vgg_loss = calcaluate_l_vgg(gen_tups, gt_tups)
    return vgg_loss

#Eq 5
def calcaluate_l_vgg(gen_tuples, gt_tuples):
    #TODO: Vectorization
    total = 0
    for i in range(len(gen_tuples)):
        total += (1/len(gen_tuples[i])) * get_l1_loss(gen_tuples[i], gt_tuples[i])
    return total

#Eq 6
def face_id_loss():
    mtcnn = MTCNN()
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    img_cropped = mtcnn()
    embedding = resnet()
#
def gan_d_loss(generated, real, G, D, D_aug, args):

    #Training Discriminator
    G_requires_reals = False
    D_loss_fn = hinge_loss
    G_loss_fn = gen_hinge_loss

    fake_output, fake_q_loss = D_aug(generated.clone().detach(), detach = True)
    real_output, real_q_loss = D_aug(real)

    real_output_loss = real_output
    fake_output_loss = fake_output
    
    disc_loss = D_loss_fn(real_output_loss, fake_output_loss)

    #real = 1, fake =1 then 1
    #real = 1, fake =0 then 1.5 
    #real = 0, fake =1 then 0.5
    #real = 0, fake = 0 then 1

    if args["epoch_id"] % 4== 0:
        gp = gradient_penalty(image_batch, real_output)
        last_gp_loss = gp.clone().detach().item()
        #track(last_gp_loss, 'GP')
        disc_loss = disc_loss + gp

    disc_loss= disc_loss.item()

    return disc_loss


def gan_g_loss(generated, real, G, D, D_aug, args):
    fake_output, _ = D_aug(generated)
    fake_output_loss = fake_output
    
    G_loss_fn = gen_hinge_loss

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
    
    loss = G_loss_fn(fake_output_loss, real_output)
    gen_loss = loss

    # Path penalty feature

    # if apply_path_penalty:
    # pl_lengths = calc_pl_lengths(w_styles, generated_images)
    # avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

    # if not is_empty(self.pl_mean):
    #     pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
    #     if not torch.isnan(pl_loss):
    #         gen_loss = gen_loss + pl_loss
    g_loss = float(gen_loss.detach.item())
    return g_loss

def patch_loss():
    pass