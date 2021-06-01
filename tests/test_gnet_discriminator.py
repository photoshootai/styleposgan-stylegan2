import torch
import torchvision
from stylegan2 import Discriminator
from models import ANet, PNet
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import DeepFashionDataset

def gnet_discriminator_forward(x):
  gnet_disc = Discriminator(image_size = 512, latent_dim = 2048)
  output =  gnet_disc(x)
  return output

def test_gnet_discriminator_basecase():
    batch_size = 2
    x_channels = 3
    x_imsize = 512
    expected_channels = 3
    expected_image_size = 512
    x = torch.rand(batch_size, x_channels, x_imsize, x_imsize)
    #assert(gnet_discriminator_forward(x)[0].size() ==  (batch_size, expected_channels, x_imsize, x_imsize)
