import torch
import torchvision
from stylegan2 import GeneratorBlock
from models import ANet, PNet
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import DeepFashionDataset

def genblock1_forward(x, prev_rgb, istyle, inoise):
  genblock = GeneratorBlock(latent_dim = 2048, input_channels = 512, filters = 512, upsample = False)
  output = genblock(x, prev_rgb, istyle, inoise)
  return output

def genblock2_forward(x, prev_rgb, istyle, inoise):
  genblock = GeneratorBlock(latent_dim = 2048, input_channels = 512, filters = 256)
  output = genblock(x, prev_rgb, istyle, inoise)
  return output

def genblock3_forward(x, prev_rgb, istyle, inoise):
  genblock = GeneratorBlock(latent_dim = 2048, input_channels = 256, filters = 128) 
  output = genblock(x, prev_rgb, istyle, inoise)
  return output

def genblock4_forward(x, prev_rgb, istyle, inoise):
  genblock = GeneratorBlock(latent_dim = 2048, input_channels = 128, filters = 64) 
  output = genblock(x, prev_rgb, istyle, inoise)
  return output

def genblock5_forward(x, prev_rgb, istyle, inoise):
  genblock = GeneratorBlock(latent_dim = 2048, input_channels = 64, filters = 32, upsample_rgb = False)
  output = genblock(x, prev_rgb, istyle, inoise)
  return output  

def test_gnet_genblock1_forward():
    batch_size = 2
    input_x_channels = 512
    input_x_imsize = 32
    expected_x_channels = 512
    expected_x_imsize = 32
    expected_rgb_channels = 3
    expected_rgb_imsize = 64 
    x = torch.rand(2, input_x_channels, input_x_imsize, input_x_imsize)
    istyle = torch.rand(batch_size, 2048)
    inoise = torch.rand(batch_size, 512, 512, 1)
    assert((genblock1_forward(x, None, istyle, inoise,))[0].size() == (batch_size, expected_x_channels, expected_x_imsize, expected_x_imsize)
     and (genblock1_forward(x, None, istyle, inoise,))[1].size() == (batch_size, expected_rgb_channels, expected_rgb_imsize, expected_rgb_imsize))

def test_gnet_genblock2_forward():
    batch_size = 2
    input_x_channels = 512
    input_x_imsize = 32 
    input_prev_rgb_channels = 3
    input_prev_rgb_imsize = 64
    expected_x_channels = 256
    expected_x_imsize = 64
    expected_rgb_channels = 3
    expected_rgb_imsize = 128 
    x = torch.rand(batch_size, input_x_channels, input_x_imsize, input_x_imsize)
    prev_rgb = torch.rand(batch_size, input_prev_rgb_channels, input_prev_rgb_imsize, input_prev_rgb_imsize)
    istyle = torch.rand(batch_size, 2048)
    inoise = torch.rand(batch_size, 512, 512, 1)
    assert((genblock2_forward(x, prev_rgb, istyle, inoise,))[0].size() == (batch_size, expected_x_channels, expected_x_imsize, expected_x_imsize)
     and (genblock2_forward(x, prev_rgb, istyle, inoise,))[1].size() == (batch_size, expected_rgb_channels, expected_rgb_imsize, expected_rgb_imsize))

def test_gnet_genblock3_forward():
    batch_size = 2
    input_x_channels = 256
    input_x_imsize = 64 
    input_prev_rgb_channels = 3
    input_prev_rgb_imsize = 128
    expected_x_channels = 128
    expected_x_imsize = 128
    expected_rgb_channels = 3
    expected_rgb_imsize = 256 
    x = torch.rand(batch_size, input_x_channels, input_x_imsize, input_x_imsize)
    prev_rgb = torch.rand(batch_size, input_prev_rgb_channels, input_prev_rgb_imsize, input_prev_rgb_imsize)
    istyle = torch.rand(batch_size, 2048)
    inoise = torch.rand(batch_size, 512, 512, 1)
    assert((genblock3_forward(x, prev_rgb, istyle, inoise,))[0].size() == (batch_size, expected_x_channels, expected_x_imsize, expected_x_imsize)
     and (genblock3_forward(x, prev_rgb, istyle, inoise,))[1].size() == (batch_size, expected_rgb_channels, expected_rgb_imsize, expected_rgb_imsize))

def test_gnet_genblock4_forward():
    batch_size = 2
    input_x_channels = 128
    input_x_imsize = 128 
    input_prev_rgb_channels = 3
    input_prev_rgb_imsize = 256
    expected_x_channels = 64
    expected_x_imsize = 256
    expected_rgb_channels = 3
    expected_rgb_imsize = 512 
    x = torch.rand(batch_size, input_x_channels, input_x_imsize, input_x_imsize)
    prev_rgb = torch.rand(batch_size, input_prev_rgb_channels, input_prev_rgb_imsize, input_prev_rgb_imsize)
    istyle = torch.rand(batch_size, 2048)
    inoise = torch.rand(batch_size, 512, 512, 1)
    assert((genblock4_forward(x, prev_rgb, istyle, inoise,))[0].size() == (batch_size, expected_x_channels, expected_x_imsize, expected_x_imsize)
     and (genblock4_forward(x, prev_rgb, istyle, inoise,))[1].size() == (batch_size, expected_rgb_channels, expected_rgb_imsize, expected_rgb_imsize))

def test_gnet_genblock5_forward():
    batch_size = 2
    input_x_channels = 64
    input_x_imsize = 256
    input_prev_rgb_channels = 3
    input_prev_rgb_imsize = 512
    expected_x_channels = 32
    expected_x_imsize = 512
    expected_rgb_channels = 3
    expected_rgb_imsize = 512 
    x = torch.rand(batch_size, input_x_channels, input_x_imsize, input_x_imsize)
    prev_rgb = torch.rand(batch_size, input_prev_rgb_channels, input_prev_rgb_imsize, input_prev_rgb_imsize)
    istyle = torch.rand(batch_size, 2048)
    inoise = torch.rand(batch_size, 512, 512, 1)
    assert((genblock5_forward(x, prev_rgb, istyle, inoise,))[0].size() == (batch_size, expected_x_channels, expected_x_imsize, expected_x_imsize)
     and (genblock5_forward(x, prev_rgb, istyle, inoise,))[1].size() == (batch_size, expected_rgb_channels, expected_rgb_imsize, expected_rgb_imsize))