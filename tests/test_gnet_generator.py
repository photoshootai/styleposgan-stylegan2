import torch
import torchvision
from stylegan2 import Generator
from models import ANet, PNet
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import DeepFashionDataset


def gnet_generator_forward(z, noise, E):
  gnet_gen = Generator(image_size = 512, latent_dim = 2048)
  output =  gnet_gen(z, noise, E)
  return output
def anet_forward(input):
  anet = ANet(im_chan=3)
  output = anet(input)
  return output
def pnet_forward(input):
  pnet = PNet(im_chan=3)
  output = pnet(input)
  return output

def test_gnet_generator_forward_basecase():
    input_batch_size = 2
    input_channels = 512
    input_E_imsize = 32
    input_imsize = 512
    input_vector_width = 1
    input_vector_length = 2048
    expected_image_size = 512
    input_z = torch.zeros(input_batch_size, input_vector_width, input_vector_length)
    input_z = input_z.expand(-1,5,-1)  
    input_noise = torch.rand(input_batch_size, input_imsize, input_imsize, 1)
    input_E = torch.zeros(input_batch_size, input_channels, input_E_imsize, input_E_imsize)
    assert(gnet_generator_forward(input_z, input_noise, input_E).size() == (2, 3, expected_image_size, expected_image_size))



def test_gnet_generator_forward_realcase1():
    source_image_path="./data_down/SourceImages"
    pose_map_path="./data_down/PoseMaps"
    texture_map_path="./data_down/TextureMaps"
    input_imsize = 512
    input_batch_size = 2
    inoise_imsize = 512
    expected_image_size = 512
    #Initialization
    ds = DeepFashionDataset(source_image_path, pose_map_path, texture_map_path, image_size=(input_imsize, input_imsize), train=True, batch_size=input_batch_size)
    dataloader = DataLoader(ds, input_batch_size, num_workers=1)
    (sourceimbatch, sourceposebatch, sourcetexturebatch),(targetimbach,targetposebatch,targettexturebatch) = next(iter(dataloader))
    input_z = (anet_forward(sourcetexturebatch))
    input_z = input_z.expand(-1,5,-1)  
    input_noise = torch.rand(input_batch_size, inoise_imsize, inoise_imsize, 1)
    input_E = (pnet_forward(sourceposebatch))
    assert(gnet_generator_forward(input_z, input_noise, input_E).size() == (2, 3, expected_image_size, expected_image_size))
