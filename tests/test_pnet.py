import torch
import torchvision
from models import PNet
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import DeepFashionDataset


def pnet_forward(input):
  pnet = PNet(im_chan=3)
  output = pnet(input)
  return output


def test_pnet_forward_rand():
    testtensor = torch.rand(1, 3, 512, 512)
    assert(pnet_forward(testtensor).size() == (1, 512, 32, 32))

def test_pnet_forward_training_sourceposebatch():
    source_image_path="./data_down/SourceImages"
    pose_map_path="./data_down/PoseMaps"
    texture_map_path="./data_down/TextureMaps"
    input_image_size = 512
    expected_image_size = 32
    input_batch_size = 1
    expected_batch_size = 1
    expected_channels = 512
    #Initialization
    ds = DeepFashionDataset(source_image_path, pose_map_path, texture_map_path, image_size=(input_image_size, input_image_size), train=True, batch_size=input_batch_size)
    dataloader = DataLoader(ds, input_batch_size, num_workers=1)
    (sourceimbatch, sourceposebatch, sourcetexturebatch),(targetimbach,targetposebatch,targettexturebatch) = next(iter(dataloader))
    assert(pnet_forward(sourceposebatch).size() == (expected_batch_size, expected_channels, (expected_image_size), expected_image_size))

def test_pnet_forward_training_targetposebatch():
    source_image_path="./data_down/SourceImages"
    pose_map_path="./data_down/PoseMaps"
    texture_map_path="./data_down/TextureMaps"
    input_image_size = 512
    expected_image_size = 32
    input_batch_size = 4
    expected_batch_size = 4
    expected_channels = 512
    #Initialization
    ds = DeepFashionDataset(source_image_path, pose_map_path, texture_map_path, image_size=(input_image_size, input_image_size), train=True, batch_size=input_batch_size)
    dataloader = DataLoader(ds, input_batch_size, num_workers=1)
    (sourceimbatch, sourceposebatch, sourcetexturebatch),(targetimbach,targetposebatch,targettexturebatch) = next(iter(dataloader))
    assert(pnet_forward(targetposebatch).size() == (expected_batch_size, expected_channels, (expected_image_size), expected_image_size))