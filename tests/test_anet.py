import torch
import torchvision
from models import ANet
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import DeepFashionDataset


def anet_forward(input):
  anet = ANet(im_chan=3)
  output = anet(input)
  return output


def test_anet_forward_rand():
    testtensor = torch.rand(4, 3, 512, 512)
    assert(anet_forward(testtensor).size() == (4, 1, 2048))

def test_anet_forward_training_sourcetexturebatch():
    source_image_path="./data_down/SourceImages"
    pose_map_path="./data_down/PoseMaps"
    texture_map_path="./data_down/TextureMaps"
    input_image_size = 512
    expected_vector_length = 2048
    input_batch_size = 1
    expected_batch_size = 1
    #Initialization
    ds = DeepFashionDataset(source_image_path, pose_map_path, texture_map_path, image_size=(input_image_size, input_image_size), train=True, batch_size=input_batch_size)
    dataloader = DataLoader(ds, input_batch_size, num_workers=1)
    (sourceimbatch, sourceposebatch, sourcetexturebatch),(targetimbach,targetposebatch,targettexturebatch) = next(iter(dataloader))
    assert(anet_forward(sourcetexturebatch).size() == (expected_batch_size, 1, expected_vector_length))

def test_anet_forward_training_targettexturebatch():
    source_image_path="./data_down/SourceImages"
    pose_map_path="./data_down/PoseMaps"
    texture_map_path="./data_down/TextureMaps"
    input_image_size = 512
    expected_vector_length = 2048
    input_batch_size = 4
    expected_batch_size = 4
    #Initialization
    ds = DeepFashionDataset(source_image_path, pose_map_path, texture_map_path, image_size=(input_image_size, input_image_size), train=True, batch_size=input_batch_size)
    dataloader = DataLoader(ds, input_batch_size, num_workers=1)
    (sourceimbatch, sourceposebatch, sourcetexturebatch),(targetimbach,targetposebatch,targettexturebatch) = next(iter(dataloader))
    assert(anet_forward(targettexturebatch).size() == (expected_batch_size, 1, expected_vector_length))