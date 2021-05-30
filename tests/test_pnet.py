import torch
#import torchvision
#import os
import sys
#sys.path.insert(0, './models')
from models import PNet

#batch_size=               1
#gpus=                     1
#image_size=               512
#tpu_cores=                8
#precision=                16 
#accumulate_grad_batches=  32
#top_k_training=           False
#deterministic=            True
testtensor = torch.rand(1, 3, 512, 512)

#from torch.utils.data import Dataset, DataLoader, random_split
#from datasets import DeepFashionDataset

#source_image_path="/content/drive/MyDrive/TrainingData/SourceImages"
#pose_map_path="/content/drive/MyDrive/TrainingData/PoseMaps"
#texture_map_path="/content/drive/MyDrive/TrainingData/TextureMaps"

#dm = DeepFashionDataset(source_image_path, pose_map_path, texture_map_path, True, batch_size)
#dataloader = DataLoader(dm, batch_size, num_workers=1)
#batch = next(iter(dataloader))

#print("batch", batch)

def pnet_forward(input):
  pnet = PNet(im_chan=3)
  output = pnet(input)
  return output.size()


def test_pnet_forward():
    testtensor= torch.rand(1, 3, 512, 512)
    assert(pnet_forward(testtensor) == (1, 512, 32, 32))