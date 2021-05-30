import torch
import torchvision
from models import PNet
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import DeepFashionDataset
batch_size=               1
gpus=                     1
image_size=               512
tpu_cores=                8
precision=                16 
accumulate_grad_batches=  32
top_k_training=           False
deterministic=            True
testtensor = torch.rand(1, 3, 512, 512)

source_image_path="/content/drive/MyDrive/TrainingData/SourceImages"
pose_map_path="/content/drive/MyDrive/TrainingData/PoseMaps"
texture_map_path="/content/drive/MyDrive/TrainingData/TextureMaps"

def pnet_forward(input):
  pnet = PNet(im_chan=3)
  output = pnet(input)
  return output.size()


def test_pnet_forward():
    dm = DeepFashionDataset(source_image_path, pose_map_path, texture_map_path, True, batch_size)
    dataloader = DataLoader(dm, batch_size, num_workers=1)
    batch = next(iter(dataloader))
    

    assert(pnet_forward(testtensor) == (1, 512, 32, 32))