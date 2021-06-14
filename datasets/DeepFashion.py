import torch
import torch.nn as nn
import os, random

from torchvision import transforms

from torch.utils.data import Dataset
from functools import partial

from PIL import Image

def exists(val):
    return val is not None

def convert_rgb_to_transparent(image):
    if image.mode != 'RGBA':
        return image.convert('RGBA')
    return image

def convert_transparent_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return transforms.functional.resize(image, min_size)
    return image

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

class DeepFashionDataset(Dataset):

  def __init__(self, main_folder, image_size, transparent = False, aug_prob = 0.):
    super().__init__()

    self.image_size_dims = (image_size, image_size)
    print("Image Size:  ", image_size)

    self.img_path = main_folder + "/SourceImages"
    self.pose_path = main_folder + "/PoseMaps"
    self.texture_path = main_folder + "/TextureMaps"

    #begin a list where we will keep the id for given photos. 
    #the same ID should be found in all 3 folders
    self.data_id = []

    #os.walk will return three values: the location it was given, root, the dirs 
    #inside that location and the files in the location
    self.data_id = [d.name for d in os.scandir(self.img_path)]

    #define how the data should be categorized
    self.class_map = {"source_img" : 0, "pose_map": 1, "texture_map": 2} 

    #for our purposes, a datapoint is actually a pair of tuples containing (image, pose, texture)
    # self.data = list(filter(lambda x: x[0] != x[1] and random.uniform(0, 1) < 0.003, product(self.data_id, self.data_id)))
    random.shuffle(self.data_id)
    mid = len(self.data_id) // 2
    self.data = list(zip(self.data_id[:mid], self.data_id[mid:]))
    self.data += [(y, x) for x, y in self.data]


    assert len(self.data) > 0, f'No training data could be obtained'

    convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
    num_channels = 3 if not transparent else 4

    self.img_transform = transforms.Compose([
        #transforms.Lambda(convert_image_fn),
        # transforms.Lambda(partial(resize_to_minimum_size, image_size)),
        transforms.Resize(self.image_size_dims),
        #RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
        transforms.ToTensor(),
        # transforms.Lambda(expand_greyscale(transparent))
    ])

    self.texture_transform = transforms.Compose([
      transforms.ToTensor()
    ])

  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
      id1, id2 = self.data[idx] #get a pair of datapoints from the list of pairings
      #the image, pose and path locations equal the concat(corresponding root, id)
      full_image_path1 = os.path.join(self.img_path, id1)
      full_image_path2 = os.path.join(self.img_path, id2)

      full_pose_path1 = os.path.join(self.pose_path, id1)
      full_pose_path2 = os.path.join(self.pose_path, id2)

      full_texture_path1 = os.path.join(self.texture_path, id1)
      # full_texture_path2 = os.path.join(self.texture_path, id2)

      #read in the source images (including pose and texture), convert to torch 
      source_img = Image.open(full_image_path1) 
      source_pose = Image.open(full_pose_path1)
      source_texture = Image.open(full_texture_path1)
      
      #read in the target images (including pose and texture), convert to torch 
      target_img = Image.open(full_image_path2)
      target_pose = Image.open(full_pose_path2)

      #put them together
      # print('source_pose size pre-transform', source_pose.size)
      source_datapoint = (self.img_transform(source_img), self.img_transform(source_pose), self.texture_transform(source_texture))
      target_datapoint = (self.img_transform(target_img), self.img_transform(target_pose))

      return source_datapoint, target_datapoint