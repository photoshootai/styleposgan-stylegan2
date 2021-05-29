import torch
import os, random
import cv2

import numpy as np
import pytorch_lightning as pl
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader, random_split
from functools import partial

from PIL import Image

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return transforms.functional.resize(image, min_size)
    return image


class DeepFashionDataset(Dataset):
  def __init__(self, source_image_path, pose_map_path, texture_map_path, image_size=(512, 512), train=False, batch_size=32):
    

    self.img_transform = transforms.Compose([
            # transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize((1024, 512)),
            #RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            #transforms.Lambda(expand_greyscale(transparent))
        ])
    
    self.texture_transform = transforms.Compose([
      transforms.ToTensor()
    ])
    
    
    self.batch_size = batch_size
    self.img_path = source_image_path
    self.pose_path = pose_map_path
    self.texture_path = texture_map_path
    self.train = train

    #begin a list where we will keep the id for given photos. 
    #the same ID should be found in all 3 folders
    self.data_id = []

    #os.walk will return three values: the location it was given, root, the dirs 
    #inside that location and the files in the location
    for root, dirs, files in os.walk(self.img_path):

        #this will take the last part of the file's location (which is just its name)
        self.data_id = [file_path.split("/")[-1] for file_path in files]

    #define how the data should be categorized
    self.class_map = {"source_img" : 0, "pose_map": 1, "texture_map": 2} 

    #for our purposes, a datapoint is actually a pair of tuples containing (image, pose, texture)
    # self.data = list(filter(lambda x: x[0] != x[1] and random.uniform(0, 1) < 0.003, product(self.data_id, self.data_id)))
    random.shuffle(self.data_id)
    mid = len(self.data_id) // 2
    self.data = list(zip(self.data_id[:mid], self.data_id[mid:]))
    self.data += [(y, x) for x, y in self.data]
    
    print('No dup pairs:', all(x[0] != x[1] for x in self.data))
    print('no dup tups:', (sum(1 for x in self.data if (self.data.count(x) > 1)) == 0))

    train_prop = int(len(self.data) * 0.99)
    if self.train:
        self.data = self.data[:train_prop]
    else:
        self.data = self.data[train_prop:]

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
    full_texture_path2 = os.path.join(self.texture_path, id2)

    #read in the source images (including pose and texture), convert to torch 
    source_img = Image.open(full_image_path1) 
    source_pose = Image.open(full_pose_path1)
    source_texture = Image.open(full_texture_path1)
    
    #read in the target images (including pose and texture), convert to torch 
    target_img = Image.open(full_image_path2)
    target_pose = Image.open(full_pose_path2)
    target_texture = Image.open(full_texture_path2)

    #put them together
    print('source_pose size pre-transform', source_pose.size)
    source_datapoint = (self.img_transform(source_img), self.img_transform(source_pose), self.texture_transform(source_texture))
    target_datapoint = (self.img_transform(target_img), self.img_transform(target_pose), self.texture_transform(target_texture))
    print('source_pose size post-transform', source_datapoint[1].shape)
    return source_datapoint, target_datapoint

class DeepFashionDataModule(pl.LightningDataModule):
  def __init__(self, source_image_path, pose_map_path, texture_map_path, batch_size=32):
    super().__init__()
    self.img_path = source_image_path
    self.pose_path = pose_map_path
    self.texture_path = texture_map_path
    self.batch_size = batch_size

  def setup(self, stage="fit"):
    self.train_data = DeepFashionDataset(self.img_path, self.pose_path, self.texture_path, True, self.batch_size)
    self.test_data = DeepFashionDataset(self.img_path, self.pose_path, self.texture_path, False, self.batch_size)
    
    training_proportion = int(len(self.train_data) * 0.95)
    self.train_data, self.val_data = random_split(self.train_data, [training_proportion, len(self.train_data)-training_proportion])
    
    print(len(self.train_data), len(self.val_data), len(self.test_data))

  def train_dataloader(self):
    return DataLoader(self.train_data, self.batch_size, num_workers=4)  # TODO: Add workers
    
  def val_dataloader(self):
    return DataLoader(self.val_data, self.batch_size, num_workers=4)

  def test_dataloader(self):
    return DataLoader(self.test_data, self.batch_size, num_workers=4)