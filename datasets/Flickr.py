import cv2
import torch
import os, random

import numpy as np
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, random_split

class FlickrDataset(Dataset):
  def __init__(self, source_image_path, pose_map_path, texture_map_path, train=False, batch_size=32):
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

    #read in the source images (including pose and texture), convert to torch, move to GPU
    source_img = torch.from_numpy(cv2.imread(full_image_path1)).cuda() 
    source_pose = torch.from_numpy(cv2.imread(full_pose_path1)).cuda()
    source_texture = torch.from_numpy(cv2.imread(full_texture_path1)).cuda()
    
    #read in the target images (including pose and texture), convert to torch, move to GPU
    target_img = torch.from_numpy(cv2.imread(full_image_path2)).cuda() 
    target_pose = torch.from_numpy(cv2.imread(full_pose_path2)).cuda()
    target_texture = torch.from_numpy(cv2.imread(full_texture_path2)).cuda()

    #put them together
    source_datapoint = (source_img, source_pose, source_texture)
    target_datapoint = (target_img, target_pose, target_texture)

    return source_datapoint, target_datapoint

