import torch
import torch.nn as nn
import os, random

from torchvision import transforms

from torch.utils.data import Dataset
from functools import partial
from typing import Union, Tuple
from itertools import chain, starmap, repeat
from math import ceil
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

class scale_and_crop(object):
    def __init__(self, img_size: Tuple[int]) -> None:
        self.img_size = img_size
        self.crop = transforms.RandomCrop(img_size)
    
    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        _, h, w = image_tensor.shape # should be applied after transforms.ToTensor()
        new_h, new_w = self.img_size
        
        min_img_dim = min(h, w)
        max_new_dim = max(new_h, new_w)

        ratio = max_new_dim / min_img_dim
        scaled_size = ceil(h * ratio), ceil(w * ratio)

        resize = transforms.Resize(scaled_size)        
        return self.crop(resize(image_tensor))

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
    def __init__(self, data_dir: str, image_size: Union[Tuple, int, float]=(512, 512), scale_crop: bool=True,
                 transparent: bool=False, seed: int=42, aug_prob: float=0.0) -> None:
        """
        Arguments:
            data_dir [str]: Path to data directory, should have subfolders {SourceImages, PoseMaps, TextureMaps}
            image_size [opt int or Tuple=(512, 512)]: Size to transform PoseMap images and SourceImages images
            scale_crop [opt bool=True]: scale and crop instead of resize
            transparent [opt bool=False]: if image has transparency channel
            seed [opt int=42]: The seed to shuffle data with
            aug_prob [opt float=0.0]: probability of augmentation on images in range [0.0, 1.0]

        Returns:
            None
        
        Side Effects:
            None
        """
        self.n_chan = 4 if transparent else 3
        self.img_size = image_size if isinstance(image_size, Tuple) else (image_size, image_size)

        in_data_dir = partial(os.path.join, data_dir)
        self.sub_dirs = ('SourceImages', 'PoseMaps', 'TextureMaps')
        self.img_dirs = tuple(map(in_data_dir, self.sub_dirs))
        assert all(map(os.path.isdir, self.img_dirs)), 'Some requisite image directories not found'

        file_names = [d.name for d in os.scandir(self.img_dirs[0])]

        random.seed(seed)
        random.shuffle(file_names)

        mid = len(file_names) // 2
        self.data = list(chain(zip(file_names[mid:], file_names[mid:]), zip(file_names[:mid], file_names[:mid])))
        self.data_len = len(self.data)

        assert self.data_len > 0, 'Empty dataset'

        self.to_rgb = convert_transparent_to_rgb if transparent else lambda x: x
        # self.expand_greyscale = expand_greyscale(transparent)
        self.scale_and_crop = scale_and_crop(self.img_size) if scale_crop else transforms.Resize(self.img_size) 
        self.transforms = (
            transforms.Compose([ # For source images
                transforms.ToTensor(),  # [0, 255] gets mapped to [0.0, 1.0]
                transforms.Lambda(self.to_rgb), # convert to 3 channels (squash alpha)
                transforms.Lambda(self.scale_and_crop)
            ]),
            transforms.Compose([  # For Posemaps
                transforms.ToTensor(),  # this will screw up pose segmentation since {0,..., 24} gets mapped to [0.0, 1.0]
                transforms.Lambda(self.scale_and_crop) # guaranteed to be 3 channels
            ]),
            transforms.Compose([  # For texture Maps
                transforms.ToTensor() # converts [0, 255] to float[0.0, 1.0]
            ])
        )
    
    def __len__(self):
        """
        length of entire dataset 
        """
        return self.data_len
    
    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor]]:
        """
        Arguments:
            index [int]: The index in data from which to get data
        
        Returns:
            Source, Target [(tensor; (n_chan, img_size, img_size))]
        """
        src_file, targ_file = self.data[index]
        src_im_paths = starmap(os.path.join, zip(self.img_dirs, repeat(src_file))) # (src, pose, txt)
        targ_im_paths = starmap(os.path.join, zip(self.img_dirs, repeat(targ_file, 2))) # (src, pose)

        src_imgs = map(Image.open, src_im_paths) 
        targ_imgs = map(Image.open, targ_im_paths)

        source_set = (f(x) for f, x in zip(self.transforms, src_imgs))
        target_set = (f(x) for f, x in zip(self.transforms, targ_imgs))

        return tuple(source_set), tuple(target_set)


# class DeepFashionDataset(Dataset):

#   def __init__(self, main_folder, image_size, transparent = False, aug_prob = 0.):
#     super().__init__()

#     self.image_size_dims = (image_size, image_size)
#     print("Image Size:  ", image_size)

#     s

#     # self.img_path = main_folder + "/SourceImages"
#     # self.pose_path = main_folder + "/PoseMaps"
#     # self.texture_path = main_folder + "/TextureMaps"

#     #begin a list where we will keep the id for given photos. 
#     #the same ID should be found in all 3 folders
#     self.data_id = []

#     #os.walk will return three values: the location it was given, root, the dirs 
#     #inside that location and the files in the location
#     self.data_id = [d.name for d in os.scandir(self.img_path)]

#     #define how the data should be categorized
#     self.class_map = {"source_img" : 0, "pose_map": 1, "texture_map": 2} 

#     #for our purposes, a datapoint is actually a pair of tuples containing (image, pose, texture)
#     # self.data = list(filter(lambda x: x[0] != x[1] and random.uniform(0, 1) < 0.003, product(self.data_id, self.data_id)))
#     random.shuffle(self.data_id)
#     mid = len(self.data_id) // 2
#     self.data = list(zip(self.data_id[:mid], self.data_id[mid:]))
#     self.data += [(y, x) for x, y in self.data]


#     assert len(self.data) > 0, f'No training data could be obtained'

#     convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
#     num_channels = 3 if not transparent else 4

#     self.img_transform = transforms.Compose([
#         #transforms.Lambda(convert_image_fn),
#         # transforms.Lambda(partial(resize_to_minimum_size, image_size)),
#         transforms.Resize(self.image_size_dims),
#         #RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
#         transforms.ToTensor(),
#         # transforms.Lambda(expand_greyscale(transparent))
#     ])

#     self.texture_transform = transforms.Compose([
#       transforms.ToTensor()
#     ])

#   def __len__(self):
#       return len(self.data)

#   def __getitem__(self, idx):
#       id1, id2 = self.data[idx] #get a pair of datapoints from the list of pairings
#       #the image, pose and path locations equal the concat(corresponding root, id)
#       full_image_path1 = os.path.join(self.img_path, id1)
#       full_image_path2 = os.path.join(self.img_path, id2)

#       full_pose_path1 = os.path.join(self.pose_path, id1)
#       full_pose_path2 = os.path.join(self.pose_path, id2)

#       full_texture_path1 = os.path.join(self.texture_path, id1)
#       # full_texture_path2 = os.path.join(self.texture_path, id2)

#       #read in the source images (including pose and texture), convert to torch 
#       source_img = Image.open(full_image_path1) 
#       source_pose = Image.open(full_pose_path1)
#       source_texture = Image.open(full_texture_path1)
      
#       #read in the target images (including pose and texture), convert to torch 
#       target_img = Image.open(full_image_path2)
#       target_pose = Image.open(full_pose_path2)

#       #put them together
#       # print('source_pose size pre-transform', source_pose.size)
#       source_datapoint = (self.img_transform(source_img), self.img_transform(source_pose), self.texture_transform(source_texture))
#       target_datapoint = (self.img_transform(target_img), self.img_transform(target_pose))

#       return source_datapoint, target_datapoint