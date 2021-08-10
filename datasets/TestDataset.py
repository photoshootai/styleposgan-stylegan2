import os
# from posixpath import basename
import time
import random
import re
from functools import partial, reduce
from itertools import chain, repeat, starmap, compress, permutations, product
from math import ceil
from typing import Iterable, Tuple, Union, Set, List
from multiprocessing import Pool
from numpy import iterable

import torch
import pickle
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image


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
            raise Exception(
                f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))


def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return transforms.functional.resize(image, min_size)
    return image


class scale_and_crop(object):
    def __init__(self, img_size: Tuple[int]) -> None:
        self.img_size = img_size
        # self.crop = transforms.RandomCrop(img_size)

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:        
        # image_tensor = resize_to_minimum_size(max(self.img_size), image_tensor)
        _, h, w = image_tensor.shape  # should be applied after transforms.ToTensor()
        new_h, new_w = self.img_size

        min_img_dim = min(h, w)
        max_new_dim = max(new_h, new_w)

        ratio = max_new_dim / min_img_dim
        scaled_size = ceil(h * ratio), ceil(w * ratio)

        resized = transforms.Resize(scaled_size)(image_tensor)

        _, h, w = resized.shape
        mid = (w - new_w) // 2
        cropped = resized[:, :new_h, mid:w - mid]

        return cropped


cat_map = {'sex': 0, 'clothing_category': 1, 'const_1': 2, 'model': 3, 'clothing_id': 4, 'idx': 5, 'pose': 6}

def extract_prop(file_name: str, props: Set[str]={'model'}) -> Tuple[str]:
    """
    /PATH/{sex}_{clothing_category}_id_{model}_{clothing_id}_{idx}_{pose}.jpg
    """
    idxs = {v for k, v in cat_map.items() if k in props}
    mask = (1 if i in idxs else 0 for i in range(len(cat_map)))
    base_name = os.path.splitext(file_name)[0]
    match = re.match(r'([A-Z]+)\_(.+)\_(id)\_(\d+)\_(\d+)\_(\d+)\_(.+)', base_name)
    return tuple(compress(match.groups(), mask))

def filter_opt(opt, files, props):
    # list for multiprocessing compatibility generator preferred for single threaded
    return list(filter(lambda f: opt == extract_prop(f, props), files))

def conditional_shuffle(files: Iterable[str], props: Set[str], n_threads: int=6, with_replacement=False) -> Iterable[Tuple[str]]:
    perm_func = partial(permutations, r=2) if not with_replacement else partial(product, repeat=2)
    extract = partial(extract_prop, props=props)
    opts = set(extract(f) for f in files)
    with Pool(n_threads) as P:
        groups = P.starmap(filter_opt, zip(opts, repeat(files), repeat(props)))
        pairs = P.map(perm_func, groups)
    
    pruned_pairs = (pair for pair in chain.from_iterable(pairs) if pair)
    return pruned_pairs


def splice_unbatched(A_s, A_t):
    assert len(A_s.shape) == 3 and len(A_t.shape) == 3, "Inputs must not be batched" 
    
    #Face Splicing
    h1 = 220 #230 #303
    w1 = 303 #288 #220

    #Hand Slicing
    h2 = -70
    w2 = -232
    
    # print(A_s.size())
    # print(A_t.size())

    A_t[:, 0:h1, 0:w1] = A_s[:, 0:h1, 0:w1]  # face
    A_t[:, h2:-1, w2:-1] = A_s[:, h2:-1, w2:-1]  # hands

    return A_t

def splice_batched(A_s, A_t):
    
    assert len(A_s.shape) == 4 and len(A_t.shape) == 4, "Inputs must be batched" 
    #Face Splicing
    h1 = 220 #230 #303
    w1 = 303 #288 #220

    #Hand Slicing
    h2 = -70
    w2 = -232
    
    # print(A_s.size())
    # print(A_t.size())

    A_t[:, :, 0:h1, 0:w1] = A_s[:, :, 0:h1, 0:w1]  # face
    A_t[:, :, h2:-1, w2:-1] = A_s[:, :, h2:-1, w2:-1]  # hands

    return A_t


def get_transforms(scale_and_crop, is_tensor=False):
    t_set = (
        [transforms.Lambda(scale_and_crop)], # source images
        [transforms.Lambda(scale_and_crop)], # pose maps
        list(),                              # texture_maps
    )
    if not is_tensor:
        to_tensor = [transforms.ToTensor()]
        t_set = tuple(to_tensor + t for t in t_set)
            
    return tuple(transforms.Compose(t) for t in t_set)


class TestDataset(Dataset):
    def __init__(self, data_dir: str, image_size: Union[Tuple[int], int, float] = (512, 512),
                 scale_crop: bool = True, transparent: bool = False, seed: int = 42,
                 aug_prob: float = 0.0, pair_method: str = 'P_and_A',
                 props: Union[Set[str], None] = None) -> None:
        """
        Arguments:
            data_dir [str]: Path to data directory, should have subfolders {SourceImages, PoseMaps, TextureMaps}
            image_size [opt int or Tuple=(512, 512)]: Size to transform PoseMap images and SourceImages images
            scale_crop [opt bool=True]: scale and crop instead of resize
            transparent [opt bool=False]: if image has transparency channel
            seed [opt int=42]: The seed to shuffle data with
            aug_prob [opt float=0.0]: probability of augmentation on images in range [0.0, 1.0]
            pair_method [opt str in {'random', 'P', 'P_and_A'}]: How to pair source/target images
                        * 'random': randomly pair source and target
                        * 'P': Randomly pair person with themselves on different garment and pose
                        * 'P_and_A': Randomly pair person with themselves (with same garment), different pose
            props  [opt set subset of {'sex', 'clothing_category', 'model', 'clothing_id', 'idx', 'pose'}]:
                    : Instead of predefined pair_method, use this to use a custom pairing. Keep None if using pair_method.
        
        """
        self.n_chan = 4 if transparent else 3
        self.img_size = image_size if isinstance(image_size, Tuple) else (int(image_size), int(image_size))

        user_data_dir = partial(os.path.join, data_dir, 'UserImages')
        self.sub_dirs = ('SourceImages', 'PoseMaps', 'TextureMaps')
        
        self.user_img_dirs = tuple(map(user_data_dir, self.sub_dirs))
        

        target_data_dir = partial(os.path.join, data_dir, 'TargetImages')
        self.target_sub_dirs = ('PoseMaps', 'TextureMaps')

        self.target_img_dirs = tuple(map(target_data_dir, self.target_sub_dirs))
        

        print(self.user_img_dirs)
        print(self.target_img_dirs)

        assert all(map(os.path.isdir, self.user_img_dirs)), 'Some requisite UserImage directories not found'
        assert all(map(os.path.isdir, self.target_img_dirs)), 'Some requisite TargetImage directories not found'


        user_file_names = [f.name for f in os.scandir(self.user_img_dirs[0]) if f.is_file()]
        
        target_pose_maps = [f.name for f in os.scandir(self.target_img_dirs[0]) if f.is_file()]
        target_texture_maps = [f.name for f in os.scandir(self.target_img_dirs[1]) if f.is_file()]


        print("Length of user_file_names: ", len(user_file_names))
        print("Length of target_pose_maps: ", len(target_pose_maps))
        print("Length of target_texture_maps: ", len(target_texture_maps))

        self.data = list(product(user_file_names, target_pose_maps, target_texture_maps))

        print(len(self.data))

        #Transforms
        self.scale_and_crop = scale_and_crop(self.img_size) if scale_crop else transforms.Resize(self.img_size)
        self.transforms = get_transforms(self.scale_and_crop)



    def __len__(self):
        """
        length of entire dataset 
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor]]:
        """
        Arguments:
            index [int]: The index in data from which to get data

        Returns:
            Source, Target [(tensor; (n_chan, img_size, img_size))]
        """
        # print(tuple(self.data[0]))
        # print(self.data[index])
        user_image_file, targ_pose, targ_texture = self.data[index]

        # print("Filenames are: ")
        # print(user_image_file)
        # print(targ_pose)
        # print(targ_texture)

        # print(type(src_file))
        src_im_paths = starmap(os.path.join, zip(self.user_img_dirs, repeat(user_image_file)))  # (src, pose, txt)
        tgt_im_paths = starmap(os.path.join, zip(self.target_img_dirs, [targ_pose, targ_texture]))  # (src, pose, txt)


        # print(list(src_im_paths))
        # print(list(tgt_im_paths))

        src_im_paths = list(src_im_paths)
        print(src_im_paths)
        src_imgs = map(Image.open, src_im_paths)
        targ_imgs = map(Image.open, tgt_im_paths)

        source_set = (f(x) for f, x in zip(self.transforms, src_imgs))
        target_set = (f(x) for f, x in zip(self.transforms[1:], targ_imgs))

        # print([x.shape for x in source_set])
        # print([x.shape for x in target_set])

        (I_s, _, A_s), (P_t, A_t) = source_set, target_set
        spliced_texture = splice_unbatched(A_s, A_t)
    
        return (I_s, spliced_texture, A_t, P_t)


