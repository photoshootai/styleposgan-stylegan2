import os
# from posixpath import basename
import time
import random
import re
from functools import partial, reduce
from itertools import chain, repeat, starmap, compress, permutations
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
    def __init__(self, img_size: Union[Tuple[int], int]) -> None:
        if isinstance(img_size, int):
            img_size = (img_size, img_size) 
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
        # transforms.Compose([  # For source images
        #     transforms.ToTensor(),  # [0, 255] gets mapped to [0.0, 1.0]
        #     # convert to 3 channels (squash alpha) -> don't need
        #     # transforms.Lambda(self.to_rgb),
        #     transforms.Lambda(scale_and_crop)
        # ]),
        # transforms.Compose([  # For Pose maps
        #     # this will screw up pose segmentation since {0,..., 24} gets mapped to [0.0, 1.0]
        #     transforms.ToTensor(),
        #     # guaranteed to be 3 channels
        #     transforms.Lambda(scale_and_crop)
        # ]),
        # transforms.Compose([  # For texture Maps
        #     transforms.ToTensor()  # converts [0, 255] to float[0.0, 1.0]
        # ])
    # )


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

def conditional_shuffle(files: Iterable[str], props: Set[str], n_threads: int=6) -> Iterable[Tuple[str]]:
    extract = partial(extract_prop, props=props)
    opts = set(extract(f) for f in files)
    with Pool(n_threads) as P:
        groups = P.starmap(filter_opt, zip(opts, repeat(files), repeat(props)))
        pairs = P.starmap(permutations, zip(groups, repeat(2)))
    
    pruned_pairs = (pair for pair in chain.from_iterable(pairs) if pair)
    return pruned_pairs

class DeepFashionDataset(Dataset):
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

        in_data_dir = partial(os.path.join, data_dir)
        self.sub_dirs = ('SourceImages', 'PoseMaps', 'TextureMaps')
        self.img_dirs = tuple(map(in_data_dir, self.sub_dirs))
        assert all(map(os.path.isdir, self.img_dirs)), 'Some requisite image directories not found'

        file_names = [d.name for d in os.scandir(self.img_dirs[0])]

        if not props:
            if pair_method == 'random':
                props = {'sex'}
            elif pair_method == 'P':
                props = {'model'}
            elif pair_method == 'P_and_A':
                props = {'model', 'clothing_id'}
            else:
                raise Exception('Please ensure pair method is one of (\'random\' | \'P\' | \'P_and_A\')')  
        elif props:
            props = props
        else:
            raise Exception(f'Please ensure props set is valid subset of {cat_map.keys()}')

        pair_pickle = f'./data/pairs_{"_".join(sorted(list(props)))}_{os.path.basename(data_dir)}.pkl'        

        if os.path.isfile(pair_pickle):
            print("Loading pair annotations from pickle: ", pair_pickle)
            with open(pair_pickle, 'rb') as f:
                data = pickle.load(f)
                self.data = data
        else:
            start = time.time()
            self.data = list(conditional_shuffle(file_names, props))
            with open(pair_pickle, 'wb') as f:
                pickle.dump(self.data, f)
            print(f'Took {(time.time() - start):.4f} seconds to make pairs')

        # Shuffle pairs based on seed
        random.seed(seed)
        random.shuffle(self.data)
        self.data_len = len(self.data)
        print(f'there are {self.data_len} pairs in this dataset')
        assert self.data_len > 0, 'Empty dataset'

        # self.to_rgb = convert_transparent_to_rgb if transparent else lambda x: x
        # self.expand_greyscale = expand_greyscale(transparent)
        self.scale_and_crop = scale_and_crop(self.img_size) if scale_crop else transforms.Resize(self.img_size)
        self.transforms = get_transforms(self.scale_and_crop)

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
        # print(tuple(self.data[0]))
        # print(self.data[index])
        src_file, targ_file = self.data[index]
        # print(type(src_file))
        src_im_paths = starmap(os.path.join, zip(self.img_dirs, repeat(src_file)))  # (src, pose, txt)
        targ_im_paths = starmap(os.path.join, zip(self.img_dirs, repeat(targ_file, 2)))  # (src, pose)

        src_imgs = map(Image.open, src_im_paths)
        targ_imgs = map(Image.open, targ_im_paths)

        source_set = (f(x) for f, x in zip(self.transforms, src_imgs))
        target_set = (f(x) for f, x in zip(self.transforms, targ_imgs))

        return tuple(source_set), tuple(target_set)

