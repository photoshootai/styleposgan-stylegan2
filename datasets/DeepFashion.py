import os
import random
from functools import partial, reduce
from itertools import chain, repeat, starmap
from math import ceil
from typing import Iterable, Tuple, Union, Set

import torch
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
        image_tensor = resize_to_minimum_size(max(self.img_size), image_tensor)
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

cat_map = {'sex': 0, 'clothing_category': 1, 'model': 3, 'clothing_id': 4, 'idx': 5, 'pose': 6}

def extract_prop(file_name: str, props: Set[str]={'model'}) -> Tuple[str]:
    """
    {sex}_{clothing_category}_id_{model}_{clothing_id}_{idx}_{pose}.jpg
    """
    idxs = (v for k, v in cat_map.items() if k in props)
    return tuple(compress(os.path.splitext(file_name)[0].split('_'), idxs))


def random_shuffle(files: List[str], seed: int=42) -> Iterable[Tuple[str]]:
    if len(files) < 2:
        return []

    random.seed(seed)
    random.shuffle(file_names)

    mid = len(file_names) // 2
    return chain(zip(file_names[:mid], file_names[mid:]), zip(file_names[mid:], file_names[:mid]))


def conditional_shuffle(files: Iterable[str], props: Set[str], seed: int=42) -> Iterable[Tuple[str]]:
    extract = partial(extract_prop, props=props)
    opts = set(extract(f) for f in files)
    groups = (list(filter(lambda f: opt == extract(f), files)) for opt in opts)
    pairs = chain(random_shuffle(group, seed=seed) for group in groups)
    return (pair for pair in pairs if pair)


class DeepFashionDataset(Dataset):
    def __init__(self, data_dir: str, image_size: Union[Tuple[int], int, float] = (512, 512),
                 scale_crop: bool = True, transparent: bool = False, seed: int = 42,
                 aug_prob: float = 0.0, pair_method: str = 'random', props: Union[Set[str], None] = None) -> None:
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

        file_names = (d.name for d in os.scandir(self.img_dirs[0]))

        if not props:
            if pair_method == 'random':
                self.data = list(random_shuffle(list(file_name), seed))
            elif pair_method == 'P':
                self.data = list(conditional_shuffle(files, props={'sex', 'model'}, seed=seed))
            elif pair_method == 'P_and_A':
                self.data = list(conditional_shuffle(files, props={'sex', 'clothing_category', 'model', 'clothing_id'}, seed=seed))
            else:
                raise Exception('Please ensure pair method is one of (\'random\' | \'P\' | \'P_and_A\')')   
        elif props:
            self.data = list(conditional_shuffle(files, props=props, seed=seed))
        else:
            raise Exception(f'Please ensure props set is valid subset of {cat_map.keys()}')
 
        
        self.data_len = len(self.data)
        print('there are {self.data_len} pairs in this dataset')
        assert self.data_len > 0, 'Empty dataset'

        # self.to_rgb = convert_transparent_to_rgb if transparent else lambda x: x
        # self.expand_greyscale = expand_greyscale(transparent)
        self.scale_and_crop = scale_and_crop(self.img_size) if scale_crop else transforms.Resize(self.img_size)
        self.transforms = (
            transforms.Compose([  # For source images
                transforms.ToTensor(),  # [0, 255] gets mapped to [0.0, 1.0]
                # convert to 3 channels (squash alpha) -> don't need
                # transforms.Lambda(self.to_rgb),
                transforms.Lambda(self.scale_and_crop)
            ]),
            transforms.Compose([  # For Pose maps
                # this will screw up pose segmentation since {0,..., 24} gets mapped to [0.0, 1.0]
                transforms.ToTensor(),
                # guaranteed to be 3 channels
                transforms.Lambda(self.scale_and_crop)
            ]),
            transforms.Compose([  # For texture Maps
                transforms.ToTensor()  # converts [0, 255] to float[0.0, 1.0]
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
        src_im_paths = starmap(os.path.join, zip(self.img_dirs, repeat(src_file)))  # (src, pose, txt)
        targ_im_paths = starmap(os.path.join, zip(self.img_dirs, repeat(targ_file, 2)))  # (src, pose)

        src_imgs = map(Image.open, src_im_paths)
        targ_imgs = map(Image.open, targ_im_paths)

        source_set = (f(x) for f, x in zip(self.transforms, src_imgs))
        target_set = (f(x) for f, x in zip(self.transforms, targ_imgs))

        return tuple(source_set), tuple(target_set)

