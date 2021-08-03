from stylegan2.stylegan import StyleGAN2
import numpy
import argparse
import os
import pickle
import shutil
import sys
import time
from functools import partial, reduce
from itertools import starmap
from multiprocessing import Pool
from subprocess import PIPE, Popen
from typing import Any, Callable, Dict, List, Tuple, Union, Iterable

import cv2
import json

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm
from UVTextureConverter import Atlas2Normal

from datasets import DeepFashion as DF



def parse_args() -> Tuple[Any]:
    default_version = '1.0.0'
    parser = argparse.ArgumentParser(description='Save model in its torchscript version for inference')
    parser.add_argument(
        '--out_path', type=str,
        default=os.path.join('.', 'checkpoints', f'scripted_model_1.0.0.pt'),
        help='file path to store scripted model'
    )
    parser.add_argument(
        '--model_dir', type=str,
        default=os.path.join('.', 'checkpoints', 'models', 'default'),
        help='path to model checkpoint directory, ' + \
             'eg. \'./checkpoints/models/dev-fixes\' (not a path to a .pt file!)'
    )
    parser.add_argument(
        '--image_size', type=int, default=[256, 256], nargs='+',
        help='image size to use in model, 1 or 2 comma separated ints'
    )
    # parser.add_argument(
    #     '--load_from', type=int, default=-1,
    #     help='checkpoint number for model, use \'-1\' for latest'
    # )
    args = parser.parse_args()

    # model_dir, name = os.path.split(args.model_dir)
    # base_dir = os.path.split(model_dir)[0]
    # image_size = (tuple(args.image_size[:2]) if len(args.image_size) >= 2
    #               else (*args.image_size, *args.image_size))

    # print('in parse', name, base_dir, args.out_path, image_size, args.load_from)
    return (args.model_dir, args.out_path, args.image_size)


def main(model_dir: str, model_save_path: str,
         image_size: Union[int, Tuple[int]]=(256, 256)):
    print('in main', model_dir, model_save_path, image_size)

    image_size = image_size if isinstance(image_size, int) else image_size[0]
   
    with open(os.path.join(model_dir, '.config.json')) as f:
        config = json.load(f)

    assert config is not None, "Config should not be loaded correctly"

    assert image_size == config['image_size'], "Image size and config image size should be equal"

    

    model = StyleGAN2(image_size = config['image_size'],
                network_capacity = config['network_capacity'],
                transparent = config['transparent'],
                fq_layers = config['fq_layers'],
                fq_dict_size = config['fq_dict_size'],
                fmap_max = config.pop('fmap_max', 512),
                attn_layers = config.pop('attn_layers', []),
                no_const = config.pop('no_const', False),
                lr_mlp = config.pop('lr_mlp', 0.1))

    pt_file = [file for file in os.listdir(model_dir) if file != '.config.json'][-1]
    assert pt_file.endswith('.pt'), "Model file should be .pt"
    pt_file = os.path.join(model_dir, pt_file)
    
    loaded_dict = torch.load(pt_file)
    model.load_state_dict(loaded_dict['GAN'])
    model.eval()

    #Example inputs
    scale_and_crop = DF.scale_and_crop(image_size)
    T = DF.get_transforms(scale_and_crop, is_tensor=False)
    to_tensor = torchvision.transforms.ToTensor()

    

    A_s = torch.randn(1, 3, 512, 512)
    P_t = torch.randn(1, 3, 256, 256)
    sample_input = (A_s, P_t)
    scripted_model = torch.jit.trace(model, sample_input)
    scripted_model.save(model_save_path)
    print(f'Model cripted and saved to {model_save_path}! Exiting...')


if __name__ == "__main__":
    args = parse_args()
    main(*args)