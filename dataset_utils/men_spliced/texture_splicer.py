from PIL import Image
import numpy as np

import multiprocessing
import os
# from posixpath import basename
import time
import random
import re
from functools import partial, reduce
from itertools import chain, cycle, repeat, starmap, compress, permutations
from math import ceil
from typing import Iterable, Tuple, Union, Set, List
from multiprocessing import Pool
from numpy import iterable
from numpy.lib.function_base import _parse_input_dimensions
import torchvision
from torchvision import transforms

from torchvision.utils import save_image
from shutil import copy2
from PIL import Image

from datasets import conditional_shuffle

NUM_CORES = multiprocessing.cpu_count()


#CONSTANTS

SEED = 42
#Face Splicing
h1= 220 #230 #303
w1= 303 #288 #220

#Hand Slicing
h2= -70
w2= -232

to_tensor = transforms.ToTensor()


def show_pruned_pairs(base_src_images_path, pruned_pairs):

    for a, b in pruned_pairs:
        print(a, b)
        Image.open(base_src_images_path + "/" + a).show()
        Image.open(base_src_images_path + "/" + b).show()
        input("Press enter for next")


def get_spliced_face_and_hands_on_target(face_tex_map, target_tex_map):
    face_tex_map = to_tensor(Image.open(face_tex_map))
    target_tex_map = to_tensor(Image.open(target_tex_map))

    #show_img(face_tex_map)

    face, hands = splice_face_and_hands(face_tex_map)
    z_spliced_toegther = paste_face_and_hands_onto_target(face, hands,target_tex_map)

    return z_spliced_toegther

def splice_face_and_hands(src_tex_map):

    # #Batched
    # contains_face = src_tex_map[:,:,0:h1,0:w1] #in format of B,C,H,W
    # contains_hands = src_tex_map[:,:,h2:-1,w2:-1]

    #Non-batched
    face = src_tex_map[0:h1,0:w1,:] #in format of B,C,H,W
    hands = src_tex_map[h2:-1,w2:-1, :]

    return face, hands

def paste_face_and_hands_onto_target(face, hands, tg_tex_map):
    # tg_tex_map.setflags(write=1)
    tg_tex_map[0:h1,0:w1,:] = face
    tg_tex_map[h2:-1,w2:-1,:] = hands

    z_spliced_toegther = tg_tex_map

    return z_spliced_toegther


## Functions to help create spliced dataset

# def functional_hell(img_dirs, new_img_dirs):
    #  def save_new_tex_map(src, targ, dest):
    #     save_image(
    #         get_spliced_face_and_hands_on_target(src, targ),
    #         dest
    #     )
    #     return 1
        
    # def copy_src_file(src, _, dest):
    #     copy2(src, dest)
    #     return 1

    # def copy_targ_file(_, targ, dest):
    #     copy2(targ, dest)
    #     return 1

    # def apply_func(f, x):
    #     return f(*x)

    # def make_trip(i, j, st):
    #     s, t = st
    #     src = os.path.join(img_dirs[i], s)
    #     targ = os.path.join(img_dirs[i], t)
    #     dest = os.path.join(new_img_dirs[j], s)
    #     return src, targ, dest

    # def make_all_trips(st):
    #     return starmap(make_trip, zip(cycle(range(3)), range(4), repeat(st))) # -> [(s, t, d), 4]


    # def asdf(func_it, std):
    #     return starmap(apply_func, zip(func_it, repeat(std)))

    # func_it = (copy_src_file, copy_targ_file, save_new_tex_map, copy_targ_file)
    # all_trips = chain.from_iterable(map(make_all_trips, pruned_pairs))
    # tot = sum(chain.from_iterable(starmap(asdf, zip(repeat(func_it), all_trips))))

def save_s_t_pair(st, img_dirs, new_img_dirs):
    src, targ = st
    copy2(os.path.join(img_dirs[0], src),  os.path.join(new_img_dirs[0], src))
    copy2(os.path.join(img_dirs[1], targ),  os.path.join(new_img_dirs[1], src))
    save_image(get_spliced_face_and_hands_on_target(os.path.join(img_dirs[2], src), os.path.join(img_dirs[2], targ)), os.path.join(new_img_dirs[2], src))
    copy2(os.path.join(img_dirs[0], targ),  os.path.join(new_img_dirs[-1], src))

def create_texture_spliced_dataset(pruned_pairs):
    start = time.time()
    data_dir = "./data/DeepFashionMenOnlyCleaned"

    in_data_dir = partial(os.path.join, data_dir)
    sub_dirs = ['SourceImages', 'PoseMaps', 'TextureMaps']
    img_dirs = tuple(map(in_data_dir, sub_dirs))
    assert all(map(os.path.isdir, img_dirs)), 'Some requisite image directories not found'

    files = [d.name for d in os.scandir(img_dirs[0])]

    print(len(files))


    parent_dir = os.path.join('.', 'data', 'DeepFashionMenOnlySpliced')
    if not os.path.isdir(parent_dir):
        os.mkdir(parent_dir)
    new_img_dirs = tuple(map(partial(os.path.join, parent_dir), sub_dirs + ['TargetImages']))
    for d in new_img_dirs:
        if not os.path.isdir(d):
            os.mkdir(d)
    
    new_f_names = list()

    save_st_w_dirs = partial(save_s_t_pair, img_dirs=img_dirs, new_img_dirs=new_img_dirs)
    with Pool(NUM_CORES) as P:
        P.map(save_st_w_dirs, pruned_pairs,chunksize=50)
    # print(tot)

    # for src, targ in pruned_pairs:
    #     copy2(os.path.join(img_dirs[0], src),  os.path.join(new_img_dirs[0], src))
    #     copy2(os.path.join(img_dirs[1], targ),  os.path.join(new_img_dirs[1], src))
    #     save_image(get_spliced_face_and_hands_on_target(os.path.join(img_dirs[2], src), os.path.join(img_dirs[2], targ)), os.path.join(new_img_dirs[2], src))
    #     copy2(os.path.join(img_dirs[0], targ),  os.path.join(new_img_dirs[-1], src))
    #     new_f_names.append(src)
    
    print('n_src_im', len(os.listdir(new_img_dirs[0])))
        
    for src, _ in pruned_pairs:
        assert all(map(os.path.isfile, starmap(os.path.join, zip(new_img_dirs, repeat(src))))), f'Missing file {src}'


    print(f'Took {(time.time() - start):.4f} seconds to make pairs')

def main():
    import pickle

    pkl = "./data/pairs_clothing_id_model_DeepFashionMenOnlyCleaned.pkl" #'./temp.pkl'
    if os.path.isfile(pkl):
        print('Pickle exists')
        with open(pkl, 'rb') as f:
            pruned_pairs = pickle.load(f)
        print(f"Pairs : {len(pruned_pairs)}")

        show_pruned_pairs("./data/DeepFashionMenOnlyCleaned/SourceImages", pruned_pairs)


        exit()
    else:
        print('Pickle does not exist')
        data_dir = "./data/DeepFashionMenOnlyCleaned"

        in_data_dir = partial(os.path.join, data_dir)
        sub_dirs = ['SourceImages', 'PoseMaps', 'TextureMaps']
        img_dirs = tuple(map(in_data_dir, sub_dirs))
        assert all(map(os.path.isdir, img_dirs)), 'Some requisite image directories not found'

        files = [d.name for d in os.scandir(img_dirs[0])]
        props = {'model'}
        pruned_pairs = list(conditional_shuffle(files, props, 1, with_replacement=True))
        print(f"Pairs : {len(pruned_pairs)}")

        random.seed(SEED)
        random.shuffle(pruned_pairs) #(A, A)
        with open(pkl, 'wb') as f:
            pickle.dump(pruned_pairs, f)
    
    create_texture_spliced_dataset(pruned_pairs)

if __name__ == "__main__":
    main()