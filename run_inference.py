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
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm
from UVTextureConverter import Atlas2Normal

from datasets import DeepFashion as DF
from models import ANet, PNet
from stylegan2.stylegan import ModelLoader


def run_densepose_network(source_img_dir: str, output_pkl: str,
                          dp_path: str=os.path.join('.', 'densepose'),
                          stdout: bool=True) -> int:
    """
    Run denspose on single image path or directory of images and produce a
    pickle file.

    Arguments:
        source_img_dir [str]: path to directory of imgs (or path to img -- bad!)
        output_pkl [str]: path to save output pickle
        dp_path [opt str=apply_net.py]: path to densepose folder
        stdout [opt bool=True]: whether or not to output densepose stdout

    Returns:
        n_files [int]: number of files densepose was run on

    Side Effects:
        Saves large pickle (~ 10mb / image) to given path
    """
    MODEL_URL = 'https://dl.fbaipublicfiles.com/densepose/' + \
                'densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl'

    apply_net_path = os.path.join(dp_path, 'apply_net.py')
    config_path = os.path.join(dp_path, 'configs',
                               f'{MODEL_URL.split("/")[-3]}.yaml')

    call =  ['python3', apply_net_path, 'dump', '-v']
    call += ['--output', output_pkl]
    call += [config_path]
    call += [MODEL_URL, source_img_dir]

    with Popen(call, stdout=PIPE, stderr=PIPE) as proc:
        if stdout:
            print(proc.stdout.read().decode('utf-8'))
        print(proc.stderr.read().decode('utf-8'))

    if os.path.isfile(source_img_dir):
        n_files = 1
    else:
        n_files = len((1 for n in os.scandir(source_img_dir)
                      if os.path.isfile(os.path.join(source_img_dir, n))))

    return n_files


def create_texture_atlas(IUV: torch.tensor,
                         XYZ: torch.tensor, N: int=200) -> torch.tensor:
    """
    Take in an IUV map generated by densepose, and a source image and produce
    a texture atlas that can be converted to a SURREAL texture map.

    Arguments:
        IUV [tensor; (3 x H x W)]: Posemap IUV coordinates
        XYZ [tensor; (H x W x 3)]: Original RGB Image tensor
        N   [optional int]: size of each texture segment (N x N x 3)

    Returns:
        Custom Atlas [tensor; (4N, 6N, 3)]: A texture atlas

    Side Effects:
        None
    Note: Densepose Default atlas is of shape (6N x 4N x 3), requires returned
    atlas to be transposed to apply to pose using apply_net.py
    """
    # 24 = (6 x 4) parts of size (N x N x [R, G, B])
    atlas = torch.zeros(4 * N, 6 * N, 3)
    I, U, V = IUV  # IUV(I in {0, ..., 24};  U, V in [0, 1]^{m \times n})
    parts = []

    for part_id in range(1, 25):
        part = torch.zeros(N, N, 3).long()  # Each part is N x N x [R, G, B]
        x, y = np.where(IUV[0] == part_id)  # select pixels belonging to part {part_id}
        x_idx = (U[x, y] * (N - 1)).long()  # must be long to index into tensor
        y_idx = (1 - (V[x, y]) * (N - 1)).long()  # x ∝ U; y ∝ (1 - V)

        part[x_idx, y_idx] = XYZ[x, y, :]  # copy RGB pixels from image on [x, y] mask
        parts.append(part)

    # 4:6 (transpose of densepose default of 6:4)
    for i in range(6):
        for j in range(4):
            # inverse of ./densepose/vis/densepose_results_textures.py:get_texture()::line 58
            atlas[(N * j):N * (j + 1), (N * i):N * (i + 1)] = parts[6 * j + i]

    return atlas


def convert_atlas_to_SURREAL(atlas: torch.tensor) -> torch.tensor:
    """
    Convert a (transposed) denspose atlas format (4N x 6N; N=200) into an SMPL
    appearance map (512 x 512).

    Arguments:
        atlas [tensor; (800, 1200, 3)]: the (transposed) densepose atlas

    Returns:
        normal_tex [tensor; (512, 512, 3)]: the SMPL format appearance map

    Side Effects:
        Adds arbitrarily to the computation graph of atlas, detach if you plan
        on a backwards pass.
    """
    atlas_tex_stack = Atlas2Normal.split_atlas_tex(atlas.numpy())
    converter = Atlas2Normal(atlas_size=200, normal_size=512)
    normal_tex = converter.convert(atlas_tex_stack)
    return normal_tex * 255.


def generate_ipa(d: Dict[str, Any], T: Iterable[Callable]) -> Tuple[torch.Tensor]:
    """
    Create I, P, A from densepose vis dict.
    Arguments:
        d: densepose image dict vis

    Returns:
        IPA [Tuple[Tensor]; (I, P, A)]:
            image tensor (I), pose map tensor (P), appearance map tensor (A)
    
    Side EFfects:
        Loads source image from file path, will error if path no longer exists.
    """
    file_path = d['file_name']
    file_name = os.path.basename(file_path)

    if 'pred_densepose' not in d:
        return (None, None, None)

    I = torch.from_numpy(cv2.imread(file_path)).long()

    x1, y1, x2, y2 = d['pred_boxes_XYXY'][0]
    x_off, x_delta = int(x1), int(x2 - x1)
    y_off, y_delta = int(y1), int(y2 - y1)

    T = d['pred_densepose'][0]
    IUV = torch.cat((T.labels.unsqueeze(0), T.uv.clamp(0, 1)), dim=0)
    P = torch.zeros(I.shape)
    P[y_off:y_off + y_delta, x_off:x_off + x_delta] = IUV.permute(1, 2, 0)

    atlas = create_texture_atlas(P.permute(2, 0, 1).cpu(), I)
    A = torch.from_numpy(convert_atlas_to_SURREAL(atlas))
    return file_name, (t(x).cuda() for t, x in zip(T, (I, P, A))


def parse_args():
    """
    Returns user input command line args in expected order for main function.

    Arguments:
        None

    Returns:
        source [str]: path to src image or dir
        target [str]: path to targ image or dir
        outputs [str]: path to output dir
        densepose [str]: path to densepose folder from detectron2
        image_size [int]: image size to use in model forward
        batched [bool]: set to true if src/targ are directories
        batch_size [int]: set size for model forward batch if --batched is used
        v [bool]: set true if you want verbose output
    """
    parser = argparse.ArgumentParser(description='Styleposegan inference')
    parser.add_argument('--source', type=str, help='path to the source image(s)')
    parser.add_argument('--target', type=str, help='path to the target image(s)')
    parser.add_argument(
        '--outputs', type=str,
        default=os.path.join('.', 'data', 'inference_outputs'),
        help='directory to store outputs and temporary inference files'
    )
    parser.add_argument(
        '--model_dir', type=str,
        default=os.path.join('.', 'results', 'default'),
        help='path to model checkpoint directory'
    )
    parser.add_argument(
        '--densepose', type=str, default=os.path.join('.', 'densepose'),
        help='path to denspose folder'
    )
    parser.add_argument(
        '--image_size', type=int, default=[256, 256], nargs='+',
        help='image size to use in model forward'
    )
    parser.add_argument(
        '--batched', action='store_true',
        help='use if the input source/target paths are for directories'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='if batched, use batch_size for model forward'
    )
    parser.add_argument('-v', action='store_true', help='verbose output')

    args = parser.parse_args()
    image_size = (tuple(args.image_size[:2]) if len(args.image_size) >= 2
                  else (*args.image_size, *args.image_size))
    return (args.source, args.target, args.outputs, args.model_dir,
            args.densepose, image_size, args.batched, args.batch_size, args.v)


def main(src: str, targ: str,
         out_dir: str=os.path.join('.', 'data', 'inference_outputs'),
         model_dir: str=os.path.join('.', 'results', 'default'),
         densepose_path: str=os.path.join('.', 'densepose'),
         image_size: Union[Tuple[int], int]=(256, 256),
         batched: bool=False, batch_size: int=4,
         verb: bool=False) -> None:
    """
    """
    curr_dir = os.getcwd()  # keep track of current user directory
    os.chdir(densepose_path)  # change to denspose dir for imports
    sys.path.append(densepose_path)  # Add densepose to path to evaluate on

    if batched:
        print('batched inference is not supported yet')
        exit()
        assert os.path.isdir(src), f'Expected {src} to be a directory'
        assert os.path.isdir(targ), f'Expected {targ} to be a directory'

        exp_targ_names = (os.path.join(targ, d.name) for d in os.scandir(src))
        assert all(map(os.path.isfile, exp_targ_names)), \
               'Expected source and target dirs to have matching file names'
    else:
        assert os.path.isfile(src), f'Expected {src} to be a file'
        assert os.path.isfile(targ), f'Expected {targ} to be a file'
        verb and print('batch_size automatically set to 1')
        batch_size = 1

    if not os.path.isdir(out_dir):
        verb and print('output directory doesn\'t exist yet, creating now...')
        os.mkdir(out_dir)

    src_pkl = os.path.join(out_dir, 'src.pkl')
    targ_pkl = os.path.join(out_dir, 'targ.pkl')

    verb and print('computing pose maps')
    n_src_files = 1#run_densepose_network(src, src_pkl, densepose_path, verb)
    n_targ_files = 1#run_densepose_network(targ, targ_pkl, densepose_path, verb)

    assert n_src_files == n_targ_files, 'src/targ had different # of files'

    with open(src_pkl, 'rb') as f1, open(targ_pkl, 'rb') as f2:
        verb and print('loading densepose results')
        src_data, targ_data = pickle.load(f1), pickle.load(f2)

    scale_and_crop = DF.scale_and_crop(image_size)
    T = DF.get_transforms(scale_and_crop, is_tensor=True)

    data_map = {'src': src_data, 'targ': targ_data}
    data_pairs = dict() # {img_name: {'src' (I, P, A), 'targ': (I, P, A)}}
    for data in ('src', 'targ'):
        verb and print(f'generating {data} I, P, A tensors')
        for d in data_map[data]:
            file_name, ipa = generate_ipa(d, T)

            if not all(t is not None for t in ipa):
                print(f'Error computing densepose for {file_name}')
                data_pairs[file_name] = None
                continue

            if file_name not in data_pairs:
                data_pairs[file_name] = dict()

            data_pairs[file_name][data] = ipa

    # data_pairs {img_name: {'src': (I, P, A), 'targ': (I, P, A)}, ...}
    batch_size = min(batch_size, n_src_files)
    verb and print(f'loading latest model at {model_dir}')
    model_name = os.path.basename(model_dir)
    model = ModelLoader(base_dir=model_dir, name=model_name, batch_size=batch_size)
    
    results_dir = os.path.join(out_dir, 'results')
    if not os.path.isdir(results_dir):
        verb and print('creating results dir {results_dir}')
        os.mkdir(results_dir)

    empty = (None, None)
    data_iter = iter(data_pairs.items())
    img, pair = next(data_iter, empty)
    while img is not None:
        print(pair)
        (I_s, P_s, A_s), (I_t, P_t, _) = tuple(*pair.values())

        I_s = I_s.unsqueeze(0)
        P_s = P_s.unsqueeze(0)
        A_s = F.interpolate(A_s.unsqueeze(0), size=image_size)
        I_t = I_t.unsqueeze(0)
        P_t = P_t.unsqueeze(0)

        (image_size, I_dash_s, I_dash_s_to_t, I_dash_s_ema,
         I_dash_s_to_t_ema) = model.generate((I_s, P_s, A_s), (I_t, P_t, _))

        regular = torch.cat(
            (I_s, P_s, A_s, I_t, P_t, I_dash_s, I_dash_s_to_t), dim=0
        )

        ema = torch.cat(
            (I_s, P_s, A_s, I_t, P_t, I_dash_s_ema, I_dash_s_to_t_ema), dim=0
        )

        # final_img = torch.cat((regular, ema), dim=0)  # to concat reg + ema together
        reg_save_path = os.path.join(results_dir, f'{img}_inference.jpg')
        ema_save_path = os.path.join(results_dir, f'{img}_inference_EMA.jpg')

        torchvision.utils.save_image(regular, reg_save_path, nrow=batch_size)
        torchvision.utils.save_image(ema, ema_save_path, nrow=batch_size)
    
    # go back to previous directory
    os.chdir(curr_dir)


if __name__ == '__main__':
    main(*parse_args())
