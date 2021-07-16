import argparse
import os
import shutil
import pickle
from functools import partial
from subprocess import Popen, PIPE
from models import ANet, PNet
from datasets import DeepFashion as DF

def run_densepose_network(source_img_dir: str, output_pkl: str,
                          apply_net_path: str='apply_net.py',
                          output_stdout: bool=True) -> int:
    """
    Run denspose on single image path or directory of images and produce a
    pickle file.

    Arguments:
        source_img_dir [str]: path to directory of imgs (or path to img -- bad!)
        output_pkl [str]: path to save output pickle
        apply_net_path [opt str=apply_net.py]: path to densepose apply_net.py
        output_stdout [opt bool=True]: whether or not to output densepose stdout

    Returns:
        n_files [int]: number of files densepose was run on

    Side Effects:
        Saves large pickle (~ 10mb / image) to given path
    """
    MODEL_URL = 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl'

    call =  ['python3', apply_net_path, 'dump', '-v']
    call += ['--output', output_pkl]
    call += [f'configs/{MODEL_URL.split("/")[-3]}.yaml']
    call += [MODEL_URL, source_img_dir]

    with Popen(call, stdout=PIPE, stderr=PIPE) as proc:
        if output_stdout:
            print(proc.stdout.read().decode('utf-8'))
        print(proc.stderr.read().decode('utf-8'))

    if os.path.isfile(source_img_dir):
        n_files = 1
    else:
        n_files = len((1 for n in os.scandir(source_img_dir) if os.path.isfile(os.path.join(source_img_dir, n))))

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
        # x, y = np.where(IUV[0, ...] == part_id)  # select pixels belonging to part {part_id}
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


def parse_args():
    parser = argparse.ArgumentParser(description='Styleposegan inference')
    parser.add_argument('-v', action='store_true', help='verbose output')
    parser.add_argument('--source', type=str, help='path to the source image(s)')
    parser.add_argument('--target', type=str, help='path to the target image(s)')
    parser.add_argument(
        '--batched', action='store_true',
        help='use if the input source/target paths are for directories'
    )
    parser.add_argument(
        '--outputs', type=str,
        default=os.path.join('.', 'data', 'inference_outputs'),
        help='directory to store outputs and temporary inference files'
    )
    parser.add_argument(
        '--apply', type=str,
        default=os.path.join('.', 'densepose', 'apply_net.py'),
        help='path to denspose apply_net path'
    )

    args = parser.parse_args()
    return (args.source, args.target, args.batched, args.outputs,
            args.apply, args.v)


def main():
    (src, targ, batched, out_dir, apply_net, verb) = parse_args()

    if batched:
        print('batched inference is not supported yet')
        exit()
        assert os.path.isdir(src), f'Expected {src} to be a directory'
        assert os.path.isdir(targ), f'Expected {targ} to be a directory'

        exp_targ_names = (os.path.join(targ, d.name) for d in os.scandir(src))
        assert all(map(os.path.isfile, exp_targ_names)), \
               'Expected source and target dirs to have matching file names'

    if not os.path.isdir(out_dir):
        verb and print('output directory doesn\'t exist, creating now...')
        os.mkdir(out_dir)

    src_pkl = os.path.join(out_dir, 'src.pkl')
    targ_pkl = os.path.join(out_dir, 'targ.pkl')
    
    n_src_files = run_densepose_network(src, src_pkl, apply_net, output=verb)
    n_targ_files = run_densepose_network(targ, targ_pkl, apply_net, output=verb)

    assert n_src_files == n_targ_files, 'src/targ had different # of files'

    with open(src_pkl, 'rb') as f:
        verb and print(f'operating on pickle {src_pkl}')
        src_data = pickle.load(f)
    with open(targ_pkl, 'rb') as f:
        verb and print(f'operating on pickle {targ_pkl}')
        targ_data = pickle.load(f)

    data_map = {'src': src_data, 'targ': targ_data}
    data_pairs = dict() # {img_name: {'src' (I, P, A), 'targ': (I, P, A)}}
    for data in ('src', 'targ'):
        for d in data_map[data]:
            file_path = d['file_name']
            file_name = os.path.basename(file_path)
                
            if 'pred_densepose' not in d:
                print(f'Error computing densepose for {file_name}')
                continue

            I = torch.from_numpy(cv2.imread(file_path)).long()

            x1, y1, x2, y2 = d['pred_boxes_XYXY'][0]
            x_off, x_delta = int(x1), int(x2 - x1)
            y_off, y_delta = int(y1), int(y2 - y1)

            T = d['pred_densepose'][0]
            IUV = torch.cat((T.labels.unsqueeze(0), T.uv.clamp(0, 1)), dim=0).cpu()
            P = torch.zeros(I.shape)
            P[y_off:y_off + y_delta, x_off:x_off + x_delta] = IUV.permute(1, 2, 0)
            
            atlas = create_texture_atlas(P.permute(2, 0, 1), I)
            A = convert_atlas_to_SURREAL(atlas)

            triplet = (I, P, A)

            if file_name not in data_pairs:
                data_pairs[file_name] = dict()
            data_pairs[file_name][data] = (I, P, A)

    # data_pairs {img_name: {'src': (I, P, A), 'targ': (I, P, A)}, ...}
    for img, pair in data_pairs.items():
        
        I_dash_s = None
        I_dash_s_to_t = None





if __name__ == '__main__':
    main()
    
