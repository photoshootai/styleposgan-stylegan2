
from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np

import multiprocessing

NUM_CORES = multiprocessing.cpu_count()
#CONSTANTS

#Face Splicing
h1= 230
w1= 288

#Hand Slicing
h2= -70
w2= -232


def show_img(img):
    imshow(np.asarray(img))

def get_spliced_face_and_hands_on_target(face_tex_map, target_tex_map):
    face_tex_map = np.array(Image.open(face_tex_map))
    target_tex_map = np.array(Image.open(target_tex_map))

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
    tg_tex_map.setflags(write=1)
    tg_tex_map[0:h1,0:w1,:] = face
    tg_tex_map[h2:-1,w2:-1,:] = hands

    z_spliced_toegther = tg_tex_map


def main():
    list_of_pairs = [(1, 2), ("a", "b")]
    with multiprocessing.Pool(NUM_CORES) as p:
        #p.starmap(get_spliced_face_and_hands_on_target, list_of_pairs)
        pass

if __name__ == "__main__":
    main()