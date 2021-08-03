from itertools import product
import os
import glob
from argparse import ArgumentParser
import pickle
from random import sample

from shutil import copy2

"""
Returns (images, image_file_names) found for each person and all pairings generated
    Images have full path to enable copying for flattening
    Image File Names are also included
    Pairings have concatenated path: '{clothing_category}_{person_folder}_{image_filename}.jpg'
        Example: 'Shirts_Polos_id_0000598606_4_full.jpg'
"""
def get_images_and_pairings(person_folder, filename_so_far):
    person_img_files = os.listdir(person_folder)

    file_names = []
    images = []
    for file in person_img_files:
        image_path = os.path.join(person_folder, file)
        if os.path.isfile(image_path):
            file_names.append(filename_so_far + file)
            images.append(image_path)

    cart_prod = product(file_names, file_names)



    return (images, file_names), cart_prod


def get_all_pairs_and_images(top_dir, pairs_limit):
    all_pairs = []
    all_individual_images = []
    all_image_file_names = []

    person_subfolders_list = os.listdir(top_dir)
    for each_person_subfolder in person_subfolders_list:
        person_dir_path = os.path.join(top_dir, each_person_subfolder)
        filename_dir = f"{each_person_subfolder}"
        (images, image_file_names), cart_prod = get_images_and_pairings(person_dir_path, filename_dir)

        pairs_to_add = list(cart_prod)

        # print(images)
        # print(image_file_names)
        # print(pairs_to_add)
        assert len(images) == len(image_file_names), "Images and file names should be of equal length"
        if len(pairs_to_add) > pairs_limit:
            indices = sample(range(len(pairs_to_add)), pairs_limit)
        
            pairs_to_add = [pairs_to_add[i] for i in indices]
            # images = [images[i] for i in indices]
            # image_file_names = [image_file_names[i] for i in indices]

        all_individual_images.extend(images)
        all_image_file_names.extend(image_file_names)
        all_pairs.extend(pairs_to_add)


    # assert len(all_individual_images) == len(all_image_file_names) == len(pairs_to_add), "Lists for paths to images and the corresponding list for the image file names should be equal"
    print(f"Total image files found {len(all_individual_images)}")
    print(f"Total {len(all_pairs)} pairs generated")

    return all_pairs, (all_individual_images, all_image_file_names)


def create_flattened_dataset(all_images, all_images_filenames, out_dir):
    out_dir = os.path.join(out_dir, 'SourceImages')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    else:
        raise(FileExistsError(f"{out_dir} already exists, please delete or choose a different one and try again"))

    input_paths = all_images
    output_paths = [os.path.join(out_dir, f'{filename}') for filename in all_images_filenames]

    for i, (input_path, output_path) in enumerate(zip(input_paths, output_paths)):
        copy2(input_path, output_path)
     

    total_files_generated =  glob.glob(out_dir + '/*.jpg', recursive=True)
    assert len(total_files_generated) == len(input_paths), "All given input files are not generated"


#Pickle helper functions

def write_to_pickle(all_pairs, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(all_pairs, f)
    print(f"Wrote out {len(all_pairs)} pairs to {out_file}")


#Helper function for testing, not used in main
def read_from_pickle(pkl_file):
    with open(pkl_file, 'rb') as f:
        all_pairs = pickle.load(f)

    print(f"Read in {len(all_pairs)} pairs from {pkl_file}")

    return all_pairs

# Main Function
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_dataset_dir", required=True)
    parser.add_argument("--out_dataset_path", required=True, type=str)
    parser.add_argument("--out_pkl_file", required=True, type=str)
    parser.add_argument("--pairs_limit", type=int, default=8)
  

    args = parser.parse_args()

    all_pairs, (all_images, all_image_filenames) = get_all_pairs_and_images(args.raw_dataset_dir, args.pairs_limit)


    
    write_to_pickle(all_pairs, args.out_pkl_file)
    create_flattened_dataset(all_images, all_image_filenames, args.out_dataset_path)

    # all_pairs = read_from_pickle(args.out_pkl_file)

    # print(sample(all_pairs, 10))