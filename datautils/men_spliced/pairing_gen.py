from itertools import product
import os
from argparse import ArgumentParser
import pickle

def get_cartesian_product(person_folder):
    person_img_files = os.listdir(person_folder)
    person_img_files = [os.path.join(person_folder, file) for file in person_img_files]
    # print(len(person_img_files))
    cart_prod = product(person_img_files, person_img_files)
    return cart_prod


def get_all_pairs(top_dir):
    clothing_cat_dirs = os.listdir(top_dir)
    clothing_cat_dirs = [os.path.join(top_dir, dir) for dir in clothing_cat_dirs]

    all_pairs = []
    for cat_dir in clothing_cat_dirs:
        person_subfolders_list = os.listdir(cat_dir)
        for each_person_subfolder in person_subfolders_list:
            person_dir_path = os.path.join(cat_dir, each_person_subfolder)
            # print(person_dir_path)
            pairs = list(get_cartesian_product(person_dir_path))
            # print(len(pairs))
            all_pairs.extend(pairs)
 
        
    
    # print(len(all_pairs))
    return all_pairs

def write_to_pickle(all_pairs, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(all_pairs, f)
    print(f"Wrote out {len(all_pairs)} pairs to {out_file}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--top_dir", required=True)
    parser.add_argument("--out_file", required=True)

    args = parser.parse_args()

    all_pairs = get_all_pairs(args.top_dir)
    write_to_pickle(all_pairs, args.out_file)