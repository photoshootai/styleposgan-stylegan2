import os

def get_men_women_dist(data_path):

    count_men = 0
    count_fem = 0
    count_other = 0

    for img_filename in os.listdir(data_path):
        if img_filename.endswith("jpg"):
            sex = img_filename.split('.')[0].split('_')[0]
            if sex == 'MEN':
                count_men +=1
            elif sex == 'WOMEN':
                count_fem +=1
            else:
                count_other +=1

    print(f"Count Men: {count_men}")
    print(f"Count Women: {count_fem}")
    print(f"Count Other: {count_other}")

def check_pose_and_texture_maps_exist_for_source_images(source_data_path, texture_data_path, pose_map_data_path):
    missing_files = {}

    text_maps = [each.split('.')[0] for each in os.listdir(texture_data_path)]
    pose_maps = [each.split('.')[0] for each in os.listdir(pose_map_data_path)]

    for img_filename in os.listdir(source_data_path):
        file_name = img_filename.split('.')[0] if img_filename.endswith("jpg") else None
        if file_name is not None:
            if file_name not in text_maps:
                if file_name in missing_files:
                    missing_files[file_name].append("missing_texture")
                else:
                    missing_files[file_name] = ["missing_texture"]
            if file_name not in pose_maps:
                if file_name in missing_files:
                    missing_files[file_name].append("missing_pose")
                else:
                    missing_files[file_name] = ["missing_pose"]
               
    print(f"Count Missing Files: {len(missing_files)}")


    

def main():
    source_images_data_path = "./data/DeepFashionWomenOnly/SourceImages"
    texture_map_data_path = "./data/DeepFashionWomenOnly/PoseMaps"
    pose_map_data_path = "./data/DeepFashionWomenOnly/TextureMaps"
    get_men_women_dist(source_images_data_path)
    check_pose_and_texture_maps_exist_for_source_images(source_images_data_path, texture_map_data_path, pose_map_data_path)
    return 



if __name__ == "__main__":
    main()