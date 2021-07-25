import os

def delete_men_or_women(sex_to_delete, source_data_path, texture_data_path, pose_map_data_path):

    deleted = 0
    non_jpg_count =0

    assert sex_to_delete in {'MEN', "WOMEN"}, "Sex to delete should be either 'MEN' or 'WOMEN'"

    for img_filename in os.listdir(source_data_path):
        if img_filename.endswith("jpg"):
            sex = img_filename.split('.')[0].split('_')[0]
            if sex == sex_to_delete:
                os.remove(texture_data_path + "/" +  img_filename)
                os.remove(pose_map_data_path + "/" + img_filename)
                os.remove(source_data_path + "/" +  img_filename)
                deleted +=1
                
        else:
            non_jpg_count+=1

    print(f"Deleted: {deleted}")
    print(f"Non JPGs: {non_jpg_count}")

def delete_pose(source_data_path, texture_data_path, pose_map_data_path):
    deleted = 0


    pose_to_delete = {"side", "back"}


    for img_filename in os.listdir(source_data_path):
        if img_filename.endswith("jpg"):
            pose = img_filename.split('.')[0].split('_')[-1]
            print(pose)
            if pose in pose_to_delete:
                os.remove(texture_data_path + "/" +  img_filename)
                os.remove(pose_map_data_path + "/" + img_filename)
                os.remove(source_data_path + "/" +  img_filename)
                deleted +=1
                
        # else:
    #     #     non_jpg_count+=1

    print(f"Deleted: {deleted}")
    # print(f"Non JPGs: {non_jpg_count}")
    
def main():
    source_images_data_path = "./data/DeepFashionMenOnly/SourceImages"
    texture_map_data_path = "./data/DeepFashionMenOnly/PoseMaps"
    pose_map_data_path = "./data/DeepFashionMenOnly/TextureMaps"
    # sex_to_delete = "MEN"
    # delete_men_or_women(sex_to_delete, source_images_data_path, texture_map_data_path, pose_map_data_path)

    delete_pose(source_images_data_path, texture_map_data_path, pose_map_data_path)

if __name__ == "__main__":
    main()