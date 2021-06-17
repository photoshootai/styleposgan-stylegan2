from pytorch_lightning.core import datamodule
import torch
from torchvision import transforms
import PIL

from datasets import DeepFashionDataModule

def main():
	source_image_path="./data/TrainingData/SourceImages"
	pose_map_path="./data/TrainingData/PoseMaps"
	texture_map_path="./data/TrainingData/TextureMaps"
	batch_size = 8
	image_size = 256

	datamodule = DeepFashionDataModule(source_image_path, pose_map_path, texture_map_path, batch_size=batch_size, image_size=(image_size, image_size), num_workers=2)
	datamodule.prepare_data()
	datamodule.setup(stage='fit')
	batch = next(iter(datamodule.train_dataloader()))
	(I_s, S_pose_map, S_texture_map), (I_t, T_pose_map) = batch #x, y = batch, so x is  the tuple, and y is the triplet
	print(I_s.size())
	im = transforms.ToPILImage()(I_s[2]).convert("RGB")
	im.show()

if __name__ == "__main__":
	main()