from pytorch_lightning.core import datamodule
import torch
from torchvision import transforms
import PIL

from datasets import DeepFashionDataset
from torch.utils.data import DataLoader

def main():
	data_path = "./data/TrainingData"
	batch_size = 4
	num_workers = 1
	image_size = (512, 512)

	dataset = DeepFashionDataset(data_path, image_size=image_size)
	dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
	
	batch = next(iter(dataloader))
	(I_s, P_s, A_s), (I_t, P_t) = batch #x, y = batch, so x is  the tuple, and y is the triplet
	for i in range(batch_size):
		print(I_s[i].size())
		im = transforms.ToPILImage()(I_s[i]).convert("RGB")
		im.show()

		print(P_s[i].size())
		im = transforms.ToPILImage()(P_s[i]).convert("RGB")
		im.show()

		print(A_s[i].size())
		im = transforms.ToPILImage()(A_s[i]).convert("RGB")
		im.show()

if __name__ == "__main__":
	main()