import torch
import torchvision

import PIL

from datasets import DeepFashionDataset
from torch.utils.data import DataLoader

def show_batch(batch):
    (I_s, P_s, A_s), (I_t, P_t) = batch
    size = I_s.shape[0]
    generated_stack = torch.cat(
        (I_s, I_t, P_s, A_s, P_t), dim=0)
    
    save_path = str("./results/debugging.jpg")
    torchvision.utils.save_image(generated_stack, save_path, nrow=size)


    # for i in range(I_s.shape[0]):
    #     print(I_s[i].size())
    #     im = transforms.ToPILImage()(I_s[i]).convert("RGB")
    #     im.show("Source Image")

    #     print(P_s[i].size())
    #     im = transforms.ToPILImage()(P_s[i]).convert("RGB")
    #     # im.show("Source Pose Map")

    #     print(A_s[i].size())
    #     im = transforms.ToPILImage()(A_s[i]).convert("RGB")
    #     # im.show("Source Texture Map")

    #     print(I_t[i].size())
    #     im = transforms.ToPILImage()(I_t[i]).convert("RGB")
    #     im.show("Target Image")

    #     print(P_t[i].size())
    #     im = transforms.ToPILImage()(P_t[i]).convert("RGB")
    #     # im.show("Target Pose Map")

    input("Press Enter to continue...")


def main():
    data_path = "./data/DeepFashionWithFace"
    batch_size = 4
    num_workers = 1
    image_size = (256, 256)

    dataset = DeepFashionDataset(data_path, image_size=image_size, seed=90, props={'model', 'clothing_id'}) #Change seed to get different images
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers)

    batch = next(iter(dataloader))
    # x, y = batch, so x is  the tuple, and y is the triplet
    show_batch(batch)


if __name__ == "__main__":
    main()
