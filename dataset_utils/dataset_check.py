import torch
import torchvision
import torch.nn.functional as F
import PIL
import os

from datasets import DeepFashionDataset
from torch.utils.data import DataLoader


def show_batch(batch, image_size=(256, 256)):
    (I_s, P_s, A_s), (I_t, P_t) = batch  # Get next iter
    size = I_s.shape[0]

    I_s = F.interpolate(I_s, size=image_size) # Resize all to same size
    P_s = F.interpolate(P_s, size=image_size)
    A_s = F.interpolate(A_s, size=image_size)
    I_t = F.interpolate(I_t, size=image_size)
    P_t = F.interpolate(P_t, size=image_size)

    generated_stack = torch.cat(
        (I_s, I_t, P_s, A_s, P_t), dim=0)
    
    save_path = str(f"./results/debugging.jpg") # Stack and save
    torchvision.utils.save_image(generated_stack, save_path, nrow=size)

    input("Enter to continue ...")


def show_batch_from_loader(dataloader, image_size=(256, 256)):
    if not os.path.isdir('results'):
        os.mkdir('results')
    dl = iter(dataloader)
    for i in range(10):
        (I_s, P_s, A_s), (I_t, P_t) = next(dl)  # Get next iter
        size = I_s.shape[0]

        I_s = F.interpolate(I_s, size=image_size) # Resize all to same size
        P_s = F.interpolate(P_s, size=image_size)
        A_s = F.interpolate(A_s, size=image_size)
        I_t = F.interpolate(I_t, size=image_size)
        P_t = F.interpolate(P_t, size=image_size)

        generated_stack = torch.cat(
            (I_s, I_t, P_s, A_s, P_t), dim=0)
        
        save_path = str(f"./results/debugging_{i}.jpg") # Stack and save
        torchvision.utils.save_image(generated_stack, save_path, nrow=size)

    # input("Press Enter to continue...")


def main():
    data_path = "/content/styleposegan/DeepFashionWithFace"
    batch_size = 48
    num_workers = 1
    image_size = (128, 128)

    dataset = DeepFashionDataset(data_path, image_size=image_size, props={'model', 'clothing_id'}) #Change seed to get different images
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers)

    # batch = next(iter(dataloader))
    # x, y = batch, so x is  the tuple, and y is the triplet
    show_batch(dataloader, image_size=image_size)


if __name__ == "__main__":
    main()
