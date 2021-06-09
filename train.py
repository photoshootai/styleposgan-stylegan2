from argparse import ArgumentParser
from itertools import accumulate

from numpy.core.numeric import True_
import torch
import torch.nn as nn

import numpy as np

from torchvision import datasets, models, transforms     # vision datasets,
import torchvision.transforms as transforms              # composable transforms
from torch.utils.data import DataLoader


#Pytorch Lightning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import DeepFashionDataModule

from models.StylePoseGAN import StylePoseGAN

import gc

#WandB
import wandb
from utils import WandbImageCallback

def main(args):
 
    gc.collect()
    torch.cuda.empty_cache()
    # torch.autograd.set_detect_anomaly(True)

    wandb.login()
    logger = WandbLogger(project='styleposegan', log_model='all')

    #For reporducibility: deterministic = True and next line
    seed_everything(42, workers=True)

    #Checkpoint Setup
    checkpoint_callback = ModelCheckpoint(

        dirpath= args.checkpoints_path, #Save to custom path
        filename='gan-img-{epoch}-g-{gen_loss:.3f}-d-{disc_loss: .3f}', #Saved filename format
        every_n_val_epochs=1 #Save every 10 epochs
    )

    #
    
    datamodule = DeepFashionDataModule(args.source_image_path, args.pose_map_path, args.texture_map_path, batch_size=args.batch_size, image_size=(args.image_size, args.image_size), num_workers=args.num_workers)

    #Setting up data samples for logging genarated images 
    # datamodule.prepare_data()
    # datamodule.setup()

    # # val_samples = next(iter(datamodule.val_dataloader()))


    # # generated_imgs_callback = WandbImageCallback(val_samples)

    model = StylePoseGAN(args.image_size, batch_size=args.batch_size)
   

    #Init Trainer
    trainer = Trainer(tpu_cores=args.tpu_cores, gpus=args.gpus, precision=args.precision, logger=logger, profiler="advanced",
                      progress_bar_refresh_rate=20, accelerator=args.accelerator, num_nodes=args.num_nodes, resume_from_checkpoint=args.resume_from_checkpoint,
                      callbacks=[checkpoint_callback], fast_dev_run=args.fast_dev_run) 
    
    #Log Gradients and model topology
    logger.watch(model)

    #trainer.fit(model, train_loader)
    trainer.fit(model, datamodule)


    print("***Finished Training***")

    
if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    parser = ArgumentParser()

    #Trainer Args
    parser.add_argument('--source_image_path', type=str, default="./data/TrainingData/SourceImages")
    parser.add_argument('--pose_map_path', type=str, default="./data/TrainingData/PoseMaps")
    parser.add_argument('--texture_map_path', type=str, default="./data/TrainingData/TextureMaps")

    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--tpu_cores', type=int, default=None)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--top_k_training', type=bool, default=False)
    parser.add_argument('--deterministic', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default=None) #Use with GPUs not with TPUs
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--checkpoints_path', type=str, default="./checkpoints/")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--fast_dev_run', type=bool, default=False)


    
    args = parser.parse_args()

    main(args)
