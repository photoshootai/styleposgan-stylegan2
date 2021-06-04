from argparse import ArgumentParser
from itertools import accumulate

from numpy.core.numeric import True_
import torch
import torch.nn as nn


import numpy as np
# from stylegan import Generator, Discriminator, StyleGAN2
# sys.path.append("./stylegan2-ada-pytorch")

from torchvision import datasets, models, transforms     # vision datasets,
# architectures &
# transforms
import torchvision.transforms as transforms              # composable transforms
from torch.utils.data import DataLoader


#Pytorch Lightning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.loggers import WandbLogger

from datasets import DeepFashionDataModule

from models.StylePoseGAN import StylePoseGAN

import gc

# # RayTune
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler


def main(args):
 
    gc.collect()
    torch.cuda.empty_cache()
    # torch.autograd.set_detect_anomaly(True)

    #Logging
    logger = TensorBoardLogger('tb_logs', name='my_model')

    #For reporducibility: deterministic = True and next line
    seed_everything(42, workers=True)
    
    #Init Train
    trainer = Trainer(tpu_cores=args.tpu_cores, gpus=args.gpus, precision=args.precision, logger=logger, profiler="advanced",
                      progress_bar_refresh_rate=20, accelerator=args.accelerator, num_nodes=args.num_nodes)

    
    datamodule = DeepFashionDataModule(args.source_image_path, args.pose_map_path, args.texture_map_path, batch_size=args.batch_size, image_size=(args.image_size, args.image_size), num_workers=args.num_workers)
    model = StylePoseGAN(args.image_size, batch_size=args.batch_size)
    
    #trainer.fit(model, train_loader)
    trainer.fit(model, datamodule)


    print("***Finished Training***")

    #Test set evaluation
    #trainer.test(test_dataloaders=test_dataloaders)
    
if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    parser = ArgumentParser()

    #Program args
    

    #Model Specific Args: hparams specific to models
    #parser = StylePoseGAN.add_model_specific_args(parser)

    #Trainer Args
    parser.add_argument('--source_image_path', type=str, default="./data/TrainingData/SourceImages")
    parser.add_argument('--pose_map_path', type=str, default="./data/TrainingData/PoseMaps")
    parser.add_argument('--texture_map_path', type=str, default="./data/TrainingData/TextureMaps")

    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--tpu_cores', type=int, default=None)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--accumulate_grad_batches', type=int, default=32)
    parser.add_argument('--top_k_training', type=bool, default=False)
    parser.add_argument('--deterministic', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default=None) #Use with GPUs not with TPUs
    parser.add_argument('--num_nodes', type=int, default=1)

    args = parser.parse_args()

    main(args)
