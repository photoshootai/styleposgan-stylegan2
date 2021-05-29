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

# # RayTune
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler


def main(args):
 
   #train_loader = DataLoader(dataset, batch_size=batch_size)
    torch.cuda.empty_cache()
    #Logging
    logger = TensorBoardLogger('tb_logs', name='my_model')

    #For reporducibility: deterministic = True and 
    seed_everything(42, workers=True)
    trainer = Trainer(gpus=args.gpus,logger=logger, profiler="simple")

    source_image_path="./data/TrainingData/SourceImages"
    pose_map_path="./data/TrainingData/PoseMaps"
    texture_map_path="./data/TrainingData/TextureMaps"
    
    datamodule = DeepFashionDataModule(source_image_path, pose_map_path, texture_map_path, batch_size=args.batch_size, image_size=(args.image_size, args.image_size))
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
    parser.add_argument('--data_path', type=str, default="./data/deepfashion")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--tpu_cores', type=int, default=None)
    parser.add_argument('--precision', type=int, default=32 )
    parser.add_argument('--accumulate_grad_batches', type=int, default=32 )
    parser.add_argument('--top_k_training', type=bool, default=False)
    parser.add_argument('--deterministic', type=bool, default=True)

    


    args = parser.parse_args()

    main(args)


# def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
#     train()
    # data_dir = os.path.abspath("./data")
    # load_data(data_dir)
    # config = {
    #     "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    #     "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    #     "lr": tune.loguniform(1e-4, 1e-1),
    #     "batch_size": tune.choice([2, 4, 8, 16])
    # }
    # scheduler = ASHAScheduler(
    #     metric="loss",
    #     mode="min",
    #     max_t=max_num_epochs,
    #     grace_period=1,
    #     reduction_factor=2)
    # reporter = CLIReporter(
    #     # parameter_columns=["l1", "l2", "lr", "batch_size"],
    #     metric_columns=["loss", "accuracy", "training_iteration"])
    # result = tune.run(
    #     partial(train_cifar, data_dir=data_dir),
    #     resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
    #     config=config,
    #     num_samples=num_samples,
    #     scheduler=scheduler,
    #     progress_reporter=reporter)

    # best_trial = result.get_best_trial("loss", "min", "last")
    # print("Best trial config: {}".format(best_trial.config))
    # print("Best trial final validation loss: {}".format(
    #     best_trial.last_result["loss"]))
    # print("Best trial final validation accuracy: {}".format(
    #     best_trial.last_result["accuracy"]))

    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(
    #             best_trained_model)  # this or DistributedDataParallel
    # best_trained_model.to(device)

    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(
    #     best_checkpoint_dir, "checkpoint"))
    # best_trained_model.load_state_dict(model_state)

    # test_acc = test_accuracy(best_trained_model, device)
    #print("Best trial test set accuracy: {}".format(test_acc))

