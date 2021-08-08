import torch
import torch.nn as nn
import argparse
import traceback

from stylegan2 import *

def load_config(config_path):
    """
    Loads config from json file.
    """
    if config_path is None:
        return None

    print("Loading config from {}".format(config_path))
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

def init_gan(config):
    """
    Initializes a GAN from a config.
    """


    image_size = config['image_size']
    network_capacity = config['network_capacity']
    transparent = config['transparent']
    fq_layers = config['fq_layers']
    fq_dict_size = config['fq_dict_size']
    fmap_max = config.pop('fmap_max', 512)
    attn_layers = config.pop('attn_layers', [])
    no_const = config.pop('no_const', False)
    lr_mlp = config.pop('lr_mlp', 0.1)
    
    gan = StyleGAN2(lr_mlp=lr_mlp, image_size=image_size, network_capacity=network_capacity, fmap_max=fmap_max,
                             transparent=transparent, fq_layers=fq_layers, fq_dict_size=fq_dict_size, attn_layers=attn_layers)

    return gan


def load_pretrained_gan(initialized_GAN, checkpoint_path, model_no):
    """
    Loads pretrained GAN model from checkpoint.
    """
    model_name = 'model_' + str(model_no)
    model_path = os.path.join(checkpoint_path, model_name)
    assert os.path.isfile(model_path), "Model not found at {}".format(model_path)

    print("Loading pretrained GAN from {}".format(model_path))
    load_data = torch.load(model_path)


    try:
        initialized_GAN.load_state_dict(load_data['GAN'])

    except Exception as e:
        print("Error: Could not load model state.")
        # traceback.print_exc()
        raise e
        
    return initialized_GAN


def main(ckpt_dir, model_no, out_dir):
    
    config_path = os.path.join(ckpt_dir, '.config.json')
    config = load_config(config_path)


    initialized_gan = init_gan(config)
    loaded_pretrained_gan = load_pretrained_gan(initialized_gan, ckpt_dir, model_no)

    print(load_pretrained_gan)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StyleGAN')
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--model_no', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)

    main()


