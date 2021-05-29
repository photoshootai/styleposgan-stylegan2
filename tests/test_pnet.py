import torch.nn as nn
from os import listdir
from PIL import Image as PImage

from models import PNet

def loadImages(path):
    # return array of images

    imagesList = listdir("E:\Test Images\PoseMaps")
    loadedImages = []
    for image in imagesList:
        img = PImage.open(path + image)
        loadedImages.append(img)

    return loadedImages

path = "E:\Test Images\PoseMaps"

# your images in an array
imgs = loadImages(path)


class test_pnet():

    def pnet_test(input):
       pnet = PNet(im_chan=3)
       output = pnet(input)
       return output.size()

    
    assert pnet_test(torch.rand(8, 3, 512, 512)) == (8, 512, 16, 16)
