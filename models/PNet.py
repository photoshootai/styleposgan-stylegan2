import torch.nn as nn
from .layers import ResidualBlock

class PNet(nn.Module):
  #this class defines the PNet 
  #Notes:
  #1. the channels input will always be the same, so we can hard code the channels
  #2. the input image dimensions (H*W) will not always be the same, so those are controlled by stride in the residual block class
    def __init__(self, im_chan=3):
        super(PNet, self).__init__()
        self.res_block1 = ResidualBlock(in_chan=im_chan,out_chan=64)
        self.res_block2 = ResidualBlock(in_chan=64,out_chan=128)
        self.res_block3 = ResidualBlock(in_chan=128,out_chan=256)
        self.res_block4 = ResidualBlock(in_chan=256,out_chan=512)
        
    def forward(self, x):
        '''
        Function for completing a forward pass of the PNet: Given a dense pose map, 
        it runs them through four residual blocks and returns a 512 element encoding of the pose.
        Parameters:
            image: an H*W*3 iamge depicting a dense pose map
        Outputs:
            E: a 1D tensor encoding the pose of (H/16)*(W/16)*512
        '''
        input = x
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        E = x
        
        return E
