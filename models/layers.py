import torch.nn as nn


"""
Double checked by Madhav with http://d2l.ai/chapter_convolutional-modern/resnet.html from the D2L.ai book.
We don't have the additional 1x1 conv layer used to change the number of channels. It helps us "transform the input into the 
desired shape for the addition operation".
"""

class ResidualBlock(nn.Module):
    #this class defines a set of functions that comprise the residual block as specified by StylePoseGAN
    #notes:
    #1. the change in channels occurs in the first convolutional layer
    #2. together, the kernel size of 3 and padding of 1 do not change the size of the tensor
    #3. the stride of two means that the image dimension (H*W) will be halved by the first convolution
    #4. in the forward pass skip connection, a single convolution must be done to the original input so that it is the same size as the output

    def __init__(self, in_chan, out_chan, kernel_size=(3,3), stride=2, padding=1, bias=False):
        super(ResidualBlock, self).__init__()
        self.res_conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.res_bn1 = nn.BatchNorm2d(out_chan, track_running_stats=False)
        self.res_relu1 = nn.ReLU(inplace=True)
        self.res_conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
        self.res_bn2 = nn.BatchNorm2d(out_chan, track_running_stats=False)
        self.skip_conv = nn.Conv2d(in_chan, out_chan, kernel_size=(1,1), stride=stride, padding=0, bias=bias)# this conv will make the sizes of input and output match as specified by He et al.
        self.res_relu2 = nn.ReLU(inplace=True)


    def forward(self, x):
      '''
      Function for completing a forward pass through a ResidualBlock object
      input:
      x: an (N*C_in*H*W) tensor defining a batch of images
      output:
      E: an (N*C_out*H/16*W/16) tensor 
        where C_in and C_out are defined in the initialization method
      '''
      identity = x

      #execute the main path (f(x))
      x = self.res_conv1(x)
      #print(x.size())
      x = self.res_bn1(x)
      x = self.res_relu1(x)
      x = self.res_conv2(x)
      x = self.res_bn2(x)
    
      #execute the identity path
      identity = self.skip_conv(identity)
      
      #they should now be the same size
      output = x + identity

      #apply the last activation function before returning
      output = self.res_relu2(output)

      return output