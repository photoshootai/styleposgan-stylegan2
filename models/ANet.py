import torch.nn as nn
from .layers import ResidualBlock

def get_anet_final_block(in_chan=512, out_chan=2048, hidden_chan=1024, kernel_size=(3,3), stride=4, padding=1, bias=False):

    return nn.Sequential(
        #the normalization was recommended by kshitij. It wasn't explicitly prescribed in the text
        nn.Conv2d(in_chan, in_chan, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        # nn.Conv2d(hidden_chan,hidden_chan,kernel_size,stride,padding),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(hidden_chan,hidden_chan,kernel_size,stride,padding),
        # nn.ReLU(inplace=True),
        nn.Conv2d(in_chan,in_chan,kernel_size,stride,padding),
        nn.ReLU(inplace=True),
        nn.Flatten()
        # nn.Conv2d(hidden_chan,hidden_chan,kernel_size,2,padding),
        # nn.ReLU(inplace=True)
    )

class ANet(nn.Module):
    def __init__(self, im_chan=3):
        super(ANet, self).__init__()
        
        self.res_block1 = ResidualBlock(in_chan=im_chan,out_chan=64)
        self.res_block2 = ResidualBlock(in_chan=64,out_chan=128)
        self.res_block3 = ResidualBlock(in_chan=128,out_chan=256)
        self.res_block4 = ResidualBlock(in_chan=256,out_chan=512)
        self.last_block = get_anet_final_block() #just the Conv2d parts
        self.linear = nn.Linear(2048,2048)
    def forward(self, x):
        '''
        Function for completing a forward pass of the ANet: Given a texture map tensor, 
        returns a vector of 2048 elements encoding of the appearance.
        Parameters:
            image: an H*W*3 iamge depicting a UV texture map
        Outputs:
            z: a 1D tensor encoding the appearance
        '''
        #pass the input image x through the res_blocks (same architecture as PNet)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.last_block(x)

        #to make it compatible with the fully connected layer, move channels to the final index
        # x = x.permute(0,2,3,1)
        z = self.linear(x)
        z = z.squeeze(dim=1) # remove an empty axis, another axis could optionally be removed
        
        return z
