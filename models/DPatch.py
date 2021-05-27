import torch

import numpy as np

# USE L1 LOSS ON OUTPUTS
class DPatch(torch.nn.Module):
    def __init__(self, input_channels, n_final_layer_filters=64, hidden_layers=3):
        """
        Parameters:
            input_channels (int)  -- the number of channels in input images
            n_final_layer_filters (int)       -- the number of filters in the last conv layer
            hidden_layers (int)  -- the number of conv layers in the discriminator
        """
        super(DPatch, self).__init__()
        self.hidden_layers = hidden_layers
        

        kernel_size, stride = 4, 2
        pad_size = int(np.ceil((kernel_size - 1.0) / 2))
        sequence = [
            torch.nn.Conv2d(input_channels, n_final_layer_filters, kernel_size=kernel_size, stride=stride, padding=pad_size),
            torch.nn.LeakyReLU(0.2, True)
        ]

        nf = n_final_layer_filters
        for n in range(1, hidden_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                torch.nn.Conv2d(nf_prev, nf, kernel_size=kernel_size, stride=stride, padding=pad_size),
                torch.nn.BatchNorm2d(nf),
                torch.nn.LeakyReLU(0.2, True)
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            torch.nn.Conv2d(nf_prev, nf, kernel_size=kernel_size, stride=1, padding=pad_size),
            torch.nn.BatchNorm2d(nf),
            torch.nn.LeakyReLU(0.2, True)
        ]

        sequence += [torch.nn.Conv2d(nf, 1, kernel_size=kernel_size, stride=1, padding=pad_size)]

        self.model = torch.nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)