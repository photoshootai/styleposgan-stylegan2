import torch


# USE L1 LOSS ON OUTPUTS
class DPatch(torch.nn.Module):
    def __init__(self, input_channels=3, n_final_layer_filters=64, hidden_layers=3):
        """
        PatchGAN based on the Pix2PixHD architecture

        Arguments:
            input_channels [opt int=3]: The number of input channels in the image
            n_final_layer_filters [opt int=64]: The number of filters in last conv layer
            hidden_layers [opt int=3]: The number of hidden layers in the discrimator
        """
        super(DPatch, self).__init__()
        self.hidden_layers = hidden_layers
        
        ReLU_slope = 0.2
        kernel_size, stride = 4, 2
        pad_size = int(np.ceil((kernel_size - 1.0) / 2))
        sequence = [
            torch.nn.Conv2d(input_channels, n_final_layer_filters, kernel_size=kernel_size, stride=stride, padding=pad_size),
            torch.nn.LeakyReLU(ReLU_slope, True)
        ]

        n_filters = n_final_layer_filters
        for n in range(1, hidden_layers):
            n_filters_prev = n_filters
            n_filters = min(n_filters * 2, 512)
            sequence += [
                torch.nn.Conv2d(n_filters_prev, n_filters, kernel_size=kernel_size, stride=stride, padding=pad_size),
                torch.nn.BatchNorm2d(n_filters),
                torch.nn.LeakyReLU(ReLU_slope, True)
            ]

        n_filters_prev = n_filters
        n_filters = min(n_filters * 2, 512)
        sequence += [
            torch.nn.Conv2d(n_filters_prev, n_filters, kernel_size=kernel_size, stride=1, padding=pad_size),
            torch.nn.BatchNorm2d(n_filters),
            torch.nn.LeakyReLU(ReLU_slope, True)
        ]

        sequence += [torch.nn.Conv2d(n_filters, 1, kernel_size=kernel_size, stride=1, padding=pad_size)]

        self.model = torch.nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)