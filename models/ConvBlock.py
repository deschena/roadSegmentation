import torch.nn as nn

def bncrl(in_channels, out_channels, kernel_size=1, padding=1, stride=1):
    """
    Almost all convolution operations are of the form conv + bn + relu.
    Makes sense to join them to avoid boilerplate
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  padding=padding, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class ConvBlock(nn.Module):
    """
    Sequence of 3x3 convolutions used by the different flavours of Unet
    """
    def __init__(self, in_channels, out_channels, nb_layers=2):
        """Sequence of 3x3 convolutions used by the different flavours of Unet

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            nb_layers (int, optional): Number of conv layers. Defaults to 2.
        """

        super(ConvBlock, self).__init__()
        assert nb_layers >= 1, "Conv. block must have at least one layer"
        # Create all conv layers
        
        conv_in = bncrl(in_channels, out_channels, kernel_size=3)
        self.add_module("input conv+bn+relu", conv_in)
        all_convs = [conv_in]
        # Create intermediate layers
        for _ in range(nb_layers - 1):
            new_layer = bncrl(out_channels, out_channels, kernel_size=3)
            all_convs.append(new_layer)

        # Concat all layers into a seq module
        self.f = nn.Sequential(*all_convs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        
    def forward(self, x):
        return self.f(x)