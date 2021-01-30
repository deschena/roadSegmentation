import torch
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

class DenseLayer(nn.Module):
    """Dense layer. Composed of 1x1 conv to reduce input dimensions, passed to 3x3 padded convolution
    to extract useful structure.
    """
    def __init__(self, in_channels, bn_channels, growth_rate):
        """Create Denselayer

        Args:
            in_channels (int): Number of input channels
            bn_channels (int): Reducing factor of bottleneck. Nb of channels after bottleneck is growth * bn_channels
            growth_rate (int): Number of channels outputed by the layer
        """
        super(DenseLayer, self).__init__()
        # Operations on input, change number of features in the layer
        
        # No padding since we do 1x1 convolution
        self.bn_conv = bncrl(in_channels, int(bn_channels * growth_rate), kernel_size=1, padding=0)
        self.out_conv = bncrl(int(bn_channels * growth_rate), growth_rate, kernel_size=3, padding=1)
        
    def forward(self, prev_features):
        # Combine input features & pass to bottleneck
        all_features = torch.cat(prev_features, 1)
        
        out = self.bn_conv(all_features)
        out = self.out_conv(out)
        return out