import torch
import torch.nn as nn
from models.DenseLayer import *

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_channels, nb_layers):
        """Create a block composed of sequential sublayers

        Args:
            in_channels (int): Number of input channels
            growth_rate (int): Number of output channels
            bn_channels (int): Factor of number of channels in layer's bottleneck. See DenseLayer for more details
            nb_layers (int): Number of Layers in the block
        """
        super(DenseBlock, self).__init__()
        # Save attributes
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.nb_layers = nb_layers        
        # Create layers
        self.layers = [DenseLayer(in_channels + i * growth_rate,
                                  bn_channels,
                                  growth_rate
                                 )  for i in range(nb_layers)]
        # Register layers in the module
        for i, l in enumerate(self.layers):
            self.add_module(f"dense layer {i}", l)
            
    def forward(self, in_features):
        out = [in_features]
        for l in self.layers:
            # forward all previous features into the next layer
            l_features = l(out)
            # save new features for next stages
            out.append(l_features)
        # Concatenate results of layers, except input
        return torch.cat(out[1:], 1)
    
    @property
    def out_channels(self):
        return self.growth_rate * self.nb_layers 