import torch
import torch.nn as nn
from models.AttentionChannels import *
from models.AttentionGrid import *
from models.ConvBlock import *
from models.DenseBlock import *

class DenseUnet(nn.Module):

    INPUT_CHANNELS = 3
    # possible attention mechanisms
    GRID = "grid"
    CHANNEL = "channel"
    NONE = None

    
    def __init__(self, down_config=(8, 16, 32, 64), bottom=128, up_channels=(512, 256, 128, 64), growth=8, attention=None, nb_layers_up=2, activation_output=True):
        assert len(down_config) == 4
        assert len(up_channels) == 4
        assert attention in (self.GRID, self.CHANNEL, self.NONE), "Unknown attention mechanism"
        super(DenseUnet, self).__init__()

        down = down_config
        up = up_channels
        bn_channels = 4 * growth
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Descending layers
        self.dense1 = DenseBlock(self.INPUT_CHANNELS, growth_rate=growth, bn_channels=bn_channels, nb_layers=down[0])
        self.dense2 = DenseBlock(growth * down[0], growth_rate=growth, bn_channels=bn_channels, nb_layers=down[1])
        self.dense3 = DenseBlock(growth * down[1], growth_rate=growth, bn_channels=bn_channels, nb_layers=down[2])
        self.dense4 = DenseBlock(growth * down[2], growth_rate=growth, bn_channels=bn_channels, nb_layers=down[3])

        # Bottom layer
        self.bottom = DenseBlock(growth * down[3], growth_rate=growth, bn_channels=bn_channels, nb_layers=bottom)

        # Attention mechanism
        if attention == self.GRID:
            self.att4 = AttentionGrid(growth * down[3], up_channels[0])
            self.att3 = AttentionGrid(growth * down[2], up_channels[1])
            self.att2 = AttentionGrid(growth * down[1], up_channels[2])
            self.att1 = AttentionGrid(growth * down[0], up_channels[3])
        elif attention == self.CHANNEL:
            # Care: attention channels forward only takes one param -> uses lambda to have generic code in all cases, because grid attentino takes the two inputs
            self._att1 = AttentionChannel(growth * down[0])
            self._att2 = AttentionChannel(growth * down[1])
            self._att3 = AttentionChannel(growth * down[2])
            self._att4 = AttentionChannel(growth * down[3])
            # Accepts 2 params but only uses 1 -> Generic code afterwards
            self.att1 = lambda opp, down: self._att1(opp)
            self.att2 = lambda opp, down: self._att2(opp)
            self.att3 = lambda opp, down: self._att3(opp)
            self.att4 = lambda opp, down: self._att4(opp)
        else:
            # No attention mechanism
            identity = lambda opp, down: opp
            self.att1 = identity
            self.att2 = identity
            self.att3 = identity
            self.att4 = identity

        self.up4 = nn.ConvTranspose2d(growth * bottom, up_channels[0], kernel_size=4, stride=2, padding=1)
        self.uconv4 = ConvBlock(2 * up_channels[0], up_channels[0], nb_layers=nb_layers_up)

        self.up3 = nn.ConvTranspose2d(up_channels[0], up_channels[1], kernel_size=4, stride=2, padding=1)
        self.uconv3 = ConvBlock(2 * up_channels[1], up_channels[1], nb_layers=nb_layers_up)

        self.up2 = nn.ConvTranspose2d(up_channels[1], up_channels[2], kernel_size=4, stride=2, padding=1)
        self.uconv2 = ConvBlock(2 * up_channels[2], up_channels[2], nb_layers=nb_layers_up)

        self.up1 = nn.ConvTranspose2d(up_channels[2], up_channels[3], kernel_size=4, stride=2, padding=1)
        self.uconv1 = ConvBlock(2* up_channels[3], up_channels[3], nb_layers=nb_layers_up)
            
        # Combine channels pixel-wise to determine segmentation
        self.out_layer = nn.Conv2d(up_channels[3], 1, kernel_size=1)
        self.out_norm = nn.BatchNorm2d(1)
        self.sigm = nn.Sigmoid()
        self.activation_output = activation_output


    def forward(self, x):
        out1 = self.dense1(x)
        x2 = self.pool(out1)

        out2 = self.dense2(x2)
        x3 = self.pool(out2)

        out3 = self.dense3(x3)
        x4 = self.pool(out3)

        out4 = self.dense4(x4)
        x5 = self.pool(out4)

        x_bot = self.bottom(x5)

        x_up4 = self.up4(x_bot)
        skipped4 = self.att4(out4, x_up4)
        # Concat input channels from below and weighted skip connection
        x_up4 = torch.cat([x_up4, skipped4], dim=1)
        # Standard 3x3 conv block
        x_up4 = self.uconv4(x_up4)
        # Pass to higher layer

        x_up3 = self.up3(x_up4)
        skipped3 = self.att3(out3, x_up3)
        # Concat input channels from below and weighted skip connection
        x_up3 = torch.cat([x_up3, skipped3], dim=1)
        # Standard 3x3 conv block
        x_up3 = self.uconv3(x_up3)
        # Pass to higher layer

        x_up2 = self.up2(x_up3)
        skipped2 = self.att2(out2, x_up2)
        # Concat input channels from below and weighted skip connection
        x_up2 = torch.cat([x_up2, skipped2], dim=1)
        # Standard 3x3 conv block
        x_up2 = self.uconv2(x_up2)
        # Pass to higher layer

        x_up1 = self.up1(x_up2)
        skipped1 = self.att1(out1, x_up1)
        # Concat input channels from below and weighted skip connection
        x_up1 = torch.cat([x_up1, skipped1], dim=1)
        # Standard 3x3 conv block
        x_up1 = self.uconv1(x_up1)
        # Pass to higher layer


        # Final output layer
        out = self.out_layer(x_up1)
        out = self.out_norm(out)
        
        if not self.training or self.activation_output:
            return self.sigm(out)
        else:
            return out
