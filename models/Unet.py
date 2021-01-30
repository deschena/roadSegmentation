import torch
from torch._C import Value
import torch.nn as nn
from models.AttentionChannels import *
from models.AttentionGrid import *
from models.ConvBlock import *

class Unet(nn.Module):

    INPUT_CHANNELS = 3
    # possible attention mechanisms
    GRID = "grid"
    CHANNEL = "channel"
    NONE = None

    def __init__(self, layout=(64, 128, 256, 512, 1024), attention=None, nb_layers=2, activation_output=True):
        super(Unet, self).__init__()
        assert len(layout) == 5, "Need to describe the 5 stages of the unet"
        assert attention in (self.GRID, self.CHANNEL, self.NONE), "Unknown attention mechanism"
        self.attention = attention
        lv1, lv2, lv3, lv4, lv5 = layout

        # Attention mechanisms
        if attention == self.GRID:
            self.att4 = AttentionGrid(lv4, lv4)
            self.att3 = AttentionGrid(lv3, lv3)
            self.att2 = AttentionGrid(lv2, lv2)
            self.att1 = AttentionGrid(lv1, lv1)
        elif attention == self.CHANNEL:
            # Care: attention channels forward only takes one param -> uses lambda to avoid issues with that
            self._att1 = AttentionChannel(lv1)
            self._att2 = AttentionChannel(lv2)
            self._att3 = AttentionChannel(lv3)
            self._att4 = AttentionChannel(lv4)
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


        # Descending path
        self.d1 = ConvBlock(self.INPUT_CHANNELS, lv1, nb_layers=nb_layers)
        self.d2 = ConvBlock(lv1, lv2, nb_layers=nb_layers)
        self.d3 = ConvBlock(lv2, lv3, nb_layers=nb_layers)
        self.d4 = ConvBlock(lv3, lv4, nb_layers=nb_layers)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bot = ConvBlock(lv4, lv5, nb_layers=nb_layers)

        # Ascending path
        
        self.up4 = nn.ConvTranspose2d(lv5, lv4, kernel_size=4, stride=2, padding=1)
        self.uconv4 = ConvBlock(lv5, lv4, nb_layers=nb_layers)
        self.up3 = nn.ConvTranspose2d(lv4, lv3, kernel_size=4, stride=2, padding=1)
        self.uconv3 = ConvBlock(lv4, lv3, nb_layers=nb_layers)
        self.up2 = nn.ConvTranspose2d(lv3, lv2, kernel_size=4, stride=2, padding=1)
        self.uconv2 = ConvBlock(lv3, lv2, nb_layers=nb_layers)
        self.up1 = nn.ConvTranspose2d(lv2, lv1, kernel_size=4, stride=2, padding=1)
        self.uconv1 = ConvBlock(lv2, lv1, nb_layers=nb_layers)
            
        # Combine channels pixel-wise to determine segmentation
        self.out_layer = nn.Conv2d(lv1, 1, kernel_size=1)
        self.out_norm = nn.BatchNorm2d(1)
        self.sigm = nn.Sigmoid()
        self.activation_output = activation_output

    def forward(self, x):
        out1 = self.d1(x)
        x2 = self.pool(out1)

        out2 = self.d2(x2)
        x3 = self.pool(out2)

        out3 = self.d3(x3)
        x4 = self.pool(out3)

        out4 = self.d4(x4)
        x5 = self.pool(out4)

        x_bot = self.bot(x5)

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
