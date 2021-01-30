import torch.nn as nn


class AttentionGrid(nn.Module):
    """
    Attention Grid from the paper "Attention U-Net".
    Uses input channels to create a pixel-wise mask,
    highlighting important parts of the image
    """
    def __init__(self, chann_to_weight, other_chann, inner_channels_factor=2):
        """Create attention mask for each pixels the channels

        Args:
            chann_to_weight (int): Number of channels of the input that we want to apply attention on
            other_chann (int): Other channels used to decide on where to focus
            inner_channels_factor (int, optional): Number of channels of the inner representation, where we combine all inputs before creating the mask. Defaults to 2.
        """

        super(AttentionGrid, self).__init__()

        # Need at least 1 channel to get an output
        inner_channels = chann_to_weight // inner_channels_factor
        if inner_channels == 0: inner_channels = 1

        # Warning, we only have one bias for both channels
        self.W_to_weight = nn.Sequential(
            nn.Conv2d(chann_to_weight, inner_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inner_channels)
        )

        self.W_other = nn.Sequential(
            nn.Conv2d(other_chann, inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels)
        )

        self.sigm1 = nn.ReLU()
        self.psi = nn.Sequential(
            nn.Conv2d(inner_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1)
        )

        self.sigm2 = nn.Sigmoid()
        self.out_channels = chann_to_weight

    def forward(self, x_to_weight, x_other):
        # Prepare inputs so that they have compatible shape
        x_w_in = self.W_to_weight(x_to_weight)
        x_o_in = self.W_other(x_other)
        # Activation
        out = self.sigm1(x_w_in + x_o_in)
        # Reduce to one channel (mask)
        out = self.psi(out)
        # Obtains alphas
        out = self.sigm2(out)
        return out * x_to_weight

