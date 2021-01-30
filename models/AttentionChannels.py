import torch.nn as nn


class AttentionChannel(nn.Module):
    """
    Attention mechanism taken from SENet architecture.
    """
    
    def __init__(self, nb_channels, se_bn=16):
        """Create channel-wise attention mechanism. Determines which ones are the most relevant using a bottleneck fully connected network

        Args:
            nb_channels (int): Number of input channels
            se_bn (int, optional): Fraction of channels inside fc net. Defaults to 16.
        """
        super(AttentionChannel, self).__init__()
        # Reduces images to one value per channel
        self.reduce = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(nb_channels, nb_channels // se_bn)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(nb_channels // se_bn, nb_channels)
        self.sg = nn.Sigmoid()
        
    def forward(self, x):
        # Remove dimensions 3 and 2 whose shape are 1 after the pooling
        out = self.reduce(x).squeeze(3).squeeze(2)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sg(out)
        # Add them again to match the shape of x
        out = out.unsqueeze(2).unsqueeze(3)
        # Weight channels
        return x * out