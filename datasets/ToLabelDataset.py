import torch
from torch.utils.data import Dataset
import os
from utils import *


class ToLabelDataset(Dataset):
    """Loader for images to label"""

    def __init__(self, root_dir="test_images/"):
        """Loader for images to label

        Args:
            root_dir (str, optional): Path to the directory containing images to label. Defaults to "test_images/".
        """
        super(ToLabelDataset, self).__init__()
        self.dir_path = root_dir
        files = os.listdir(self.dir_path)
        
        # Filter .iypnb saves
        files = [f for f in files if f[0] != "."]
        self.size = len(files)
        
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Pytorch shape convention: (batch size, channels, width, height)
        index += 1
        path = self.dir_path + f"test_{index}/test_{index}.png"
        img = load_image(path)
        img = torch.tensor(img).permute(2, 0, 1)
        
        return img