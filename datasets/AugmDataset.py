import torch
from torch.utils.data import Dataset
import os
from utils import *




class AugmDataset(Dataset):
    """Easy loading of images resulting from data augmentation"""

    def __init__(self, root_dir="augmented_dataset/", name="default/"):
        """Dataset for the augmented images set.

        Args:
            root_dir (str, optional): Path containing all datasets. Defaults to "augmented_dataset_justin/".
            name (str, optional): Name of the dataset in general directory. Defaults to "default/".
        """     
        super(AugmDataset, self).__init__()
        self.images_path = root_dir + name + "images/"
        self.gt_path = root_dir + name + "ground_truth/"
        files = os.listdir(self.images_path)
        
        # Filter .iypnb saves
        self.files = [f for f in files if f[0] != "."]
        self.size = len(files)
        
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Pytorch shape convention: (batch size, channels, width, height)
        img = load_image(self.images_path + self.files[index])
        gt = load_image(self.gt_path + self.files[index])
        img = torch.tensor(img).permute(2, 0, 1)
        gt = torch.tensor(gt[:, :, np.newaxis]).permute(2, 0, 1)
        
        return img, gt