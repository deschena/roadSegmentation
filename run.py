import numpy as np
import torch
import os
from PIL import Image

from datasets.ToLabelDataset import ToLabelDataset
from models.DenseUnet import DenseUnet
from utils import *
from mask_to_submission import *


def create_submission(submission_filename, temp_dirname, model):
    
    test_dataset = ToLabelDataset()
    if not os.path.exists(temp_dirname):
        os.mkdir(temp_dirname)

    image_filenames = []
    for i in range(50):
        img = test_dataset[i]
        img = img.to(device).reshape(1, 3, 608, 608)
        pred = predict_larger_image(img, model)
        # threshold
        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1
        i = i + 1 # This is due to the way the images are labelled from 1 to 50, instead of the standard 0 to 49
        image_filename = temp_dirname + 'prediction_' + '%.3d' % i + '.png'
        Image.fromarray((pred * 255).astype(np.uint8)).save(image_filename)
        image_filenames.append(image_filename)

    masks_to_submission(submission_filename, *image_filenames)

def load_net_params(net, name):
    path = "experiments/best_arch/" + name + ".pth"
    params = torch.load(path)
    net.load_state_dict(params)
    net.eval()
    net.to("cuda")
    return net


net4 = DenseUnet(down_config=(4, 8, 16, 32), bottom=64, up_channels=(256, 128, 64, 32), activation_output=True, attention="grid")
net4 = load_net_params(net4, "dense_unet_massach")
create_submission("dense_unet_massach.csv", "denseunet_massach_preds/", net4)