import random
import torch
import numpy as np
from utils import *


# For reproducibility
SEED = 8786
random.seed(SEED)
torch.random.manual_seed(SEED)
np.random.seed(SEED)

"""
CREATE THE MODEL SELECTION DATASET
"""

images, gt_images = load_original_dataset()
# Take 60% Training, 20% validation, 20% test
train_end = int(len(images) * 0.6)
val_end = train_end + int(len(images) * 0.2)

train_img, train_gt = images[:train_end], gt_images[:train_end]
val_img, val_gt = images[train_end: val_end], gt_images[train_end: val_end]
test_img, test_gt = images[val_end:], gt_images[val_end:]

NB_TRAINING_IMGS = 3_000
NB_VAL_IMGS = int(NB_TRAINING_IMGS * 0.2)
NB_TEST_IMGS = NB_VAL_IMGS
# For test images, we directly compute the score on the full images

# Different seeds, otherwise we get the same images
training_imgs, training_gt = data_augmentation(train_img, train_gt, nb_images=NB_TRAINING_IMGS, seed=234, ratios=(1, 3, 3, 2))
val_imgs, val_gts = data_augmentation(val_img, val_gt, nb_images=NB_VAL_IMGS, seed=999, ratios=(1, 3, 3, 2))
test_imgs, test_gts = data_augmentation(val_img, val_gt, nb_images=NB_TEST_IMGS, seed=3545, ratios=(1, 3, 3, 2))


# Save the two datasets
save_dataset(training_imgs, training_gt, name="msel_train/")
save_dataset(val_imgs, val_gts, name="msel_valid/")
save_dataset(test_imgs, test_gts, name="msel_test/")
print("Model selection datasets successfully created & saved")

# ----------------------------------------------------------------------------------------------------------
"""
TRAINING BEST MODEL DATASET
"""

images, gt_images = load_original_dataset()
# Take 80% Training, 20% validation
train_end = int(len(images) * 0.8)

train_img, train_gt = images[:train_end], gt_images[:train_end]
val_img, val_gt = images[train_end:], gt_images[train_end:]

NB_TRAINING_IMGS = 3_000
NB_VAL_IMGS = int(NB_TRAINING_IMGS * 0.2)

training_imgs, training_gt = data_augmentation(train_img, train_gt, nb_images=NB_TRAINING_IMGS, seed=234, ratios=(1, 3, 3, 2))
val_imgs, val_gts = data_augmentation(val_img, val_gt, nb_images=NB_VAL_IMGS, seed=999, ratios=(1, 3, 3, 2))

save_dataset(training_imgs, training_gt, name="train_clean/")
save_dataset(val_imgs, val_gts, name="valid_clean/")
print("Datasets for best model training successfully created & saved")