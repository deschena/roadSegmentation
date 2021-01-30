import numpy as np
import torch
import torchvision
import matplotlib.image as mpimg
import os
import random
from PIL import Image
from sklearn.utils import shuffle



def load_image(filename):
    """Load image using matplotlib

    Args:
        filename (str): Path to image

    Returns:
        np.ndarray: numpy array containing the image
    """
    data = mpimg.imread(filename)
    return data
# ----------------------------------------------------------------------------------------------------------


def random_pair(images, gt_images):
    """Samples a random pair of elements from the two input iterables

    Args:
        images (array): 
        gt_images (array): 

    Returns:
        (array, array): Pair of elements of the array (typically array themselves, like images)
    """
    i = np.random.randint(0, len(images))
    return images[i], gt_images[i]
# ----------------------------------------------------------------------------------------------------------


def load_original_dataset(shuffle_data=True):
    """Load provided Aerial image dataset as numpy array. Colors are 8bit encoded

    Returns:
        (array, array): (aerial images, ground-truth masks)
    """
    root_dir = "training/"
    gt_dir = root_dir + "groundtruth/"
    images_dir = root_dir + "images/"
    files = os.listdir(images_dir)
    # Filter .ipynb_checkpoints that somehow appears here
    files = [f for f in files if f[0] != "."]
    n = len(files)
    print("Loading " + str(n) + " images")

    # Compress images on 8 bits, otherwise they use too much memory
    imgs = [(load_image(images_dir + files[i]) * 255).astype(np.uint8)
            for i in range(n)]
    gt_imgs = [(load_image(gt_dir + files[i]) * 255).astype(np.uint8)
               for i in range(n)]

    # Map numpy arrays into Image class
    def l(x): return Image.fromarray(x)
    imgs = list(map(l, imgs))
    gt_imgs = list(map(l, gt_imgs))

    # Shuffle images
    if shuffle_data:
        imgs, gt_imgs = shuffle(imgs, gt_imgs)

    return imgs, gt_imgs
# ----------------------------------------------------------------------------------------------------------


def rotate_img(img, gt_img, angle=45):
    """Rotates the provided image and ground-truth. Note that the shape of the image may change 
    to avoid adding dead pixels around the image after rotation. Expects square images

    Args:
        img (array): Image to transform
        gt_img (array): Segmentation mask of the image
        angle (int, optional): Angle of the rotation. Defaults to 45.

    Returns:
        (array, array): (rotated image, rotated mask)
    """
    angle = angle % 360

    if angle == 0:
        return img, gt_img

    # The code is easier when considering only angles between 0 and 90
    while angle > 90:
        img, gt_img = rotate_img(img, gt_img, 90)
        angle = angle - 90

    img = torchvision.transforms.functional.affine(
        img, angle, (0, 0), 1, 0, fillcolor=0, resample=Image.NEAREST)
    img = np.array(img)

    gt_img = torchvision.transforms.functional.affine(
        gt_img, angle, (0, 0), 1, 0, fillcolor=0, resample=Image.NEAREST)
    gt_img = np.array(gt_img)
    size = img.shape[0]
    angle_r = angle * np.pi / 180
    t = np.tan(angle_r)
    target_size = size * t / (np.sin(angle_r)*(1 + t))
    target_size = min(target_size, size)
    margin = int((size - target_size) / 2)

    img = img[margin: size - margin, margin: size - margin]
    gt_img = gt_img[margin: size - margin, margin: size - margin]
    # Transform in image again
    return Image.fromarray(img), Image.fromarray(gt_img)
# ----------------------------------------------------------------------------------------------------------


def shear_img(img, gt_img, shear=0, target_size=256):
    """Applies a shear transformation on image

    Args:
        img (array): Image to transform
        gt_img (array): Segmentation mask of the image
        shear ({int, tuple}, optional): Shearing parameter. Can be a tuple to affect both axis differently. Defaults to 0.
        target_size (int, optional): Shape of the output image. Defaults to 256.

    Returns:
        (array, array): (sheared image, sheared segmentation mask)
    """

    img = torchvision.transforms.functional.affine(
        img, 0, (0, 0), 1, shear, fillcolor=0, resample=Image.NEAREST)
    img = np.array(img)

    gt_img = torchvision.transforms.functional.affine(
        gt_img, 0, (0, 0), 1, shear, fillcolor=0, resample=Image.NEAREST)
    gt_img = np.array(gt_img)

    size = img.shape[0]
    margin = int((size - target_size) / 2)
    img = img[margin: size - margin, margin: size - margin]
    gt_img = gt_img[margin: size - margin, margin: size - margin]
    return Image.fromarray(img), Image.fromarray(gt_img)
# ----------------------------------------------------------------------------------------------------------


def data_augmentation(images, gt_images, nb_images=2000, ratios=(2, 2, 2, 1), size=256, seed=12):
    """
    Kinds of transforms
    1) Five crop
    2) Random rotation, crop, jitter
    3) Random shear, jitter
    4) Random rotation, shear, jitter

    Input images, and gt_images are expected to be instances of the PIL Image class
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    total = sum(ratios)
    # Defines which fraction of the images must be of each transform kind
    fractions = (np.array(ratios) / total * nb_images).astype(int)
    # Makes sure that we have nb_images in total
    fractions[0] = 0
    sub_total = np.sum(fractions)
    fractions[0] = nb_images - sub_total

    angles = np.linspace(-180, 180, 16)[:-1]
    shears = np.linspace(-5, 5, 11)
    # Values were chosen so that images are still recognizable, not too dark but still noticeably modified
    jitters = torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
    crops = torchvision.transforms.FiveCrop(size)

    all_images = []
    all_gt_images = []

    # 1) Random crops
    for _ in range(fractions[0]):
        img, gt_img = random_pair(images, gt_images)
        # crop and pick a crop (returns tuple of 5 crops)
        img = crops(img)
        gt_img = crops(gt_img)
        # Pick a crop
        img, gt_img = random_pair(img, gt_img)
        # Save in list of images
        all_images.append(img)
        all_gt_images.append(gt_img)

    # 2) Random rotation, crop, jitter
    for _ in range(fractions[1]):
        a = np.random.choice(angles)
        img, gt_img = random_pair(images, gt_images)
        # Rotate random image
        img, gt_img = rotate_img(img, gt_img, a)
        # Crop
        img = crops(img)
        gt_img = crops(gt_img)
        img, gt_img = random_pair(img, gt_img)
        # Random jitter on image
        img = jitters(img)
        # Save
        all_images.append(img)
        all_gt_images.append(gt_img)

    # 3) Random shear, jitter
    for _ in range(fractions[2]):
        # Pick shears
        s_x = np.random.choice(shears)
        s_y = np.random.choice(shears)

        img, gt_img = random_pair(images, gt_images)
        # transforms
        img, gt_img = shear_img(
            img, gt_img, shear=(s_x, s_y), target_size=size)
        img = jitters(img)
        # Save
        all_images.append(img)
        all_gt_images.append(gt_img)

    # 4) Random rotation, shear, jitter
    for _ in range(fractions[3]):
        # Parameters
        a = np.random.choice(angles)
        s_x = np.random.choice(shears)
        s_y = np.random.choice(shears)

        img, gt_img = random_pair(images, gt_images)
        # Transforms
        img, gt_img = rotate_img(img, gt_img, a)
        img, gt_img = shear_img(
            img, gt_img, shear=(s_x, s_y), target_size=size)
        img = jitters(img)
        # Save
        all_images.append(img)
        all_gt_images.append(gt_img)

    # Return augmented dataset
    return all_images, all_gt_images
# ----------------------------------------------------------------------------------------------------------


def save_dataset(images, gt_images, save_path="augmented_dataset/", name="default/"):
    """Save the created dataset in the directly save_path/name. If they do not exist, create them.
    Note that the ordering of images and gt_images is expected to be the same.

    Args:
        images (array): Array of images
        gt_images (array): Array of segmentation masks
        save_path (str, optional): Path to directory containing all datasets. Defaults to "augmented_dataset_justin/".
        name (str, optional): Name of dataset. Defaults to "default/".
    """
    # Check that directories exist
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    dataset_path = save_path + name
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    # Create directories images and ground truth
    img_path = dataset_path + "images/"
    gt_path = dataset_path + "ground_truth/"
    os.mkdir(img_path)
    os.mkdir(gt_path)

    for i, img, gt, in zip(range(len(images)), images, gt_images):
        img.save(img_path + f"img_{i}.png", format="png")
        gt.save(gt_path + f"img_{i}.png", format="png")
# ----------------------------------------------------------------------------------------------------------