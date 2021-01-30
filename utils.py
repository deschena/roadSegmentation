import numpy as np
import torch
import matplotlib.image as mpimg
from metrics import f1_score


def dice_coeff(pred, target, smooth=1):
    """Compute Sørensen-Dice coefficient

    Args:
        pred (array): Prediction result
        target (array): Ground-truth
        smooth (int, optional): Smoothing of distribution, avoids issues with divisions by zero. Defaults to 1.

    Returns:
        Number: Computed coefficient
    """
    # Shape of the form (batch size, channels, width, height) -> don't want to aggregate over batches to compute dice loss
    # Only per example

    # b size x 1 x 1 x 1
    nomin = torch.sum(2 * pred * target, dim=(1, 2, 3)) + smooth
    denom = torch.sum(pred ** 2 + target ** 2, dim=(1, 2, 3)) + smooth

    # Dice loss for each prediction in the batch
    dice_loss_batch = nomin / denom

    return dice_loss_batch.mean()
# ----------------------------------------------------------------------------------------------------------


def dice_loss(pred, target, smooth=1):
    """Compute loss based on Sørensen-Dice. That is 1 - Sørensen-Dice.

    Args:
        pred (array): Prediction result
        target (array): Ground-truth
        smooth (int, optional): Smoothing of distribution, avoids issues with divisions by zero. Defaults to 1.

    Returns:
        Number: Computed loss
    """
    return 1 - dice_coeff(pred, target, smooth=smooth)
# ----------------------------------------------------------------------------------------------------------

def composite_loss(closs, smooth=1, alpha=1., beta=0.001):
    """Create composite loss function using Sørensen-Dice loss and an other one.

    Args:
        closs (function): second loss
        smooth (int, optional): Smooting param of the Sørensen-Dice loss. Defaults to 1.
        alpha (float, optional): Weight of the Sørensen-Dice loss. Defaults to 1.
        beta (float, optional): Weight of the second loss. Defaults to 0.001.

    Returns:
        function: New composite loss function
    """
    return lambda pred, target: alpha * dice_loss(pred, target, smooth) + beta * closs(pred, target)
# ----------------------------------------------------------------------------------------------------------


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

def predict_larger_image(img, model, excess = 0, model_input = 256):
    """Compute segmentation of an image that has different size to the one used by the network

    Args:
        img (array): Image to segment
        model (nn.Module): Pytorch model used for prediction
        excess (int, optional): By default, predict as few patches as possible. Change this value to predict more overlapping patches of the images (scales on both axes). Defaults to 0.
        model_input (int, optional): Expected side length of the input of the model. Defaults to 256.

    Returns:
        array: Segmented (fully) image
    """
    # img shape: (batch size = 1, channels, height, width)
    width = img.shape[2]
    height = img.shape[3]
    N_V = height // model_input + 1 + excess # Number of images on vertical axis
    N_H = width // model_input + 1 + excess # Number of images on horizontal axis
    r_h = np.round(np.linspace(0, width  - model_input,  N_H)).astype("int") # starting points on h axis
    r_v = np.round(np.linspace(0, height - model_input, N_V)).astype("int") # starting points on v axis
    
    result = np.zeros((width, height))
    mask = np.zeros((width, height))
    # NOTE: cast of indices is to avoid type errors, as indices 
    for v in r_v:
        v = int(v)
        for h in r_h:
            h = int(h)
            sub_image = img[:, :, v: v + model_input, h: h + model_input]
            with torch.no_grad():
                # Predict
                sub_pred = model(sub_image).squeeze()
                sub_pred = sub_pred.to("cpu").numpy()
                # Add to total
                result[v: v + model_input, h: h + model_input] += sub_pred
                mask[v: v + model_input, h: h + model_input] += np.ones((model_input, model_input))
    result /= mask
    return result
# ----------------------------------------------------------------------------------------------------------

def validation_perf(model, loader, crit=f1_score):
    """Compute performance of the model on the validation set

    Args:
        model (nn.Module): Pytorch model stored on GPU
        loader (DataLoader): Dataloader of the validation set
        crit (function, optional): Criterion to measure performance. Defaults to f1_score.

    Returns:
        float: Mean performance on the validation set
    """
    # Compute F1-score of the model on the test set
    model.eval() # Note: the model is on the GPU: must send images on GPU as well
    total_loss = 0
    with torch.no_grad():
        for batch_test, batch_gt in loader:
            # Send to GPU, then to cpu to compute f1-score
            batch_test = batch_test.to("cuda")
            batch_gt = batch_gt.to("cpu").squeeze()
            pred = model(batch_test)
            pred = pred.to("cpu").squeeze()
            total_loss += crit(pred, batch_gt)
    return total_loss / len(loader)
# ----------------------------------------------------------------------------------------------------------