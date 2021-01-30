import numpy as np

def true_positives(pred, target, threshold = 0.5):
    """Compute True positive rate over segmentation masks

    Args:
        pred (array): Predicted mask
        target (array): Target mask
        threshold (float, optional): We deal with binary categories, but predictions in range [0, 1]. Hence need to discretize above/under threshold. Defaults to 0.5.

    Returns:
        float: Number of true positive detections
    """
    #true positive
    return len(np.where(np.logical_and(pred >= threshold, target >= threshold))[0])

def positives(pred, threshold = 0.5):
    #positive
    return len(np.where(pred >= threshold)[0])

def f1_score(pred, target, threshold = 0.5):
    """Compute the F1 score of the prediction

    Args:
        pred (array): Prediction result
        target (array): Ground-truth
        threshold (float, optional): Decision threshold. See true_positive function. Defaults to 0.5.

    Returns:
        float: F1-score
    """
    tp = true_positives(pred, target, threshold)
    pos = positives(pred, threshold)
    gt_pos = positives(target, threshold)
    if pos == 0 or gt_pos == 0 or tp == 0:
        return 0
    precision = tp / pos
    recall = tp / gt_pos
    return 2 / (1/precision + 1/recall)