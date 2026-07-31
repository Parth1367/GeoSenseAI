import torch


def _threshold(predictions, threshold=0.5):
    """
    Convert logits to binary mask.
    """
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    return predictions


def iou_score(predictions, targets, smooth=1e-6):

    predictions = _threshold(predictions)

    intersection = (predictions * targets).sum()

    union = predictions.sum() + targets.sum() - intersection

    return ((intersection + smooth) / (union + smooth)).item()


def dice_score(predictions, targets, smooth=1e-6):

    predictions = _threshold(predictions)

    intersection = (predictions * targets).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        predictions.sum() + targets.sum() + smooth
    )

    return dice.item()


def precision_score(predictions, targets, smooth=1e-6):

    predictions = _threshold(predictions)

    tp = (predictions * targets).sum()

    fp = (predictions * (1 - targets)).sum()

    precision = (tp + smooth) / (tp + fp + smooth)

    return precision.item()


def recall_score(predictions, targets, smooth=1e-6):

    predictions = _threshold(predictions)

    tp = (predictions * targets).sum()

    fn = ((1 - predictions) * targets).sum()

    recall = (tp + smooth) / (tp + fn + smooth)

    return recall.item()


def f1_score(predictions, targets):

    p = precision_score(predictions, targets)

    r = recall_score(predictions, targets)

    return (2 * p * r) / (p + r + 1e-6)