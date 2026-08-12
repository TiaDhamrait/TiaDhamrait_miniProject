import torch


def logits_to_binary_mask(logits, threshold=0.5):
    """
    Convert raw model outputs into a binary mask.

    Expected output shape:
      - (B, 1, H, W) if out_channels=1
      - (B, 2, H, W) if out_channels=2
    """
    # Helper function to conver t logits to binary mask. 
    # Feel free to modify this if you want.
    out_channels = logits.shape[1]
    if out_channels == 1:
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()
    elif out_channels == 2:
        pred_class = torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=True)
        return pred_class.float()
    else:
        raise ValueError(f"Unexpected out_channels: {out_channels}")


def dice_score(pred_mask, target_mask, eps=1e-6):
    """
    Compute mean Dice score over a batch.

    pred_mask and target_mask should already be binary tensors with matching
    shape, usually (B, 1, H, W).
    """
    # TODO: fill in the Dice formula.
    #intersection = ...
    #denominator = ...
    #dice_per_image = ...
    #return dice_per_image.mean()

    raise NotImplementedError("Complete dice_score() in src/utils/metrics.py")


def iou_score(pred_mask, target_mask, eps=1e-6):
    """
    Compute mean intersection-over-union (IoU) over a batch.

    pred_mask and target_mask should already be binary tensors with matching
    shape, usually (B, 1, H, W).
    """
    # TODO: fill in the IoU formula.
    #intersection = ...
    #union = ...
    #iou_per_image = ...
    #return iou_per_image.mean()

    raise NotImplementedError("Complete iou_score() in src/utils/metrics.py")


def hausdorff_distance(pred_mask, target_mask):
    """
    Compute mean Hausdorff distance over a batch.

    pred_mask and target_mask should already be binary tensors with matching
    shape, usually (B, 1, H, W). Hausdorff distance is a boundary/shape error:
    lower is better.
    """
    # TODO: fill in the Hausdorff distance calculation.
    #
    # Hint: for each image in the batch:
    #   1. get the foreground pixel coordinates from pred_mask and target_mask
    #      using torch.nonzero(...)
    #   2. compute all pairwise distances between the two coordinate sets
    #      using torch.cdist(...)
    #   3. compute the largest nearest-neighbor distance in both directions:
    #        pred_to_target = ...
    #        target_to_pred = ...
    #   4. the Hausdorff distance is max(pred_to_target, target_to_pred)
    #
    # You also need to decide what to return if one mask is empty.
    #hausdorff_per_image = []
    #...
    #return torch.stack(hausdorff_per_image).mean()

    raise NotImplementedError("Complete hausdorff_distance() in src/utils/metrics.py")


def batch_segmentation_metrics(logits, target_mask, threshold=0.5):
    """
    Convert model outputs to masks, then return the main metrics for one batch.

    Returns a small dictionary so train/eval loops can accumulate or print it.
    """
    pred_mask = logits_to_binary_mask(logits, threshold=threshold)

    return {
        "dice": dice_score(pred_mask, target_mask),
        "iou": iou_score(pred_mask, target_mask),
        "hausdorff": hausdorff_distance(pred_mask, target_mask),
    }
