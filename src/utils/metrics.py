import torch


def logits_to_binary_mask(logits, threshold=0.5):
    """Convert raw model outputs into a binary mask.

    Expected output shape:
      - (B, 1, H, W) if out_channels=1
      - (B, 2, H, W) if out_channels=2
    """
    out_channels = logits.shape[1]
    if out_channels == 1:
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()
    elif out_channels == 2:
        pred_class = torch.argmax(
            torch.softmax(logits, dim=1), dim=1, keepdim=True
        )
        return pred_class.float()
    else:
        raise ValueError(f"Unexpected out_channels: {out_channels}")


def dice_score(pred_mask, target_mask, eps=1e-6):
    """Compute mean Dice score over a batch.

    pred_mask and target_mask should already be binary tensors with matching
    shape, usually (B, 1, H, W).
    """
    # Flatten spatial dimensions per image: (B, -1)
    pred_flat = pred_mask.view(pred_mask.size(0), -1)
    target_flat = target_mask.view(target_mask.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice_per_image = (2.0 * intersection + eps) / (denominator + eps)
    return dice_per_image.mean()


def iou_score(pred_mask, target_mask, eps=1e-6):
    """Compute mean intersection-over-union (IoU) over a batch.

    pred_mask and target_mask should already be binary tensors with matching
    shape, usually (B, 1, H, W).
    """
    # Flatten spatial dimensions per image: (B, -1)
    pred_flat = pred_mask.view(pred_mask.size(0), -1)
    target_flat = target_mask.view(target_mask.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    iou_per_image = (intersection + eps) / (union + eps)
    return iou_per_image.mean()


def hausdorff_distance(pred_mask, target_mask):
    """Compute mean Hausdorff distance over a batch.

    pred_mask and target_mask should already be binary tensors with matching
    shape, usually (B, 1, H, W). Hausdorff distance is a boundary/shape error:
    lower is better.
    """
    batch_size = pred_mask.size(0)
    hausdorff_per_image = []

    for b in range(batch_size):
        # Extract foreground pixel coordinates (H, W) -> shape (N, 2)
        pred_coords = torch.nonzero(pred_mask[b, 0], as_tuple=False).float()
        target_coords = torch.nonzero(
            target_mask[b, 0], as_tuple=False
        ).float()

        # Handle edge cases where predictions or ground truths are empty
        if len(pred_coords) == 0 and len(target_coords) == 0:
            # Both empty: perfect match
            hausdorff_per_image.append(
                torch.tensor(0.0, device=pred_mask.device)
            )
            continue
        elif len(pred_coords) == 0 or len(target_coords) == 0:
            # One is empty: penalize with maximum spatial diagonal distance (~sqrt(H^2 + W^2))
            h, w = pred_mask.shape[2:]
            max_dist = torch.tensor(
                (h**2 + w**2) ** 0.5, device=pred_mask.device
            )
            hausdorff_per_image.append(max_dist)
            continue

        # Compute pairwise Euclidean distances between predicted and target points: shape (N_pred, N_target)
        dists = torch.cdist(pred_coords, target_coords, p=2)

        # Directed Hausdorff distance: max of nearest-neighbor distances
        pred_to_target = dists.min(dim=1)[0].max()
        target_to_pred = dists.min(dim=0)[0].max()

        # Undirected Hausdorff distance
        hd = torch.max(pred_to_target, target_to_pred)
        hausdorff_per_image.append(hd)

    return torch.stack(hausdorff_per_image).mean()


def batch_segmentation_metrics(logits, target_mask, threshold=0.5):
    """Convert model outputs to masks, then return the main metrics for one
    batch.

    Returns a small dictionary so train/eval loops can accumulate or print it.
    """
    pred_mask = logits_to_binary_mask(logits, threshold=threshold)

    return {
        "dice": dice_score(pred_mask, target_mask).item(),
        "iou": iou_score(pred_mask, target_mask).item(),
        "hausdorff": hausdorff_distance(pred_mask, target_mask).item(),
    }