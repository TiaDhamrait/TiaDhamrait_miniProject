import argparse
import os
import matplotlib.pyplot as plt
import torch

from src.models import build_model
from src.utils.dataset import MicroSegDataset, MicroSegLoader
from src.utils.metrics import batch_segmentation_metrics, logits_to_binary_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation models on MicroSeg"
    )
    parser.add_argument(
        "--model_architecture",
        type=str,
        default="unet",
        help="Architecture name",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to checkpoint .pth file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save qualitative overlays",
    )
    return parser.parse_args()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Dataset (using 'val' or 'test' split)
    try:
        test_dataset = MicroSegDataset(args.data_dir, split="test")
    except Exception:
        print(
            "[!] 'test' split not found, falling back to 'val' split for evaluation."
        )
        test_dataset = MicroSegDataset(args.data_dir, split="val")

    test_loader = MicroSegLoader(test_dataset, batch_size=1, shuffle=False)

    # 2. Load Model & Weights
    model = build_model(args.model_architecture).to(device)
    checkpoint = torch.load(args.ckpt_path, map_location=device)

    # Handle checkpoints saved as dict or direct state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"[+] Loaded weights successfully from: {args.ckpt_path}")

    # 3. Quantitative Metric Accumulation
    dice_list, iou_list, hd_list = [], [], []

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            metrics = batch_segmentation_metrics(outputs, masks)
            dice_list.append(metrics["dice"])
            iou_list.append(metrics["iou"])
            hd_list.append(metrics["hausdorff"])

            # 4. Save Qualitative Overlays for the first 3 images
            if i < 3:
                pred_mask = (
                    logits_to_binary_mask(outputs).cpu().squeeze().numpy()
                )
                img_np = images.cpu().squeeze().numpy()
                gt_mask = masks.cpu().squeeze().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img_np, cmap="gray")
                axes[0].set_title("Ultrasound Image")
                axes[0].axis("off")

                axes[1].imshow(img_np, cmap="gray")
                axes[1].imshow(gt_mask, cmap="jet", alpha=0.5)
                axes[1].set_title("Ground Truth Mask")
                axes[1].axis("off")

                axes[2].imshow(img_np, cmap="gray")
                axes[2].imshow(pred_mask, cmap="jet", alpha=0.5)
                axes[2].set_title("Predicted Mask")
                axes[2].axis("off")

                save_path = os.path.join(
                    args.output_dir,
                    f"{args.model_architecture}_overlay_sample_{i+1}.png",
                )
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()
                print(f"[+] Saved qualitative figure to: {save_path}")

    # Summary Results
    mean_dice = sum(dice_list) / len(dice_list)
    mean_iou = sum(iou_list) / len(iou_list)
    mean_hd = sum(hd_list) / len(hd_list)

    print("\n" + "=" * 40)
    print(f"EVALUATION RESULTS ({args.model_architecture.upper()})")
    print("=" * 40)
    print(f"Mean Dice Score:         {mean_dice:.4f}")
    print(f"Mean IoU Score:          {mean_iou:.4f}")
    print(f"Mean Hausdorff Distance: {mean_hd:.4f}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)