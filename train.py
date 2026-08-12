import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW

# imports for models and utilities
from src.models import build_model
from src.utils.dataset import MicroSegDataset, MicroSegLoader
from src.utils.metrics import batch_segmentation_metrics


def train(args):
    # determine compute backend (prefer CUDA GPU, fall back to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ensure the target checkpoint/artifact folder exists
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data Loading & Preprocessing
    # ------------------------------------------------------------------
    # instantiate dataset splits (assuming transforms are handled internally or passed via args)
    train_dataset = MicroSegDataset(args.data_dir, split="train")
    val_dataset = MicroSegDataset(args.data_dir, split="val")

    # wrap datasets in DataLoaders; shuffle training batches to prevent sequence order bias
    train_loader = MicroSegLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = MicroSegLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # ------------------------------------------------------------------
    # 2. Model, Loss Function, and Optimizer Setup
    # ------------------------------------------------------------------
    # dynamically build selected model architecture (UNet, AttentionUNet, SegResNet, etc.)
    model = build_model(args.model_architecture).to(device)

    # define BCEWithLogitsLoss: combines Sigmoid activation + Binary Cross Entropy in a numerically stable way
    # works with single-channel logit outputs (out_channels=1)
    criterion = nn.BCEWithLogitsLoss()

    # AdamW provides adaptive learning rates with decoupled weight decay for better regularization
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )

    # track metrics across training epochs for visualization and early stopping
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
    }
    best_val_dice = 0.0

    # ------------------------------------------------------------------
    # 3. Epoch Loop
    # ------------------------------------------------------------------
    print(
        f"Starting training for {args.epochs} epochs on {args.model_architecture}..."
    )

    for epoch in range(1, args.epochs + 1):
        # set model to training mode (enables dropout, batch norm update)
        model.train()
        running_train_loss = 0.0

        for images, masks in train_loader:
            # push input tensors to target compute device
            images = images.to(device)
            masks = masks.to(device)

            # zero parameter gradients from previous iteration
            optimizer.zero_grad()

            # forward pass: compute predicted logits from input images
            outputs = model(images)

            # compute training loss between raw logits and ground-truth binary masks
            loss = criterion(outputs, masks)

            # backward pass: compute gradients via backpropagation
            loss.backward()

            # update model weights
            optimizer.step()

            # accumulate batch loss scaled by batch size
            running_train_loss += loss.item() * images.size(0)

        # compute average training loss over the full dataset epoch
        epoch_train_loss = running_train_loss / len(train_dataset)
        history["train_loss"].append(epoch_train_loss)

        # --------------------------------------------------------------
        # 4. Validation Loop
        # --------------------------------------------------------------
        # set model to evaluation mode (disables dropout, freezes batch norm)
        model.eval()
        running_val_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_val_batches = 0

        # disable gradient calculation to conserve memory and compute speed
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                # evaluate model on validation batch
                outputs = model(images)
                val_loss = criterion(outputs, masks)
                running_val_loss += val_loss.item() * images.size(0)

                # apply sigmoid threshold at 0.5 (logits >= 0.0) to get binarized predictions
                preds = (torch.sigmoid(outputs) > 0.5).float()

                # compute batch-level Dice and IoU using project metrics utility
                batch_metrics = batch_segmentation_metrics(preds, masks)
                total_dice += batch_metrics["dice"]
                total_iou += batch_metrics["iou"]
                num_val_batches += 1

        # calculate average validation statistics across the validation split
        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_dice = total_dice / max(num_val_batches, 1)
        epoch_val_iou = total_iou / max(num_val_batches, 1)

        # log stats into tracking history dictionary
        history["val_loss"].append(epoch_val_loss)
        history["val_dice"].append(epoch_val_dice)
        history["val_iou"].append(epoch_val_iou)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Dice: {epoch_val_dice:.4f} | "
            f"Val IoU: {epoch_val_iou:.4f}"
        )

        # --------------------------------------------------------------
        # 5. Checkpointing Best Model
        # --------------------------------------------------------------
        # save model state whenever validation Dice score reaches a new peak
        if epoch_val_dice > best_val_dice:
            best_val_dice = epoch_val_dice
            ckpt_path = os.path.join(
                args.ckpt_dir, f"best_{args.model_architecture}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": epoch_val_dice,
                    "val_iou": epoch_val_iou,
                },
                ckpt_path,
            )
            print(f"  [+] Saved new best model checkpoint to {ckpt_path}")

    # ------------------------------------------------------------------
    # 6. Plot & Save Training Curves
    # ------------------------------------------------------------------
    plot_training_curves(history, args)


def plot_training_curves(history, args):
    """Generate and save loss and metric curves for the training run."""
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss subplot
    axes[0].plot(
        epochs_range,
        history["train_loss"],
        label="Train Loss",
        color="tab:blue",
    )
    axes[0].plot(
        epochs_range,
        history["val_loss"],
        label="Val Loss",
        color="tab:orange",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title(f"Loss Curves ({args.model_architecture})")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Validation Metrics subplot
    axes[1].plot(
        epochs_range, history["val_dice"], label="Val Dice", color="tab:green"
    )
    axes[1].plot(
        epochs_range, history["val_iou"], label="Val IoU", color="tab:red"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"Validation Performance ({args.model_architecture})")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(
        args.ckpt_dir, f"learning_curves_{args.model_architecture}.png"
    )
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved training curves to {plot_path}")
    
    import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train segmentation models on MicroSeg"
    )
    parser.add_argument(
        "--model_architecture",
        type=str,
        default="unet",
        help="Architecture name (e.g., unet, attention_unet, segresnet)",
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for dataloaders"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints and learning curves",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing data folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)