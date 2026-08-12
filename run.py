"""
Model Runner

Usage:
    python run.py --list_models
    python run.py --mode train --data_dir data/ --model_architecture unet --ckpt_dir results/unet
    python run.py --mode eval  --data_dir data/ --model_architecture unet --checkpoint results/unet/best_model.pt
"""
import argparse

from train import train
from eval import evaluate
from src.models import MODEL_REGISTRY, list_models


def build_parser():
    parser = argparse.ArgumentParser(description="MicroSeg multi-model segmentation pipeline (A4 Part 1)")
    parser.add_argument("--mode", choices=["train", "eval"], default=None,
                         help="Whether to train a new model or evaluate an existing checkpoint.")
    parser.add_argument("--data_dir", type=str, default=None,
                         help="Path to the MicroSeg dataset.")
    parser.add_argument("--ckpt_dir", type=str, default="results/",
                         help="Where to save checkpoints, logs, and figures (training mode).")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a saved model checkpoint (required for eval mode).")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for reproducibility.")
    parser.add_argument("--model_architecture", type=str, default=None,
                         choices=list(MODEL_REGISTRY.keys()),
                         help="Which model to train/evaluate -- see src/models/ or --list_models.")
    parser.add_argument("--list_models", action="store_true",
                         help="Print a one-line summary of every available model in src/models/ and exit.")
    # TODO: add any additional arguments your pipeline needs,
    # e.g. --batch_size, --lr, --epochs
    return parser


def main():
    args = build_parser().parse_args()

    if args.list_models:
        list_models()
        return

    if args.mode is None or args.data_dir is None:
        raise ValueError("--mode and --data_dir are required (unless using --list_models)")
    if args.model_architecture is None:
        raise ValueError(f"--model_architecture is required. Choose from: {list(MODEL_REGISTRY.keys())}")

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in eval mode")
        evaluate(args)


if __name__ == "__main__":
    main()
