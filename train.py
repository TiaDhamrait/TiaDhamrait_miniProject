import os
import torch

from src.models import build_model, list_models  # noqa: F401 -- list_models used by run.py --list_models

# from src.utils.dataset import MicroSegDataset  # TODO: implement your Dataset class
# from src.utils.dataset import MicroSegLoader  # TODO: implement your DataLoader class
# from src.utils.metrics import batch_segmentation_metrics


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # TODO: Data loading & preprocessing
    #   - Build train/val Datasets and DataLoaders for MicroSeg
    #   - Decide your train/val/test split (see assignment instructions)
    #   - Define your image/mask transforms
    # ------------------------------------------------------------------
    # train_dataset = MicroSegDataset(args.data_dir, split="train", transform=...)
    # val_dataset   = MicroSegDataset(args.data_dir, split="val", transform=...)
    # train_loader = MicroSegLoader(train_dataset, batch_size=..., shuffle=True)
    # val_loader   = MicroSegLoader(val_dataset, batch_size=..., shuffle=False)

    # ------------------------------------------------------------------
    # TODO: Model, loss, optimizer
    #   - Choose a model: complete AT LEAST 3 of the stubs in src/models/
    #     (see that package's docstring, or run `python run.py --list_models`
    #     for a one-line summary of each architecture's pros/cons).
    #   - model = build_model(args.model_architecture).to(device)
    #   - Define your loss function (e.g. Dice, BCE, or a combination) and
    #     justify it in your report -- make sure its channel convention
    #     matches the out_channels you chose when completing the model.
    #   - Choose an optimizer and learning rate.
    # ------------------------------------------------------------------
    # model = build_model(args.model_architecture).to(device)
    # loss_fn = ...
    # optimizer = torch.optim.Adam(model.parameters(), lr=...)

    # ------------------------------------------------------------------
    # TODO: Training loop
    #   - Track training/validation loss and metric(s) per epoch
    #   - Use src/utils/metrics.py for validation Dice/IoU once completed
    #   - Save your best checkpoint to args.ckpt_dir
    #   - Save a training/validation curve figure to args.ckpt_dir for your report
    # ------------------------------------------------------------------
    raise NotImplementedError("Implement your training loop here.")
