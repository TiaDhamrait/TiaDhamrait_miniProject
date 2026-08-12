import torch

from src.models import build_model  # noqa: F401

# from src.utils.dataset import MicroSegDataset
# from src.utils.metrics import batch_segmentation_metrics


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # TODO: Load your test set
    # ------------------------------------------------------------------
    # test_dataset = MicroSegDataset(args.data_dir, split="test", transform=...)

    # ------------------------------------------------------------------
    # TODO: Load your model and checkpoint
    #   - Must match the same --model_architecture (and out_channels
    #     convention) used to produce the checkpoint you're loading.
    # ------------------------------------------------------------------
    # model = build_model(args.model_architecture).to(device)
    # model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    # model.eval()

    # ------------------------------------------------------------------
    # TODO: Run inference on the test set and report metrics
    #   - Quantitative metrics from src/utils/metrics.py, for your comparison
    #   - Qualitative overlays for at least 3 test images (save as figures)
    # ------------------------------------------------------------------
    raise NotImplementedError("Implement your evaluation loop here.")
