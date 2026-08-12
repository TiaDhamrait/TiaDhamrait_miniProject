import glob
import os
import torch
from torch.utils.data import DataLoader, Dataset
import monai.transforms as mt


class MicroSegDataset(Dataset):
    """
    Dataset class to handle .mha medical ultrasound and segmentation files.
    Extracts 2D spatial slices from 3D volumes for model consumption.
    """

    def __init__(
        self, data_dir: str, split: str = "train", val_ratio: float = 0.2
    ):
        """
        Args:
            data_dir (str): Base path to the dataset folder.
            split (str): One of 'train' or 'val'.
            val_ratio (float): Fraction of slices reserved for validation.
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split

        # 1. Locate medical volume files (.mha)
        # Search nested directories in case files are in data/ or data/data/
        img_paths = glob.glob(
            os.path.join(data_dir, "**", "*ultrasound.mha"), recursive=True
        )
        seg_paths = glob.glob(
            os.path.join(data_dir, "**", "*segment.mha"), recursive=True
        )

        if not img_paths or not seg_paths:
            raise FileNotFoundError(
                f"Could not find .mha volume files in {data_dir}"
            )

        # 2. Load 3D volumes using MONAI's LoadImage transform (ITK backend)
        loader = mt.LoadImage(image_only=True, ensure_channel_first=True)
        raw_img_vol = loader(img_paths[0])  # Shape: (1, H, W, D) or (1, D, H, W)
        raw_seg_vol = loader(seg_paths[0])

        # Convert to standard PyTorch FloatTensors
        img_vol = torch.tensor(raw_img_vol, dtype=torch.float32)
        seg_vol = torch.tensor(raw_seg_vol, dtype=torch.float32)

        # Normalize mask values to binary {0, 1}
        seg_vol = (seg_vol > 0).float()

        # Normalize ultrasound image intensity to range [0, 1]
        img_vol = (img_vol - img_vol.min()) / (
            img_vol.max() - img_vol.min() + 1e-8
        )

        # Identify slice dimension (assuming channel-first 3D volume, depth is the last axis)
        num_slices = img_vol.shape[-1]

        # 3. Create deterministic Train / Val slice indices split
        num_val = int(num_slices * val_ratio)
        train_indices = list(range(0, num_slices - num_val))
        val_indices = list(range(num_slices - num_val, num_slices))

        chosen_indices = train_indices if split == "train" else val_indices

        # 4. Extract 2D slices corresponding to split
        self.images = [img_vol[..., i] for i in chosen_indices]
        self.masks = [seg_vol[..., i] for i in chosen_indices]

        # 5. Define augmentation / preprocessing transforms
        self.transforms = self._get_transforms()

    def _get_transforms(self):
        if self.split == "train":
            # Apply light spatial augmentations during training
            return mt.Compose(
                [
                    mt.RandFlip(prob=0.5, spatial_axis=0),
                    mt.RandGaussianNoise(prob=0.2, std=0.1),
                ]
            )
        else:
            # Identity pass for validation
            return mt.Compose([])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Apply spatial/intensity transformations if configured
        if self.transforms:
            image = self.transforms(image)

        return image, mask


def MicroSegLoader(dataset: Dataset, batch_size: int = 8, shuffle: bool = True):
    """
    Standard PyTorch DataLoader wrapper for MicroSegDataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )