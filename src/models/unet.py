"""
UNet
------------
A symmetric encoder-decoder: the encoder repeatedly downsamples the image
while increasing feature channels, the decoder upsamples back to full
resolution, and skip connections copy encoder features across to the
matching decoder level so fine spatial detail isn't lost to downsampling.

Pros
----
- Simple and well understood. The most popular baseline model for medical image segmentation.
- Fast to train and cheap at inference relative to most alternatives below.

Cons
---------------
- No explicit mechanism to suppress irrelevant background.
- Purely local features per layer: stacking multiple layers is needed
  to see large-scale context, which can be costly.

MONAI reference: monai.networks.nets.UNet
"""
from monai.networks.nets import UNet


def build_unet(**kwargs):
    """Build a MONAI UNet configured for MicroSeg."""
    model_args = {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "channels": (16, 32, 64, 128, 256),
        "strides": (2, 2, 2, 2),
        "num_res_units": 0,
    }
    model_args.update(kwargs)

    return UNet(**model_args)
