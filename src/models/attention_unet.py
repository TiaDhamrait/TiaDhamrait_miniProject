"""
AttentionUnet
------------
Structurally the same encoder-decoder as UNet, but each skip connection
passes through a small learned "attention gate" first. The gate uses the
coarser decoder features to decide which parts of the encoder features look
useful, then weakens the less useful regions before the skip connection is
merged into the decoder.

Pros
----
- Attention gates can learn to down-weight background clutter, leading to better segmentation of small structures.
- Adds relatively few parameters over UNet to stay lightweight.
- Attention maps are visualizable and interpretable

Cons
---------------
- Slightly more compute/memory than UNet.

MONAI reference: monai.networks.nets.AttentionUnet
"""
from monai.networks.nets import AttentionUnet


def build_attention_unet(**kwargs):
    """Build a MONAI AttentionUnet configured for MicroSeg (see module docstring)."""
    spatial_dims = 2 # 2D image slices (height x width)
    in_channels = 1 # one input channel (grayscale image)

    ## TODO: choose these three values.
    out_channels = 1
    channels = (16, 32, 64, 128, 256) 
    strides = (2, 2, 2, 2)

    return AttentionUnet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        **kwargs,
    )

    # Comment this out and uncomment the above once you've chosen the three values.
   # raise NotImplementedError("Complete build_attention_unet() in src/models/attention_unet.py")
