"""
SegResNet 
------------
An encoder-decoder built from residual blocks (each block adds its input
back to its output) with group normalization, using more down-blocks in the
encoder than up-blocks in the decoder by default. 

Pros
----
- Residual connections make deeper networks easier to train.
- Consistently competitive across many medical segmentation benchmarks.
- Supports 2D or 3D inputs directly via `spatial_dims`.

Cons
---------------
- More parameters and computational burden than UNet at a comparable depth.

MONAI reference: monai.networks.nets.SegResNet
"""
from monai.networks.nets import SegResNet


def build_segresnet(**kwargs):
    """Build a MONAI SegResNet configured for MicroSeg (see module docstring)."""
    spatial_dims = 2
    in_channels = 1

    ## TODO: choose these four values. Keep out_channels consistent with
    ## your other models, then decide how wide/deep this residual model is.
# output single logit map for binary target evaluation
    out_channels = 1

    init_filters = 16

    blocks_down = (1, 2, 2, 4)

    blocks_up = (1, 1, 1)

    return SegResNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        **kwargs,
    )

    # Comment this out and uncomment the above once you've chosen the values.
    # raise NotImplementedError("Complete build_segresnet() in src/models/segresnet.py")
