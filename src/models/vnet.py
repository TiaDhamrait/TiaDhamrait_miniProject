"""
VNet
------------
An encoder-decoder model where each stage is a residual block of convolutions
(input added back to output), downsampling via strided convolutions rather
than pooling. It was introduced together with the Dice loss specifically
to address severe class imbalance between the foreground and background classes.

Pros
----
- Designed specifically with class imbalance in mind.
- Residual blocks throughout aid gradient flow
- Supports 2D or 3D via `spatial_dims`.

Cons
---------------
- Originally a 3D architecture (volumetric CT/MRI); 
  2D usage is supported but less common in the literature.
- Comparatively heavier than U-Net for 2D images

MONAI reference: monai.networks.nets.VNet
"""
from monai.networks.nets import VNet


def build_vnet(**kwargs):
    """Build a MONAI VNet configured for MicroSeg (see module docstring)."""
    spatial_dims = 2
    in_channels = 1

    ## TODO: choose these values. Keep out_channels consistent with your
    ## other models, and decide how much dropout to use.
    #out_channels = None
    #dropout_prob = None

    #return VNet(
    #    spatial_dims=spatial_dims,
    #    in_channels=in_channels,
    #    out_channels=out_channels,
    #    dropout_prob=dropout_prob,
    #    **kwargs,
    #)

    # Comment this out and uncomment the above once you've chosen the values.
    raise NotImplementedError("Complete build_vnet() in src/models/vnet.py")
