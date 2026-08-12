"""
SwinUNETR
------------
Replaces the convolutional encoder with a hierarchical Swin Transformer:
self-attention computed within local, shifted windows at multiple scales,
which lets the network build long-range spatial relationships that a
purely convolutional encoder's local receptive field can't directly see.
A CNN-style decoder with skip connections (similar role to UNet's) upsamples
back to full resolution.

Pros
----
- Strong published results on several large-scale 3D medical benchmarks.

Cons / caveats
---------------
- Much larger and memory-hungry than every other model here.
  Expect noticeably slower training.
- Transformers are generally more data-hungry than CNNs. 
  Think about whether your dataset is large enough to justify this model.

MONAI reference: monai.networks.nets.SwinUNETR
"""
from monai.networks.nets import SwinUNETR


def build_swin_unetr(**kwargs):
    """Build a MONAI SwinUNETR configured for MicroSeg (see module docstring)."""
    spatial_dims = 2
    in_channels = 1

    ## TODO: choose these three values. feature_size must be divisible by 12;
    ## use_checkpoint trades compute time for lower memory use.
    #out_channels = None
    #feature_size = None
    #use_checkpoint = None

    #return SwinUNETR(
    #    spatial_dims=spatial_dims,
    #    in_channels=in_channels,
    #    out_channels=out_channels,
    #    feature_size=feature_size,
    #    use_checkpoint=use_checkpoint,
    #    **kwargs,
    #)

    # Comment this out and uncomment the above once you've chosen the values.
    raise NotImplementedError("Complete build_swin_unetr() in src/models/swin_unetr.py")
