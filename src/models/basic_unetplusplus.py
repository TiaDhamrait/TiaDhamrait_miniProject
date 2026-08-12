"""
BasicUNetPlusPlus
------------
UNet++ inserts a grid of nested densely-connected convolutional blocks
between the encoder and decoder, so decoder features at each level are
built up from multiple intermediate feature maps. 

The intent is to reduce the "semantic gap" between what the
encoder and decoder see at each resolution. 

Pros
----
- More gradual feature fusion than a single skip connection. Can improve boundary precision.

Cons
---------------
- Substantially more parameters and memory than UNet. Slower to train, and easier to overfit 
on a small dataset.

MONAI reference: monai.networks.nets.BasicUNetPlusPlus
"""
from monai.networks.nets import BasicUNetPlusPlus


def build_basic_unetplusplus(**kwargs):
    """Build a MONAI BasicUNetPlusPlus configured for MicroSeg (see module docstring)."""
    spatial_dims = 2
    in_channels = 1

    ## TODO: choose these three values. features controls the encoder/decoder
    ## widths; deep_supervision changes what forward() returns.
    #out_channels = None
    #features = None
    #deep_supervision = None

    #return BasicUNetPlusPlus(
    #    spatial_dims=spatial_dims,
    #    in_channels=in_channels,
    #    out_channels=out_channels,
    #    features=features,
    #    deep_supervision=deep_supervision,
    #    **kwargs,
    #)

    # Comment this out and uncomment the above once you've chosen the values.
    raise NotImplementedError("Complete build_basic_unetplusplus() in src/models/basic_unetplusplus.py")
