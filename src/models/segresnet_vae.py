"""
SegResNetVAE
------------
Same segmentation backbone as SegResNet, but during training an extra
decoder branch tries to reconstruct the *input image* from the bottleneck
features through a VAE (with a KL-style regularization term). 

By forcing the bottleneck to also retain enough information to reconstruct the
image, this also ends up improving the shared encoder.

Pros
----
- Was specifically introduced to help in limited-data settings. 
  Very useful if the training set is small, and can be especially helpful 
  if the foreground structures are small and sparse.

Cons
---------------
- On noisy data, reconstructing the input can lead to artifact memorization, which 
  may hurt the performance.
- More parameters and memory than SegResNet, slower to train and easier to overfit on a small dataset.

MONAI reference: monai.networks.nets.SegResNetVAE
"""
from monai.networks.nets import SegResNetVAE


def build_segresnet_vae(**kwargs):
    """Build a MONAI SegResNetVAE configured for MicroSeg (see module docstring)."""
    spatial_dims = 2
    in_channels = 1

    ## TODO: choose these values
    #input_image_size = None
    #out_channels = None
    #init_filters = None
    #blocks_down = None
    #blocks_up = None            
    #vae_nz = None         # controls the latent bottleneck size.

    #return SegResNetVAE(
    #    input_image_size=input_image_size,
    #    spatial_dims=spatial_dims,
    #    in_channels=in_channels,
    #    out_channels=out_channels,
    #    init_filters=init_filters,
    #    blocks_down=blocks_down,
    #    blocks_up=blocks_up,
    #    vae_nz=vae_nz,
    #    **kwargs,
    #)

    # Comment this out and uncomment the above once you've chosen the values.
    raise NotImplementedError("Complete build_segresnet_vae() in src/models/segresnet_vae.py")
