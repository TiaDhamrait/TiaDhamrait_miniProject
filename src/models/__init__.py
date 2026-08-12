"""
Models Registry
---------------
Maps architecture string identifiers to builder functions defined across the model submodules.
"""

from src.models.attention_unet import build_attention_unet
from src.models.basic_unetplusplus import (
    build_basic_unetplusplus,  # available if implemented later
)
from src.models.segresnet import build_segresnet
from src.models.segresnet_vae import (
    build_segresnet_vae,  # available if implemented later
)
from src.models.swin_unetr import (
    build_swin_unetr,  # available if implemented later
)
from src.models.unet import build_unet
from src.models.vnet import build_vnet  # available if implemented later

# dictionary registry mapping lower-case string keys to model builder functions
MODEL_REGISTRY = {
    "unet": build_unet,
    "attention_unet": build_attention_unet,
    "segresnet": build_segresnet,
    # optional stubs dynamically referenced if needed by run.py --list_models
    "basic_unetplusplus": build_basic_unetplusplus,
    "segresnet_vae": build_segresnet_vae,
    "swin_unetr": build_swin_unetr,
    "vnet": build_vnet,
}


def list_models():
    """Return a list of available model architecture keys registered in the codebase."""
    # return sorted keys so run.py can display options consistently
    return sorted(list(MODEL_REGISTRY.keys()))


def build_model(model_name: str, **kwargs):
    """
    Factory function to instantiate a model architecture by name.

    Args:
        model_name (str): Key matching one of the registered architecture strings.
        **kwargs: Additional parameters passed through to the model builder.
    """
    # normalize user string input to lowercase to prevent casing mismatches
    key = model_name.lower()

    # verify the requested model exists in our dictionary registry
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model architecture '{model_name}'. Available choices: {list_models()}"
        )

    # retrieve builder function and instantiate model
    builder_fn = MODEL_REGISTRY[key]
    return builder_fn(**kwargs)