# Copyright (c) Dandelion Science. Adapted from facebookresearch/tribev2,
# licensed under CC BY-NC 4.0 (see LICENSE in the root directory of this tree).

"""Entry points for ``torch.hub``.

Example: ``torch.hub.load("Dandelion-Science/tribev2:dandelion-v2",
"tribev2_brain_mapper", head="subcortical")``.
"""

dependencies = [
    "einops",
    "huggingface_hub",
    "neuraltrain",
    "torch",
    "torchvision",
    "transformers",
    "x_transformers",
    "yaml",
]

from tribev2.pretrained import load_brain_mapper as tribev2_brain_mapper  # noqa: E402, F401
from tribev2.video import VideoFeatures as tribev2_video_features  # noqa: E402, F401
