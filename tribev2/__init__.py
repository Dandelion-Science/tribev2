# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tribev2.pretrained import load_brain_mapper
from tribev2.video import VideoFeatures, zero_fill

__all__ = ["TribeModel", "VideoFeatures", "load_brain_mapper", "zero_fill"]


def __getattr__(name):
    # Lazy, so the inference API does not import the neuralset event stack.
    if name == "TribeModel":
        try:
            from tribev2.demo_utils import TribeModel
        except ImportError as e:
            raise ImportError(
                "TribeModel needs the original release environment: "
                'pip install "tribev2[demo]" (Python <= 3.13)'
            ) from e
        return TribeModel
    raise AttributeError(f"module 'tribev2' has no attribute {name!r}")
