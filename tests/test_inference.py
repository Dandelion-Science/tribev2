# Copyright (c) Dandelion Science. Adapted from facebookresearch/tribev2,
# licensed under CC BY-NC 4.0 (see LICENSE in the root directory of this tree).

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tribev2.model import FmriEncoder


def test_import_does_not_load_event_stack():
    code = (
        "import sys, tribev2; "
        "assert not [m for m in sys.modules if m.startswith('neuralset')]"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_forward_accepts_plain_dict():
    model = FmriEncoder(
        hidden=64,  # x-transformers needs dim // heads >= 32
        low_rank_head=8,
        encoder={"heads": 2, "depth": 1},
        subject_layers={
            "n_subjects": 0,
            "subject_dropout": 0.1,
            "average_subjects": True,
        },
    ).build(feature_dims={"video": (2, 8)}, n_outputs=5, n_output_timesteps=10)
    x = torch.randn(3, 2, 8, 20)
    model.eval()
    out = model({"video": x})
    assert out.shape == (3, 5, 10)
    torch.testing.assert_close(out, model(SimpleNamespace(data={"video": x})))
    assert model({"video": x}, pool_outputs=False).shape == (3, 5, 20)
    with pytest.raises(ValueError, match="no known modality"):
        model({"videos": x})
