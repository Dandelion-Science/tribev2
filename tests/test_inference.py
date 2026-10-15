# Copyright (c) Dandelion Science. Adapted from facebookresearch/tribev2,
# licensed under CC BY-NC 4.0 (see LICENSE in the root directory of this tree).

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tribev2.model import FmriEncoder
from tribev2.pretrained import load_brain_mapper
from tribev2.video import VideoFeatures, window_indices, zero_fill

DATA = Path(__file__).parent / "data"
# Directory with tribev2-{cortical,subcortical}/ checkpoints; default: download.
CHECKPOINT_ROOT = os.environ.get("TRIBEV2_CHECKPOINT_ROOT")


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


def test_window_indices_match_release():
    # 16 fps, 1 s: sample 0 ends at 0.5 s, sample 1 at 1 s; frames before the
    # start clamp to frame 0.
    idx = window_indices(16, 16.0, 2)
    assert idx.shape == (2, 64)
    torch.testing.assert_close(idx[0], torch.arange(-55, 9).clamp(min=0))
    torch.testing.assert_close(idx[1], torch.arange(-47, 17).clamp(min=0, max=15))
    # A single image lasts 1/fps seconds; every frame index is 0.
    assert window_indices(1, 1.0, 2).eq(0).all()


def test_zero_fill():
    features = torch.ones(1, 2, 3, 4)
    out = zero_fill(features, onset_seconds=20)
    assert out.shape == (1, 2, 3, 200)
    assert out[..., 40:44].eq(1).all()
    assert out.sum() == features.sum()
    with pytest.raises(ValueError, match="do not fit"):
        zero_fill(features, onset_seconds=99)


@pytest.fixture(scope="module")
def tiny_video_features(tmp_path_factory):
    from transformers import VJEPA2Config, VJEPA2Model

    path = tmp_path_factory.mktemp("vjepa2")
    config = VJEPA2Config(
        hidden_size=32,
        num_hidden_layers=20,
        num_attention_heads=2,
        mlp_ratio=1.0,
        crop_size=256,
        frames_per_clip=64,
        pred_num_hidden_layers=1,
        pred_hidden_size=16,
        pred_num_attention_heads=2,
    )
    VJEPA2Model(config).save_pretrained(path)
    return lambda **kw: VideoFeatures(model_name=str(path), **kw)


def test_video_features_modes_and_gradients(tiny_video_features):
    still = torch.rand(1, 1, 3, 240, 320).expand(-1, 16, -1, -1, -1)
    video = still.clone().requires_grad_()
    sliding = tiny_video_features(fps=16.0)(video)
    static = tiny_video_features(fps=16.0, mode="static")(video)
    assert sliding.shape == static.shape == (1, 2, 32, 2)
    # A still video gives the same clip in every window, in both modes.
    torch.testing.assert_close(sliding, static)
    sliding.sum().backward()
    assert video.grad is not None
    assert video.grad.abs().sum() > 0
    with pytest.raises(TypeError, match="float RGB"):
        tiny_video_features(fps=16.0)(video.detach().mul(255).byte())


@pytest.mark.slow
@pytest.mark.parametrize("head", ["cortical", "subcortical"])
def test_released_heads_match_upstream(head):
    # Goldens: every 64th output of unmodified upstream af58661 (see
    # data/make_golden.py). Cortical uses TribeModel.from_pretrained. The
    # subcortical golden is built by the same recipe as load_brain_mapper (which
    # from_pretrained cannot load), so it checks the stack, not that recipe.
    model = load_brain_mapper(
        f"{CHECKPOINT_ROOT}/tribev2-{head}" if CHECKPOINT_ROOT else head
    )
    assert not model.training
    x = torch.sin(torch.arange(2 * 1408 * 200, dtype=torch.float32) * 0.37)
    with torch.no_grad():
        out = model({"video": x.reshape(1, 2, 1408, 200)})
    torch.testing.assert_close(out[:, ::64], torch.load(DATA / f"golden_{head}.pt"))


@pytest.mark.slow
def test_video_features_match_release():
    # Golden: the released neuralset 0.0.2 HuggingFaceVideo extractor with the
    # TRIBE v2 config on this clip, written losslessly (see data/make_golden.py).
    y, x = np.mgrid[:240, :320]
    frames = []
    for t in range(24):
        r = 127 + 120 * np.sin(x / 17 + t / 3)
        g = 127 + 120 * np.cos(y / 11 - t / 5)
        b = 127 + 120 * np.sin((x + y) / 23 + t / 2)
        frames.append(np.stack([r, g, b]))
    video = np.clip(np.round(np.array(frames)), 0, 255)
    video = torch.from_numpy(video).float()[None] / 255
    with torch.no_grad():
        out = VideoFeatures(fps=16.0)(video)[0]
    ref = torch.load(DATA / "golden_video_features.pt")
    assert out.shape == ref.shape == (2, 1408, 3)
    assert (out - ref).norm() / ref.norm() < 1e-4
