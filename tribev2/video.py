# Copyright (c) Dandelion Science. Adapted from facebookresearch/tribev2,
# licensed under CC BY-NC 4.0 (see LICENSE in the root directory of this tree).

"""Differentiable TRIBE v2 video features (V-JEPA 2 giant at 2 Hz).

Reimplements, in torch, the released ``HuggingFaceVideo`` extractor of
neuralset 0.0.2 with the TRIBE v2 config (``clip_duration=4``, 64 frames,
``layers=[0.5, 0.75, 1.0]``, ``cache_n_layers=20``, ``group_mean`` layer and
``mean`` token aggregation), so gradients can flow from fMRI predictions back
to pixels.
"""

import typing as tp
from functools import partial

import torch
import torchvision.transforms.v2.functional as TF
from torch import nn
from torch.utils.checkpoint import checkpoint

FREQUENCY = 2.0  # feature samples per second
CLIP_DURATION = 4.0  # seconds of look-back per feature sample
NUM_FRAMES = 64  # frames per V-JEPA 2 clip
N_LAYERS = 20  # hidden states kept, evenly spaced over the network
LAYER_GROUPS = ((9, 14), (14, 20))  # group_mean over layers [0.5, 0.75, 1.0]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class VideoFeatures(nn.Module):
    """Map videos to the 2 Hz V-JEPA 2 features the TRIBE v2 brain mapper consumes.

    Parameters
    ----------
    fps : float
        Frame rate of the input videos. A single image lasts ``1 / fps``
        seconds, so pass ``fps=1`` for a one-second image.
    mode : ``"sliding"`` | ``"static"``
        ``"sliding"`` (released behaviour): feature ``i`` encodes the 4 s of
        video ending at its sample time, frames before the start clamped to
        the first frame. ``"static"``: the whole stimulus is encoded once as a
        single 64-frame clip and that feature is repeated at every sample.
    model_name : str
        HuggingFace id or local path of the V-JEPA 2 checkpoint.
    dtype : torch.dtype
        V-JEPA 2 compute dtype. The release used float32; bfloat16 is faster
        and uses less memory, at a small numerical cost.
    """

    def __init__(
        self,
        fps: float,
        *,
        mode: tp.Literal["sliding", "static"] = "sliding",
        model_name: str = "facebook/vjepa2-vitg-fpc64-256",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        from transformers import VJEPA2Model

        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if mode not in ("sliding", "static"):
            raise ValueError(f"mode must be 'sliding' or 'static', got {mode!r}")
        self.fps = fps
        self.mode = mode
        self.model = VJEPA2Model.from_pretrained(model_name, torch_dtype=dtype)
        self.model.requires_grad_(False).eval()
        self.register_buffer(
            "mean", torch.tensor(MEAN)[:, None, None], persistent=False
        )
        self.register_buffer("std", torch.tensor(STD)[:, None, None], persistent=False)

    def n_samples(self, n_frames: int) -> int:
        """Return the number of 2 Hz feature samples for ``n_frames`` frames."""
        return max(1, round(n_frames / self.fps * FREQUENCY))

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T, 3, H, W)`` float RGB in ``[0, 1]`` to features.

        Parameters
        ----------
        video : torch.Tensor
            Float videos of shape ``(B, T, 3, H, W)`` with values in ``[0, 1]``.

        Returns
        -------
        features : torch.Tensor
            Features of shape ``(B, 2, 1408, n_samples)`` at 2 Hz.
        """
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(
                f"video must have shape (B, T, 3, H, W), got {tuple(video.shape)}"
            )
        if not video.is_floating_point():
            raise TypeError(f"video must be float RGB in [0, 1], got {video.dtype}")
        n_frames = video.shape[1]
        n_samples = self.n_samples(n_frames)
        frames = self.preprocess(video)
        if self.mode == "static":
            index = torch.arange(NUM_FRAMES) * n_frames // NUM_FRAMES
            features = self.encode(frames[:, index])
            return features[..., None].expand(-1, -1, -1, n_samples)
        # ponytail: one V-JEPA pass per sample (2 per second of video); batch the
        # windows if throughput matters.
        windows = window_indices(n_frames, self.fps, n_samples)
        # Recompute each window in backward instead of holding every window's
        # ViT-g graph (~30 GB each in float32). HF gradient checkpointing is
        # inactive in eval mode, so it cannot do this.
        encode = self.encode
        if torch.is_grad_enabled():
            encode = partial(checkpoint, self.encode, use_reentrant=False)
        return torch.stack([encode(frames[:, w]) for w in windows], dim=-1)

    def preprocess(self, video: torch.Tensor) -> torch.Tensor:
        """Resize, center-crop and normalize frames as the release did.

        The short side is resized to 292 and the center 256 x 256 crop kept, as
        the released ``VJEPA2VideoProcessor`` (transformers 4.x) does.
        """
        batch, n_frames = video.shape[:2]
        frames = video.flatten(0, 1).float() * 255
        frames = TF.resize(
            frames, 292, interpolation=TF.InterpolationMode.BILINEAR, antialias=True
        )
        # The release resized uint8 frames: quantize, with a straight-through gradient.
        frames = frames + (frames.round().clamp(0, 255) - frames).detach()
        frames = TF.center_crop(frames, [256, 256]) / 255
        frames = (frames - self.mean) / self.std
        return frames.unflatten(0, (batch, n_frames))

    def encode(self, clips: torch.Tensor) -> torch.Tensor:
        """Map ``(B, 64, 3, 256, 256)`` clips to ``(B, 2, 1408)``."""
        output = self.model(
            pixel_values_videos=clips.to(self.model.dtype),
            output_hidden_states=True,
            skip_predictor=True,
        )
        # TRIBE was trained on transformers 4.x, where the last hidden state is
        # the final layer-normed output; in 5.x hidden_states[-1] is pre-norm.
        states = (*output.hidden_states[:-1], output.last_hidden_state)
        keep = torch.linspace(0, len(states) - 1, N_LAYERS).round().long().tolist()
        layers = torch.stack([states[i].mean(1) for i in keep], dim=1).float()
        return torch.stack([layers[:, a:b].mean(1) for a, b in LAYER_GROUPS], dim=1)


def window_indices(n_frames: int, fps: float, n_samples: int) -> torch.Tensor:
    """Return frame indices ``(n_samples, 64)`` of each sliding 4 s look-back window.

    Sample ``i`` ends at ``duration * (i + 1) / n_samples``. Frame times are
    clamped at 0 and looked up as ``int(fps * t + 1e-5)`` (moviepy), clamped
    to the last frame, as in the released extractor.
    """
    duration = n_frames / fps
    offsets = torch.arange(NUM_FRAMES - 1, -1, -1, dtype=torch.float64)
    offsets = offsets / NUM_FRAMES * CLIP_DURATION  # same float64 order as release
    ends = torch.linspace(0, duration, n_samples + 1, dtype=torch.float64)[1:]
    times = (ends[:, None] - offsets).clamp(min=0)
    return (times * fps + 1e-5).long().clamp(max=n_frames - 1)


def zero_fill(
    features: torch.Tensor,
    *,
    onset_seconds: float = 20.0,
    context_seconds: float = 100.0,
) -> torch.Tensor:
    """Place 2 Hz features at ``onset_seconds`` in a zero-filled context.

    Zero-filling matches the released pipeline, which predicts a short stimulus
    in a 100 s segment with no features around it; the release starts that
    segment at the stimulus (onset 0). The 20 s default leaves pre-stimulus
    context for the bidirectional encoder. With the default 100 s context the
    brain mapper's output sample ``k`` is the response at ``k`` seconds, with the
    hemodynamic lag already learned in.

    Parameters
    ----------
    features : torch.Tensor
        Features of shape ``(..., n_samples)`` at 2 Hz.
    onset_seconds : float
        Where the stimulus starts in the context.
    context_seconds : float
        Context length.

    Returns
    -------
    context : torch.Tensor
        Features of shape ``(..., 2 * context_seconds)``, zero outside the
        stimulus.
    """
    start = round(onset_seconds * FREQUENCY)
    total = round(context_seconds * FREQUENCY)
    if start < 0 or start + features.shape[-1] > total:
        raise ValueError(
            f"{features.shape[-1]} samples at onset {onset_seconds}s do not fit in "
            f"a {context_seconds}s context"
        )
    out = features.new_zeros(*features.shape[:-1], total)
    out[..., start : start + features.shape[-1]] = features
    return out
