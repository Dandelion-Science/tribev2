# Copyright (c) Dandelion Science. Adapted from facebookresearch/tribev2,
# licensed under CC BY-NC 4.0 (see LICENSE in the root directory of this tree).

"""Load released brain mappers without the neuralset/exca experiment stack.

TRIBE v2's brain mapper (upstream ``FmriEncoderModel``, which upstream calls the
"brain model") maps 2 Hz V-JEPA 2 video features, and text and audio features
unused here, to fMRI responses: a transformer over the projected features followed
by a linear average-subject head. It is the mapper of an encoding pipeline whose
backbone is :class:`tribev2.VideoFeatures`.
"""

from pathlib import Path

import torch
import yaml

from tribev2.model import FmriEncoder, FmriEncoderModel

# HuggingFace repo and the revision the tests' golden outputs were made from.
HEADS = {
    "cortical": ("facebook/tribev2", "f894e783020944dcd96e5568550afe2aa9743f9f"),
    "subcortical": (
        "facebook/tribev2-subcortical",
        "214a5d939573b09772fdb47415585201e94e2e15",
    ),
}

# Released feature dims (layer groups, dim) for text, audio and video. The
# subcortical checkpoint does not record them in ``model_build_args``.
FEATURE_DIMS = {"text": (2, 3072), "audio": (2, 1024), "video": (2, 1408)}
# Output size of each released head, to build one without its checkpoint.
N_OUTPUTS = {"cortical": 20484, "subcortical": 8808}


class _ConfigLoader(yaml.SafeLoader):
    """Read the released config's ``!!python/...`` tags (paths, tuples) as ``None``.

    None occur in the sections read here, and upstream's ``UnsafeLoader`` would
    execute arbitrary constructors from a downloaded file.
    """


_ConfigLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/", lambda loader, suffix, node: None
)


def load_brain_mapper(
    head: str = "cortical",
    *,
    checkpoint_name: str = "best.ckpt",
    device: str | torch.device = "cpu",
    pretrained: bool = True,
) -> FmriEncoderModel:
    """Load a released TRIBE v2 brain mapper in average-subject mode.

    Builds the model from the released ``config.yaml`` and checkpoint only,
    exactly as ``TribeModel.from_pretrained`` does with ``average_subjects=True``.
    The model is returned frozen and in eval mode: the released config sets
    ``modality_dropout=0.3``, so outputs are wrong in train mode.

    Parameters
    ----------
    head : str
        ``"cortical"`` (fsaverage5, 20484 vertices) or ``"subcortical"`` (8808
        voxels), downloaded at a pinned revision; or a HuggingFace Hub repo id
        or a local directory containing ``config.yaml`` and the checkpoint.
    checkpoint_name : str
        Checkpoint filename inside the directory or repo.
    device : str | torch.device
        Device to place the model on.
    pretrained : bool
        If ``False``, build a released head (``"cortical"`` or ``"subcortical"``)
        from its ``config.yaml`` alone, randomly initialized, for callers that
        load their own weights next; the checkpoint is not downloaded.

    Returns
    -------
    brain_mapper : FmriEncoderModel
        Call it with ``{"video": (B, 2, 1408, T)}`` 2 Hz features, ``T <= 1024``;
        missing modalities are zero-filled. The output ``(B, n_outputs, 100)``
        is ``AdaptiveAvgPool1d(100)`` over ``T``, so it is 1 Hz only for
        ``T = 200`` (a 100 s context, as from :func:`tribev2.zero_fill`). For any
        other ``T`` pass ``pool_outputs=False`` to get ``(B, n_outputs, T)`` at
        2 Hz.
    """
    if not pretrained and head not in N_OUTPUTS:
        raise ValueError(
            "pretrained=False builds a released head ('cortical' or "
            f"'subcortical'), got {head!r}"
        )
    if Path(head).is_dir():
        config_path = Path(head) / "config.yaml"
        ckpt_path = Path(head) / checkpoint_name
    else:
        from huggingface_hub import hf_hub_download

        repo_id, revision = HEADS.get(head, (head, None))
        config_path = hf_hub_download(repo_id, "config.yaml", revision=revision)
        if pretrained:
            ckpt_path = hf_hub_download(repo_id, checkpoint_name, revision=revision)

    with open(config_path) as f:
        config = yaml.load(f, Loader=_ConfigLoader)
    model_config = config["brain_model_config"]
    model_config.pop("name")
    model_config["subject_layers"].update(average_subjects=True, n_subjects=0)
    build_args = {
        "feature_dims": FEATURE_DIMS,
        "n_outputs": N_OUTPUTS.get(head),
        "n_output_timesteps": config["data"]["duration_trs"],
    }
    if not pretrained:
        model = FmriEncoder(**model_config).build(**build_args)
        # Sets the release's x_transformers options; the weights are random anyway.
        convert_released_encoder(model)
        return model.requires_grad_(False).eval().to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    state_dict = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
    build_args["n_outputs"] = state_dict["predictor.bias"].shape[-1]
    model = FmriEncoder(**model_config).build(
        **(ckpt.get("model_build_args") or build_args)
    )
    model.load_state_dict(state_dict, strict=True, assign=True)
    convert_released_encoder(model)
    return model.requires_grad_(False).eval().to(device)


def convert_released_encoder(model: FmriEncoderModel) -> None:
    """Make a released checkpoint, loaded on x_transformers>=2.31, match its release.

    The checkpoints were trained on x_transformers 1.27.20. Two later defaults load
    under the same parameter names but change the outputs:

    - 1.30.1 rotates rotary pairs ``(2i, 2i+1)`` instead of ``(i, i + rot_dim/2)``.
      Permuting the query and key projection rows the same way restores the
      release's attention scores, since ``q @ k`` is unchanged.
    - 1.31.3 adds 1 to every norm gain (``norm_add_unit_offset=True``); the
      checkpoints store the gains without that offset.

    Call it once, in place, right after loading released weights.
    """
    from x_transformers.x_transformers import Attention

    encoder = model.encoder
    for module in encoder.modules():
        if getattr(module, "unit_offset", False):
            module.unit_offset = False
    if encoder.rotary_pos_emb is None:
        return
    rot_dim = 2 * encoder.rotary_pos_emb.inv_freq.numel()
    half = torch.arange(rot_dim // 2)
    for attn in encoder.modules():
        if not isinstance(attn, Attention):
            continue
        dim_head = attn.to_q.weight.shape[0] // attn.heads
        perm = torch.arange(dim_head)
        perm[:rot_dim] = torch.stack([half, half + rot_dim // 2], dim=-1).flatten()
        for proj in (attn.to_q, attn.to_k):
            heads = proj.weight.shape[0] // dim_head
            rows = (torch.arange(heads)[:, None] * dim_head + perm).flatten()
            proj.weight = torch.nn.Parameter(
                proj.weight[rows], requires_grad=proj.weight.requires_grad
            )
