# Copyright (c) Dandelion Science. Adapted from facebookresearch/tribev2,
# licensed under CC BY-NC 4.0 (see LICENSE in the root directory of this tree).

"""Regenerate the golden tensors in this directory from the unmodified release.

Run with the *original* upstream code and its environment, not this fork, on CPU::

    uv venv -p 3.12 release && . release/bin/activate
    uv pip install "tribev2 @ git+https://github.com/facebookresearch/tribev2@af58661" \
        "transformers<5" "torch==2.6.*" "torchvision==0.21.*" \
        --extra-index-url https://download.pytorch.org/whl/cpu
    CUDA_VISIBLE_DEVICES= python -I tests/data/make_golden.py tests/data

``python -I`` keeps this checkout off ``sys.path`` so ``tribev2`` is the release.
Pass a second argument to read checkpoints from local ``tribev2-{cortical,
subcortical}`` directories (copies of the pinned HuggingFace snapshots) instead
of downloading them. The goldens were made with Python 3.12, torch 2.6.0,
transformers 4.57.6 and neuralset/neuraltrain 0.0.2.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from huggingface_hub import snapshot_download

# Same snapshots as tribev2.pretrained.HEADS (this script runs without the fork).
REPOS = {
    "cortical": ("facebook/tribev2", "f894e783020944dcd96e5568550afe2aa9743f9f"),
    "subcortical": (
        "facebook/tribev2-subcortical",
        "214a5d939573b09772fdb47415585201e94e2e15",
    ),
}


def checkpoint_dir(head: str, root: Path | None) -> Path:
    """Return the local directory holding ``head``'s config and checkpoint."""
    if root:
        return root / f"tribev2-{head}"
    repo_id, revision = REPOS[head]
    return Path(snapshot_download(repo_id, revision=revision))


def mapper_input() -> torch.Tensor:
    """Return the deterministic ``(1, 2, 1408, 200)`` video features."""
    x = torch.sin(torch.arange(2 * 1408 * 200, dtype=torch.float32) * 0.37)
    return x.reshape(1, 2, 1408, 200)


def brain_mapper_goldens(out: Path, root: Path | None) -> None:
    """Save every 64th output of each released head on :func:`mapper_input`."""
    from tribev2 import TribeModel
    from tribev2.model import FmriEncoder

    for head in REPOS:
        path = checkpoint_dir(head, root)
        if head == "cortical":
            model = TribeModel.from_pretrained(path, device="cpu")._model
        else:
            # TribeModel.from_pretrained cannot load this checkpoint (it has no
            # model_build_args), so build the same model by hand, as
            # TribeExperiment(average_subjects=True) configures it.
            text = (path / "config.yaml").read_text()
            config = yaml.load(text, Loader=yaml.UnsafeLoader)
            model_config = config["brain_model_config"]
            model_config.pop("name")
            model_config["subject_layers"].update(average_subjects=True, n_subjects=0)
            ckpt = torch.load(path / "best.ckpt", map_location="cpu", mmap=True)
            state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
            model = FmriEncoder(**model_config).build(
                feature_dims={
                    "text": (2, 3072),
                    "audio": (2, 1024),
                    "video": (2, 1408),
                },
                n_outputs=state["predictor.bias"].shape[-1],
                n_output_timesteps=config["data"]["duration_trs"],
            )
            model.load_state_dict(state, strict=True, assign=True)
            model.eval()
        assert not model.training
        with torch.no_grad():
            y = model(SimpleNamespace(data={"video": mapper_input()}))
        torch.save(y[:, ::64].clone(), out / f"golden_{head}.pt")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    torch.set_num_threads(16)
    brain_mapper_goldens(out, root)
