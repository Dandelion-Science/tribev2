<div align="center">

# TRIBE v2

**A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

📄 [Paper](https://arxiv.org/abs/2605.04326)  |  ▶️ [Demo](https://aidemos.atmeta.com/tribev2/) | 🤗 [Weights](https://huggingface.co/facebook/tribev2)

</div>

TRIBE v2 is a deep multimodal brain encoding model that predicts fMRI brain responses to naturalistic stimuli (video, audio, text). It combines state-of-the-art text, audio and video models into a unified Transformer architecture that maps multimodal representations onto the cortical surface.

## Dandelion fork

This fork adds differentiable inference on a current stack without the original event
pipeline. It is tested on Python 3.14 / torch 2.13 / transformers 5.19 / neuraltrain 0.4 and
on the release stack (Python 3.12 / torch 2.6 / transformers 4.57 / neuraltrain 0.0.2).
Dandelion changes live on `dandelion-v*` branches (latest: `dandelion-v2`); `main`
tracks upstream. All changes are listed in [CHANGELOG.md](CHANGELOG.md).

```python
import torch
from tribev2 import VideoFeatures, load_brain_mapper, zero_fill

# The brain mapper (upstream FmriEncoderModel) maps V-JEPA 2 features to fMRI.
brain_mapper = load_brain_mapper("subcortical")  # or "cortical"; frozen, eval mode
features = VideoFeatures(fps=16.0)           # V-JEPA 2 giant; mode="sliding" | "static"
video = torch.rand(1, 32, 3, 480, 640)       # (B, T, 3, H, W) RGB in [0, 1]
feats = features(video)                      # (B, 2, 1408, 2 Hz samples)
bold = brain_mapper({"video": zero_fill(feats)})  # (B, n_outputs, 100), 1 Hz
# Without zero_fill: brain_mapper({"video": feats}, pool_outputs=False) -> 2 Hz, (B, n_outputs, T)
```

Or via `torch.hub`:
`torch.hub.load("Dandelion-Science/tribev2:dandelion-v2", "tribev2_brain_mapper", head="subcortical")`.

Tests: `pytest` (fast), `pytest -m slow` (golden comparisons with the release; downloads
~5 GB, or set `TRIBEV2_CHECKPOINT_ROOT`). Regenerate goldens with `tests/data/make_golden.py`.

The original pipeline below (`TribeModel`, plotting, training) needs the `demo` extra
(included by `plotting` and `training`), which keeps the release's pins (torch<2.7,
neuralset 0.0.2) and therefore Python 3.11–3.13.

## Quick start

Load a pretrained model from HuggingFace and predict brain responses to a video:

```python
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

df = model.get_events_dataframe(video_path="path/to/video.mp4")
preds, segments = model.predict(events=df)
print(preds.shape)  # (n_timesteps, n_vertices)
```

Predictions are for the "average" subject (see paper for details) and live on the **fsaverage5** cortical mesh (~20k vertices).
They are offset by 5 seconds in the past, in order to compensate for the hemodynamic lag.

You can also pass `text_path` or `audio_path` to `model.get_events_dataframe` — text is automatically converted to speech and transcribed to obtain word-level timings.

For a full walkthrough with brain visualizations, see the [Colab demo notebook](https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb).

## Installation

**Basic** (`load_brain_mapper`, `VideoFeatures`):
```bash
pip install -e .
```

**Original release pipeline** (`TribeModel`, as in the quick start):
```bash
pip install -e ".[demo]"
```

**With brain visualization**:
```bash
pip install -e ".[plotting]"
```

**With training dependencies** (PyTorch Lightning, W&B, etc.):
```bash
pip install -e ".[training]"
```

## Training a model from scratch

### 1. Set environment variables

Configure data/output paths and Slurm partition (or edit `tribev2/grids/defaults.py` directly):

```bash
export DATAPATH="/path/to/studies"
export SAVEPATH="/path/to/output"
```


### 2. Run training

**Local test run:**
```bash
python -m tribev2.grids.test_run
```

**Grid search on Slurm:**
```bash
python -m tribev2.grids.run_cortical
python -m tribev2.grids.run_subcortical
```

## Project structure

```
tribev2/
├── main.py              # Experiment pipeline: Data, TribeExperiment
├── model.py             # FmriEncoder: Transformer-based multimodal→fMRI model
├── pl_module.py         # PyTorch Lightning training module
├── demo_utils.py        # TribeModel and helpers for inference from text/audio/video
├── eventstransforms.py  # Custom event transforms (word extraction, chunking, …)
├── utils.py             # Multi-study loading, splitting, subject weighting
├── utils_fmri.py        # Surface projection (MNI / fsaverage) and ROI analysis
├── grids/
│   ├── defaults.py      # Full default experiment configuration
│   └── test_run.py      # Quick local test entry point
├── plotting/            # Brain visualization (PyVista & Nilearn backends)
└── studies/             # Dataset definitions (Algonauts2025, Lahner2024, …)
```

## Contributing to open science

If you use this software, please share your results with the broader research community using the following citation:

```bibtex
@article{dascoli2026foundation,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, St{\'e}phane and Rapin, J{\'e}r{\'e}my and Benchetrit, Yohann and Brooks, Teon and Begany, Katelyn and Raugel, Jos{\'e}phine and Banville, Hubert and King, Jean-R{\'e}mi},
  journal={arXiv preprint arXiv:2605.04326},
  year={2026}
}
```

## License

This project is licensed under CC-BY-NC-4.0. See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.
