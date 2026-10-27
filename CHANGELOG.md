# Changelog

This is Dandelion Science's fork of
[facebookresearch/tribev2](https://github.com/facebookresearch/tribev2), used for
non-commercial research. As the CC BY-NC 4.0 licence requires, every change to the
original work is listed here. Dandelion changes live on `dandelion-v*` branches cut
from upstream; `main` tracks upstream. Upstream base: `af58661`.

## 0.1.0+dandelion.3 (2026-10-27, branch `dandelion-v3`)

Builds the models without downloading their released weights, for callers that load
their own weights next, such as fine-tuned models stored elsewhere. Tested on the
current stack (Python 3.14, torch 2.13, transformers 5.20); golden outputs are
unchanged.

### Added

- `VideoFeatures(..., pretrained=False)` builds V-JEPA 2 from its config alone,
  randomly initialized; only `config.json` is downloaded instead of the ~4 GB
  checkpoint.
- `load_brain_mapper(head, pretrained=False)` builds a released head
  (`"cortical"` or `"subcortical"`) from its `config.yaml` alone, with the
  release's x_transformers options set, and does not download the checkpoint.
  Given the released weights, it reproduces the goldens.

## 0.1.0+dandelion.2 (2026-10-20, branch `dandelion-v2`)

Requires x_transformers>=2.31 instead of pinning 1.27.20, tested on the same
current and release stacks as `dandelion-v1`. Golden outputs are unchanged.

### Added

- `tribev2.pretrained.convert_released_encoder(model)` makes a released checkpoint
  loaded on x_transformers>=2.31 match its release, which was trained on 1.27.20.
  Two later x_transformers defaults load under the same parameter names but change
  the outputs: 1.30.1 rotates interleaved rotary pairs, undone by permuting the
  query and key projection rows, and 1.31.3 adds 1 to the norm gains
  (`norm_add_unit_offset`), undone by turning the offset off. Without it, outputs
  differ from the goldens by up to 1.6.

### Changed

- `load_brain_mapper` and `TribeModel.from_pretrained` call
  `convert_released_encoder` after loading the weights.
- Dependencies: `x_transformers>=2.31` replaces `x_transformers==1.27.20`.

### Notes

- Fine-tuning from a released checkpoint (`TribeExperiment`, `main.py`) does not
  convert the weights.

## 0.1.0+dandelion.1 (2026-10-14, branch `dandelion-v1`)

Inference on a current stack, tested on Python 3.14, torch 2.13, transformers 5.19
and neuraltrain 0.4, and on the release stack (Python 3.12, torch 2.6, transformers
4.57, neuraltrain 0.0.2). It runs without the original event and caching pipeline,
with gradients from fMRI predictions back to pixels.

### Added

- `tribev2.load_brain_mapper(head)` loads a released brain mapper (upstream
  `FmriEncoderModel`, which maps V-JEPA 2 features to fMRI) from `config.yaml`
  and the checkpoint alone, frozen and in eval mode. `head` can be `"cortical"` or
  `"subcortical"` (HuggingFace snapshots pinned by revision), a Hub repo id, or a
  local directory. It also loads `facebook/tribev2-subcortical`, which
  `TribeModel.from_pretrained` cannot (that checkpoint has no `model_build_args`).
  It reads the config with a safe YAML loader instead of `yaml.UnsafeLoader`.
- `tribev2.VideoFeatures` is a differentiable torch reimplementation of the released
  V-JEPA 2 video features (neuralset 0.0.2 `HuggingFaceVideo` with the TRIBE v2
  config), in two modes:
  - `"sliding"`, as released: one 4 s look-back window per 2 Hz sample.
  - `"static"`: the whole stimulus encoded as one clip and repeated.

  It matches the released extractor to 7.7e-8 relative L2. With gradients enabled,
  sliding windows are recomputed in backward (activation checkpointing), so memory
  doesn't grow with video length.
- `tribev2.zero_fill` places features at an onset inside a zero-filled context
  (100 s by default), as the released pipeline sees a short stimulus.
- `hubconf.py` adds the `torch.hub` entry points `tribev2_brain_mapper` and
  `tribev2_video_features`.
- `tests/` adds unit tests, plus slow tests (`pytest -m slow`) against golden
  outputs of unmodified upstream `af58661`. `tests/data/make_golden.py` regenerates
  the goldens.

### Changed

- `import tribev2` no longer imports the event stack. `TribeModel` loads on first
  access, with a hint to install the `demo` extra if that fails, and `tribev2.model`
  imports neuralset only for type checking.
- `FmriEncoderModel.forward` also accepts a plain `{modality: tensor}` dict. It
  raises an error if none of the dict's keys is a known modality.
- Dependencies: the core install no longer pins `torch<2.7`, `numpy==2.2.6`,
  `neuralset`, `neuraltrain` or `exca`. The original release pipeline (`TribeModel`,
  event extraction) moves to the `demo` extra with its original pins, plus
  `transformers<5`; it fails on neuralset 0.2.3 and 0.4.0. The `plotting` and
  `training` extras include `demo`.

### Notes

- transformers 5 changed `VJEPA2Model`'s `hidden_states[-1]` from the final
  layer-normed output to the pre-norm block output. TRIBE was trained on the
  former, so `VideoFeatures` uses `last_hidden_state` for the last layer. Reading
  `hidden_states[-1]` on transformers 5 shifts the second layer group by about 10%.
- The brain mapper always pools its output to 100 samples. That is 1 Hz only for a
  100 s (200-sample) input such as `zero_fill` produces; otherwise call it with
  `pool_outputs=False`.
