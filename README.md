# vox-box
An AI that can be trained to mimic anyone's voice.

A from-scratch PyTorch reimplementation of NVIDIA's Tacotron 2 (text → mel spectrogram), paired with
NVIDIA's pretrained WaveGlow (mel → audio) to fine-tune a voice-cloning text-to-speech model on your own
recordings.

## Requirements

* Python 3.x
* An NVIDIA GPU with CUDA (all code calls `.cuda()` unconditionally — there is no CPU fallback)
* Dependencies from `requirements.txt`:
  ```
  pip install -r requirements.txt
  ```

## Project layout

* `settings.py` — single `hparams` dict consumed by every module (model, dataset, and training). All
  configuration changes (batch size, learning rate, epochs, dataset path, starting checkpoint, etc.) go
  here — there are no command-line arguments.
* `tacotron2_nvidia.pt` — NVIDIA's pretrained Tacotron 2 checkpoint, used as the starting point for
  fine-tuning (kept at the repo root, not `.gitignore`'d, so it always survives).
* `saved-models/` — where your own fine-tuned checkpoints get saved during/after training. This
  directory is `.gitignore`'d — checkpoints are not committed.
* `results/` — generated `.wav` output from `execute.py` (created automatically, also `.gitignore`'d).
* `tts-dataset/` — git submodule ([prattnj/tts-dataset](https://github.com/prattnj/tts-dataset))
  containing the tools used to record and prepare your own voice dataset. See its own README for how to
  record samples and build `audio-dataset/{train,val}`.

## Setup

1. Clone the repo, including the submodule:
   ```
   git clone --recurse-submodules <repo-url>
   ```
   (or, if already cloned: `git submodule update --init --recursive`)
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Follow the [tts-dataset README](tts-dataset/README.md) to record your voice and produce
   `tts-dataset/audio-dataset/train/metadata.csv` and `tts-dataset/audio-dataset/val/metadata.csv`
   (plus their accompanying `.wav` files).
4. Confirm `settings.py`'s `dataset_dir` points at `tts-dataset/audio-dataset` and `starting_point`
   points at `tacotron2_nvidia.pt` (both are the defaults).

## Training

```
python train.py
```

There are no CLI flags — everything is controlled via `hparams` in `settings.py`. Fine-tuned checkpoints
are written to `saved-models/` as dicts with a `'state_dict'` key (matching the format NVIDIA's
checkpoint uses, so they load the same way).

## Inference

```
python execute.py
```

This is interactive: it prompts for text to synthesize and an output filename, then writes a `.wav` to
`results/`. Before running, edit the `SETTINGS` block at the top of `execute.py`:

* `tacotron_sd_filepath` — path to a local checkpoint (e.g. one of your fine-tuned models in
  `saved-models/`), or `None` to use NVIDIA's pretrained model straight from `torch.hub` instead.

WaveGlow (mel → audio) is always fetched via `torch.hub` from NVIDIA's repo, regardless of which
Tacotron 2 checkpoint is used.
