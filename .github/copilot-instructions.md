# Copilot Instructions for vox-box

Voice-cloning TTS project: a from-scratch PyTorch reimplementation of NVIDIA's Tacotron 2 (text → mel spectrogram), paired with NVIDIA's pretrained WaveGlow (mel → audio) fetched via `torch.hub`.

## Running

There is no build system, test suite, linter, or dependency manifest. Requires Python with `torch`, `numpy`, `scipy`, `librosa`, and `inflect`, plus a CUDA GPU (all code calls `.cuda()` unconditionally).

- **Train:** `python train.py` — configured entirely by `hparams` in `settings.py` (no CLI args). Fine-tunes from `starting_point`, clips gradients, validates after each epoch, and saves `checkpoint_last.pt`/`checkpoint_best.pt` (by validation loss) to `saved-models/`.
- **Inference:** `python execute.py` — interactive; prompts for text and output filename, writes a `.wav` to `results/`. Model choice is set by editing the `SETTINGS` block at the top of `execute.py` (`tacotron_sd_filepath = None` uses NVIDIA's pretrained hub model instead of a local checkpoint).

## Architecture

Data flow: `metadata.csv` (pipe-delimited `wav_path|transcript`) → `dataset.VocalData` (text → symbol IDs via `util.text_to_sequence`; wav → mel via `util.layers.TacotronSTFT`) → `dataset.CollateData` (sorts by length descending, zero-pads text/mel, builds gate targets) → `model.Tacotron2` → `loss.Tacotron2Loss` (MSE on mel + postnet-mel, BCE on gate).

- `settings.py` — single `hparams` dict consumed by every module (model, dataset, and training sections). All configuration changes go here, not on the command line.
- `model.py` — full Tacotron 2: Encoder, location-sensitive Attention, Prenet, Decoder, Postnet. `parse_batch()` moves batches to GPU; `forward()` is teacher-forced training; `inference()` is autoregressive generation. Layer structure mirrors `tacotron-architecture.txt` so checkpoints from NVIDIA's implementation (`tacotron2_nvidia.pt`, kept at repo root so it survives gitignoring `saved-models/`) load via `load_state_dict`.
- `util/` — text/audio processing vendored from keithito/tacotron and NVIDIA (cleaners, symbols, CMUdict/ARPAbet, STFT). Treat as reference code; avoid restyling it.
- `tts-dataset/` — git submodule (github.com/prattnj/tts-dataset) holding the training data; expected layout is `audio-dataset/{train,val}/metadata.csv` with wav paths relative to each metadata file.

## Conventions

- 2-space indentation throughout (non-standard for Python — match it).
- Checkpoints are dicts with a `'state_dict'` key: `torch.load(path)['state_dict']`. Saving/loading must preserve this format for compatibility with NVIDIA checkpoints.
- `saved-models/` and `results/` are runtime directories not tracked in git; `execute.py` creates `results/` on demand.
- Text may embed ARPAbet in curly braces (e.g. `{HH AW1 S}`); `text_to_sequence` handles this — don't strip braces in preprocessing.
- Audio is 22050 Hz, normalized by `max_wav_value` (32768.0); `VocalData` raises if a wav's sample rate doesn't match `hparams['sampling_rate']`.
