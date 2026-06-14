# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependencies are managed with `uv`. Do not use `pip install` directly.

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_download_maestro_sample.py

# Run a single test by name
uv run pytest tests/test_download_maestro_sample.py::test_stratified_sample_proportions

# Feature extraction pipeline (run in order)
python create_dataset.py          # generate 10 synthetic WAV files in dataset/
python extract_features.py        # extract features → dataset/extracted_features.json
python evaluate_features.py       # parameter sweep + accuracy report

# Dataset downloaders (require GCS credentials)
python scripts/download_maestro_sample.py [--limit N]
python scripts/download_e_gmd.py [--limit N]
```

## Architecture

### Feature extraction pipeline

The main pipeline flows: `create_dataset.py` → `extract_features.py` → `evaluate_features.py`.

`extract_features.py` is the canonical extraction entry point. It calls `crepe_inference_parallel.maximally_parallel_predict` for pitch and librosa for RMS and spectral centroid, then trims all three time series to the same frame count before saving to `dataset/extracted_features.json`. The trim is necessary because CREPE internally resamples audio to 16 kHz before framing, so its output frame count differs slightly from librosa's (which operates at the original 44.1 kHz).

### CREPE wrapper (`crepe_inference_parallel.py`)

A custom wrapper around `torchcrepe` that avoids OOM on long audio by processing frames in CPU chunks (`infer_batch_size` controls chunk size) and streaming each batch to GPU for inference. The loaded model is cached as a function attribute on `maximally_parallel_predict` to avoid reloading between calls. Two decoders are provided: `maximally_parallel_predict` (argmax) and `maximally_parallel_predict_weighted` (weighted argmax over a local window).

### AudioAnnotator (`audio_annotator.py`)

Higher-level class that chains all three extractors. Intended for batch use: `process_local_folders()` for local WAV directories and `process_bucket()` / `process_gs_bucket()` for GCS pipelines. The GCS functions hardcode bucket names as placeholders (`"bucket_name"`, `"output_bucket_name"`) — update these before use.

### Sparse Autoencoder (`SAE.py`)

`SparseAutoencoderBlock` is a standalone PyTorch `nn.Module`. It returns `(reconstructed, sparse_code, l1_loss, kl_loss)` from `forward()` — the caller is responsible for combining losses and running the optimizer.

### Dataset download scripts (`scripts/`)

`download_maestro_sample.py` and `download_e_gmd.py` share the same pattern: they use a `_RangeFile` class that implements seekable HTTP Range requests, allowing `zipfile.ZipFile` to parse and extract individual entries from remote archives without downloading the full ZIP. Both scripts do stratified sampling (10% for MAESTRO, 8% for e-GMD) with `random_state=42` for determinism, upload a `manifest.csv` to GCS, and skip blobs that already exist.

GCP project: `ravelatents` | GCS bucket: `rave-latents`

### Tests

Tests cover only the download scripts (not the feature extraction pipeline). `tests/conftest.py` adds `scripts/` to `sys.path` so test files can import download script modules directly. All GCS and HTTP calls are mocked — no network access during testing.
