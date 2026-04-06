# RAVE Latent Analysis

Tools for extracting and analyzing acoustic features from audio files, with the goal of understanding what the latent dimensions of [RAVE](https://github.com/acids-icml/RAVE) (a real-time variational autoencoder for audio) encode.

## Project Structure

```
.
├── SAE.py                        # Sparse Autoencoder for latent feature analysis
├── audio_annotator.py            # AudioAnnotator class + GCS/local batch processing
├── crepe_inference_parallel.py   # Memory-efficient CREPE pitch extractor
├── create_dataset.py             # Generates synthetic test dataset
├── extract_features.py           # Runs feature extraction and saves results
├── evaluate_features.py          # Parameter sweep + accuracy evaluation
└── dataset/
    ├── metadata.json             # Ground-truth labels for the 10 test files
    └── extracted_features.json   # Extracted pitch, RMS, centroid values
```

## Components

### Feature Extraction

Three acoustic features are extracted from each audio file:

| Feature | Tool | Best Parameters |
|---|---|---|
| Pitch (Hz) | CREPE (`torchcrepe`) | `model=tiny`, `hop=512`, `periodicity_thr=0.5` |
| RMS energy | librosa | `frame_length=1024`, `hop=512` |
| Spectral centroid (Hz) | librosa | `n_fft=8192`, `hop=512` |

**`crepe_inference_parallel.py`** — a custom wrapper around `torchcrepe` that processes audio in CPU chunks and streams batches to GPU, avoiding out-of-memory errors on long audio files.

**`audio_annotator.py`** — high-level `AudioAnnotator` class that chains all three extractors. Supports both local folder processing and Google Cloud Storage bucket pipelines.

### Sparse Autoencoder (`SAE.py`)

A PyTorch `SparseAutoencoderBlock` that compresses extracted features into a sparse latent code. Uses L1 loss and KL divergence (against a target sparsity `rho=0.05`) to encourage interpretable, sparse activations — intended for analyzing RAVE's latent space.

### Test Dataset

10 synthetic WAV files (10 seconds, 44100 Hz) with controlled acoustic properties, used to validate and tune the feature extractors.

Pitch and harmonic content are varied independently to create a diverse range of RMS and spectral centroid values:

| File | Pitch | RMS | Centroid | Harmonic Profile |
|---|---|---|---|---|
| audio_01 | 100 Hz | 0.08 | 101 Hz | Pure tone |
| audio_02 | 100 Hz | 0.25 | 271 Hz | Rich harmonics |
| audio_03 | 200 Hz | 0.10 | 260 Hz | H1 + weak H2 |
| audio_04 | 200 Hz | 0.45 | 202 Hz | Pure tone |
| audio_05 | 440 Hz | 0.15 | 722 Hz | H1 + H2 + H3 |
| audio_06 | 440 Hz | 0.05 | 1480 Hz | Upper harmonics dominant |
| audio_07 | 880 Hz | 0.20 | 884 Hz | Pure tone |
| audio_08 | 150 Hz | 0.30 | 422 Hz | Very rich harmonics |
| audio_09 | 300 Hz | 0.12 | 636 Hz | H2 dominant |
| audio_10 | 600 Hz | 0.28 | 1879 Hz | Strong upper harmonics |

WAV files are not committed (regenerate with `create_dataset.py`). Ground-truth labels are in `dataset/metadata.json`.

## Quickstart

```bash
pip install torch torchcrepe librosa resampy soundfile

# Generate the 10 synthetic test files
python create_dataset.py

# Run feature extraction (saves to dataset/extracted_features.json)
python extract_features.py

# Evaluate extractor accuracy and tune parameters
python evaluate_features.py
```

## Evaluation Results

Feature extractor accuracy against ground-truth labels (mean % error):

| Feature | Error |
|---|---|
| Pitch (CREPE tiny) | ~0.5% |
| RMS (n_fft=1024) | ~0.09% |
| Spectral Centroid (n_fft=8192) | ~0.10% |

Spectral centroid is the most parameter-sensitive: frequency resolution improves monotonically with `n_fft`, especially for low-frequency signals.

## Batch Processing

`audio_annotator.py` supports processing audio at scale:

```python
from audio_annotator import AudioAnnotator

annotator = AudioAnnotator(sample_rate=44100, hop_length=512, win_length=8192, device="cuda")
pitch, rms, centroid = annotator.annotate("path/to/file.wav")
```

For GCS pipelines, configure bucket names in `process_gs_bucket()` and run `python audio_annotator.py`.
