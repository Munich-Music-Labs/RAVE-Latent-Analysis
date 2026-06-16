# RAVE Latent Analysis

Implementation of [Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders](https://arxiv.org/abs/2510.23802) (Paek et al., NeurIPS 2025 MechInterp Workshop) on a subset of our GCS datasets.

The goal is to train sparse autoencoders on audio encoder latents and learn linear mappings from SAE features to acoustic concepts (pitch, amplitude, timbre).

## Current Status

Three datasets are in GCS bucket `rave-latents` (GCP project: `ravelatents`):

| Dataset | GCS path | Format | Files | Paper usage |
|---|---|---|---|---|
| e-GMD (drums) | `gs://rave-latents/e-gmd/` | WAV (ready) | 3,642 | 7.8 h in paper |
| MAESTRO (piano) | `gs://rave-latents/maestro-sample/` | WAV (ready) | 128 | 33 min in paper |
| CocoChorales (Bach) | `gs://rave-latents/cocochorales-sample/` | `.tar.bz2` (packed) | 2 archives | 11.2 h in paper |

The paper also uses DAMP-VSEP (pop/rock singing) and GuitarSet (guitar), which we don't have — our subset covers drums, piano, and chorales.

## Repository

```
SAE.py                        # Sparse Autoencoder (matches paper architecture)
audio_annotator.py            # Acoustic feature extraction (pitch, RMS, spectral centroid)
crepe_inference_parallel.py   # Memory-efficient CREPE pitch extractor
scripts/
  download_maestro_sample.py  # MAESTRO dataset downloader
  download_e_gmd.py           # e-GMD dataset downloader
  download_cocochorales.py    # CocoChorales dataset downloader
```

## Next Action Items — Audio Preprocessing

**1. Extract CocoChorales archives.**
The two `.tar.bz2` files in `cocochorales-sample/main_dataset/train/` contain the audio. Extract them and upload the WAV/FLAC files back to GCS so they're in the same ready state as e-GMD and MAESTRO.

**2. Run acoustic feature extraction on all three datasets.**
For each WAV file, compute frame-level features using `audio_annotator.py`:
- Pitch (Hz) via CREPE (`crepe_inference_parallel.py`)
- Amplitude via windowed RMS (librosa)
- Timbre via windowed spectral centroid (librosa)

Save frame-level arrays alongside each file (e.g. as `.npy` or into a manifest) in GCS.

**3. Aggregate to chunk level.**
Reduce frame-level arrays to a single scalar per chunk (e.g. median pitch, mean RMS, mean centroid). Chunk boundary can be the full file or a fixed window — to be decided based on encoder requirements later.

**4. Discretize into bins.**
Following the paper exactly:
- Pitch: 66 logarithmic bins aligned to MIDI note numbers
- Amplitude: 20 linearly-spaced bins over the RMS range of the dataset
- Timbre: 20 linearly-spaced bins over the spectral centroid range of the dataset

Write a final manifest to GCS: one row per chunk with `(gcs_path, pitch_bin, amplitude_bin, timbre_bin)`.
