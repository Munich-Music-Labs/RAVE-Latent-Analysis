#!/usr/bin/env python3
"""
Extracts pitch (CREPE), RMS, and spectral centroid (librosa) from all audio
files in dataset/ and saves the full frame-level time series to
dataset/extracted_features.json.

Best parameters from evaluate_features.py:
  Pitch    - CREPE tiny, hop=512, periodicity_threshold=0.5
  RMS      - librosa frame_length=1024, hop=512
  Centroid - librosa n_fft=8192, hop=512

All three use the same hop_length so their frame counts align.
"""

import json
import numpy as np
import librosa
import torch
from pathlib import Path

import crepe_inference_parallel

DATASET_DIR = Path("dataset")
SR = 44100

CREPE_HOP             = 512
CREPE_MODEL           = "tiny"
CREPE_PERIODICITY_THR = 0.5

RMS_FRAME_LEN  = 1024
CENTROID_N_FFT = 8192
LIBROSA_HOP    = 512   # same hop for RMS and centroid so frames align


def extract(audio_path: Path, device: str) -> dict:
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    y_t  = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)

    # --- Pitch via CREPE (one value per hop) ---
    pitch, periodicity = crepe_inference_parallel.maximally_parallel_predict(
        y_t, SR, CREPE_HOP,
        model=CREPE_MODEL,
        device=device,
        infer_batch_size=2048,
        return_periodicity=True,
    )
    pitch_frames       = pitch.squeeze(0).cpu().numpy()
    periodicity_frames = periodicity.squeeze(0).cpu().numpy()

    # --- RMS via librosa (one value per hop) ---
    rms_frames = librosa.feature.rms(
        y=y, frame_length=RMS_FRAME_LEN, hop_length=LIBROSA_HOP
    )[0]

    # --- Spectral centroid via librosa (one value per hop) ---
    centroid_frames = librosa.feature.spectral_centroid(
        y=y, sr=SR, n_fft=CENTROID_N_FFT, hop_length=LIBROSA_HOP
    )[0]

    # Trim all to the same length (CREPE resamples internally to 16 kHz,
    # so its frame count can differ slightly from librosa's)
    n = min(len(pitch_frames), len(rms_frames), len(centroid_frames))

    return {
        "n_frames":          n,
        "hop_length":        LIBROSA_HOP,
        "pitch_hz":          [round(float(v), 3) for v in pitch_frames[:n]],
        "periodicity":       [round(float(v), 4) for v in periodicity_frames[:n]],
        "rms":               [round(float(v), 6) for v in rms_frames[:n]],
        "spectral_centroid_hz": [round(float(v), 3) for v in centroid_frames[:n]],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Config : CREPE {CREPE_MODEL} hop={CREPE_HOP} | "
          f"RMS frame={RMS_FRAME_LEN} | centroid n_fft={CENTROID_N_FFT}\n")

    audio_files = sorted(DATASET_DIR.glob("audio_*.wav"))
    if not audio_files:
        print(f"No WAV files found in {DATASET_DIR}/. Run create_dataset.py first.")
        return

    results = {}
    print(f"{'File':<12} {'Frames':>8} {'Pitch mean':>12} {'RMS mean':>10} {'Centroid mean':>15}")
    print("-" * 62)

    for path in audio_files:
        name  = path.stem
        feats = extract(path, device)
        results[name] = feats

        p_mean = np.mean(feats["pitch_hz"])
        r_mean = np.mean(feats["rms"])
        c_mean = np.mean(feats["spectral_centroid_hz"])
        print(f"{name:<12} {feats['n_frames']:>8}  {p_mean:>11.2f}  {r_mean:>9.4f}  {c_mean:>14.2f}")

    out_path = DATASET_DIR / "extracted_features.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} files -> {out_path}")
    print(f"Each entry has {results[list(results.keys())[0]]['n_frames']} frames "
          f"(hop={LIBROSA_HOP} samples = {LIBROSA_HOP/SR*1000:.1f} ms)")


if __name__ == "__main__":
    main()
