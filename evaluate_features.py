#!/usr/bin/env python3
"""
Evaluates CREPE + librosa feature extractors against ground-truth labels.

Strategy
--------
1. Extract pitch with CREPE (tiny vs full model, two periodicity thresholds).
2. Extract RMS and spectral centroid with librosa across a sweep of n_fft values
   (n_fft is the dominant parameter - it sets frequency resolution for centroid
   and window length for RMS).
3. Report per-file % error and mean error per config.
4. Print a final summary identifying the best parameter set for each feature.
"""

import json
import numpy as np
import librosa
import torch
from pathlib import Path

import crepe_inference_parallel

DATASET_DIR = Path("dataset")
SR = 44100
FEATURES = ["pitch_hz", "rms", "spectral_centroid_hz"]
FEATURE_LABELS = {
    "pitch_hz":             "Pitch (Hz)",
    "rms":                  "RMS",
    "spectral_centroid_hz": "Centroid (Hz)",
}


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_pitch(y: np.ndarray, hop_length: int, model: str,
                  periodicity_thr: float, device: str) -> float:
    y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
    pitch, periodicity = crepe_inference_parallel.maximally_parallel_predict(
        y_t, SR, hop_length,
        model=model,
        device=device,
        infer_batch_size=2048,
        return_periodicity=True,
    )
    pitch_np = pitch.squeeze(0).cpu().numpy()
    per_np   = periodicity.squeeze(0).cpu().numpy()
    voiced   = per_np > periodicity_thr
    return float(np.median(pitch_np[voiced]) if voiced.any() else np.median(pitch_np))


def extract_librosa(y: np.ndarray, n_fft: int, hop_length: int) -> dict:
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=n_fft, hop_length=hop_length)[0]
    return {
        "rms":                  float(np.mean(rms)),
        "spectral_centroid_hz": float(np.mean(cent)),
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def pct_err(truth: float, pred: float) -> float:
    return abs(truth - pred) / truth * 100.0 if truth != 0 else float("inf")


def header(title: str):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)


# ─── Evaluation runners ───────────────────────────────────────────────────────

def eval_pitch(metadata: dict, audio_files: list, device: str):
    """Sweep CREPE model size × periodicity threshold."""
    configs = [
        ("tiny", 0.30),
        ("tiny", 0.50),
        ("tiny", 0.70),
        ("full", 0.50),
    ]
    hop_length = 512  # ~11 ms - reasonable for pitch tracking

    header("CREPE Pitch  (hop=512 samples ~ 11 ms)")
    print(f"  {'File':<12}", end="")
    for model, thr in configs:
        col = f"{model}/thr={thr}"
        print(f"  {col:>14}", end="")
    print()
    print(f"  {'-'*68}")

    summary = {c: [] for c in configs}

    for path in audio_files:
        name = path.stem
        y, _ = librosa.load(str(path), sr=SR, mono=True)
        truth_pitch = metadata[name]["pitch_hz"]
        print(f"  {name:<12}", end="", flush=True)

        for cfg in configs:
            model, thr = cfg
            pred = extract_pitch(y, hop_length, model, thr, device)
            err  = pct_err(truth_pitch, pred)
            summary[cfg].append(err)
            flag = " !" if err > 10 else "  "
            print(f"  {err:>12.1f}%{flag}", end="", flush=True)
        print()

    print(f"\n  {'MEAN':<12}", end="")
    for cfg in configs:
        print(f"  {np.mean(summary[cfg]):>13.1f}%", end="")
    print()

    best = min(configs, key=lambda c: np.mean(summary[c]))
    print(f"\n  Best CREPE config: model={best[0]}, periodicity_thr={best[1]}"
          f"  ({np.mean(summary[best]):.1f}% mean error)")
    return best, np.mean(summary[best])


def eval_librosa(metadata: dict, audio_files: list):
    """Sweep n_fft for RMS and spectral centroid."""
    n_fft_values  = [512, 1024, 2048, 4096, 8192]
    hop_length    = 512

    header(f"Librosa RMS + Centroid  (hop={hop_length} samples, n_fft sweep)")
    col_w = 16

    # RMS table
    print(f"\n  --- RMS ---")
    print(f"  {'File':<12}", end="")
    for n in n_fft_values:
        print(f"  {('n_fft='+str(n)):>{col_w}}", end="")
    print()
    print(f"  {'-'*80}")

    rms_errors   = {n: [] for n in n_fft_values}
    cent_errors  = {n: [] for n in n_fft_values}
    lib_cache    = {}  # (name, n_fft) -> features

    for path in audio_files:
        name  = path.stem
        y, _  = librosa.load(str(path), sr=SR, mono=True)
        truth = metadata[name]
        print(f"  {name:<12}", end="")
        for n in n_fft_values:
            feats = extract_librosa(y, n_fft=n, hop_length=hop_length)
            lib_cache[(name, n)] = feats
            err = pct_err(truth["rms"], feats["rms"])
            rms_errors[n].append(err)
            print(f"  {err:{col_w}.2f}%", end="")
        print()

    print(f"\n  {'MEAN':<12}", end="")
    for n in n_fft_values:
        print(f"  {np.mean(rms_errors[n]):{col_w}.2f}%", end="")
    print()

    # Centroid table
    print(f"\n  --- Spectral Centroid ---")
    print(f"  {'File':<12}", end="")
    for n in n_fft_values:
        print(f"  {('n_fft='+str(n)):>{col_w}}", end="")
    print()
    print(f"  {'-'*80}")

    for path in audio_files:
        name  = path.stem
        truth = metadata[name]
        print(f"  {name:<12}", end="")
        for n in n_fft_values:
            feats = lib_cache[(name, n)]
            err = pct_err(truth["spectral_centroid_hz"], feats["spectral_centroid_hz"])
            cent_errors[n].append(err)
            print(f"  {err:{col_w}.2f}%", end="")
        print()

    print(f"\n  {'MEAN':<12}", end="")
    for n in n_fft_values:
        print(f"  {np.mean(cent_errors[n]):{col_w}.2f}%", end="")
    print()

    best_rms  = min(n_fft_values, key=lambda n: np.mean(rms_errors[n]))
    best_cent = min(n_fft_values, key=lambda n: np.mean(cent_errors[n]))
    print(f"\n  Best n_fft for RMS:      {best_rms}  ({np.mean(rms_errors[best_rms]):.2f}% mean error)")
    print(f"  Best n_fft for Centroid: {best_cent}  ({np.mean(cent_errors[best_cent]):.2f}% mean error)")
    return best_rms, best_cent


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(DATASET_DIR / "metadata.json") as f:
        metadata = json.load(f)

    audio_files = sorted(DATASET_DIR.glob("audio_*.wav"))
    print(f"Files : {len(audio_files)}  |  SR: {SR} Hz\n")

    # Print ground-truth table
    header("Ground-Truth Labels")
    print(f"  {'File':<12} {'Pitch (Hz)':>12} {'RMS':>10} {'Centroid (Hz)':>15}")
    print(f"  {'-'*52}")
    for name, v in sorted(metadata.items()):
        print(f"  {name:<12} {v['pitch_hz']:>12.1f} {v['rms']:>10.4f} {v['spectral_centroid_hz']:>15.1f}")

    # Evaluations
    best_crepe_cfg, best_pitch_err = eval_pitch(metadata, audio_files, device)
    best_rms_nfft, best_cent_nfft = eval_librosa(metadata, audio_files)

    # ── Final recommendation ──────────────────────────────────────────────────
    header("RECOMMENDED PARAMETERS")
    print(f"  Pitch   - CREPE model='{best_crepe_cfg[0]}', "
          f"periodicity_threshold={best_crepe_cfg[1]}, hop_length=512")
    print(f"  RMS     - librosa frame_length={best_rms_nfft}, hop_length=512")
    print(f"  Centroid- librosa n_fft={best_cent_nfft}, hop_length=512")
    print()
    print(f"  Mean errors:  pitch={best_pitch_err:.1f}%  |  "
          f"(see tables above for RMS/centroid per n_fft)")


if __name__ == "__main__":
    main()
