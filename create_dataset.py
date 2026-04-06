#!/usr/bin/env python3
"""
Creates 10 synthetic WAV files with controlled pitch, RMS, and spectral centroid.
Each file is 10 seconds at 44100 Hz. Harmonic content lets us vary centroid
independently of pitch.

Ground-truth labels are saved to dataset/metadata.json.
"""

import numpy as np
import soundfile as sf
import librosa
import json
from pathlib import Path

SR = 44100
DURATION = 10  # seconds
OUT_DIR = Path("dataset")

# ─── Signal designs ──────────────────────────────────────────────────────────
# (filename, pitch_hz, rms_target, [(harmonic_number, relative_amplitude), ...])
#
# Spectral centroid ≈ sum(f_n * A_n) / sum(A_n)  where f_n = pitch * n
# By weighting upper harmonics more heavily we raise the centroid
# without changing pitch.
DESIGNS = [
    # pure tones — centroid == pitch
    ("audio_01",  100, 0.08, [(1, 1.0)]),
    ("audio_04",  200, 0.45, [(1, 1.0)]),
    ("audio_07",  880, 0.20, [(1, 1.0)]),

    # moderate harmonic content
    ("audio_03",  200, 0.10, [(1, 1.0), (2, 0.4)]),
    ("audio_05",  440, 0.15, [(1, 1.0), (2, 0.6), (3, 0.3)]),

    # rich harmonics — centroid >> pitch
    ("audio_02",  100, 0.25, [(1, 1.0), (2, 0.8), (3, 0.7), (4, 0.5), (5, 0.3), (6, 0.2)]),
    ("audio_08",  150, 0.30, [(1, 1.0), (2, 0.9), (3, 0.7), (4, 0.5), (5, 0.3), (6, 0.2), (7, 0.1)]),

    # dominant upper harmonics — large centroid/pitch ratio
    ("audio_06",  440, 0.05, [(1, 0.2), (2, 0.5), (3, 1.0), (4, 0.8), (5, 0.6)]),
    ("audio_09",  300, 0.12, [(1, 0.3), (2, 1.0), (3, 0.5)]),
    ("audio_10",  600, 0.28, [(1, 1.0), (2, 0.4), (3, 0.7), (4, 0.9), (5, 1.0)]),
]


# ─── Synthesis ───────────────────────────────────────────────────────────────

def synthesize(pitch_hz: float, rms_target: float,
               harmonics: list, sr: int = SR, duration: float = DURATION) -> np.ndarray:
    n = int(duration * sr)
    t = np.linspace(0, duration, n, endpoint=False)

    y = np.zeros(n, dtype=np.float64)
    for h, amp in harmonics:
        freq = pitch_hz * h
        if freq < sr / 2:  # below Nyquist
            y += amp * np.sin(2.0 * np.pi * freq * t)

    # Scale to target RMS
    y *= rms_target / np.sqrt(np.mean(y ** 2))
    return y.astype(np.float32)


# ─── Ground-truth computation ─────────────────────────────────────────────────

def compute_truth(y: np.ndarray, pitch_hz: float, sr: int = SR) -> dict:
    """
    Pitch   — the fundamental frequency we designed (exact by construction).
    RMS     — sqrt(mean(y²)) over the full signal; exact, no frame effects.
    Centroid— librosa with a very large FFT (8192) for high freq resolution.
    """
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))

    cent = librosa.feature.spectral_centroid(
        y=y.astype(np.float32), sr=sr,
        n_fft=8192, hop_length=4096
    )[0]

    return {
        "pitch_hz": float(pitch_hz),
        "rms": round(rms, 6),
        "spectral_centroid_hz": round(float(np.mean(cent)), 3),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(exist_ok=True)
    metadata = {}

    print(f"{'File':<12} {'Pitch':>8} {'RMS':>8} {'Centroid':>12}  Harmonics")
    print("-" * 68)

    for name, pitch_hz, rms_target, harmonics in DESIGNS:
        y = synthesize(pitch_hz, rms_target, harmonics)
        sf.write(str(OUT_DIR / f"{name}.wav"), y, SR, subtype="FLOAT")

        truth = compute_truth(y, pitch_hz)
        metadata[name] = truth

        h_str = "+".join(f"H{h}({a:.1f})" for h, a in harmonics)
        print(
            f"{name:<12} {truth['pitch_hz']:>7.0f} Hz "
            f"{truth['rms']:>8.4f}  "
            f"{truth['spectral_centroid_hz']:>9.1f} Hz  {h_str}"
        )

    meta_path = OUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{len(metadata)} files + metadata saved -> {meta_path}")


if __name__ == "__main__":
    main()
