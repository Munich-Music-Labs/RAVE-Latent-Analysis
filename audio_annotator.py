import datetime
from pathlib import Path

import torch
import librosa
import torchcrepe
import tempfile
import os
import datetime

import alternate_crepe
import crepe_inference_parallel

from google.cloud import storage

class AudioAnnotator:
    def __init__(self, sample_rate=44100, hop_length=1, win_length=1024, device="cpu"):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.device = device

    def load_audio(self, path):
        y, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        return y

    def extract_pitch(self, y):
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

        pitch, periodicity = crepe_inference_parallel.maximally_parallel_predict(y_tensor,
            self.sample_rate,
            self.hop_length,
            model='tiny',
            device=self.device,
            infer_batch_size=8192*2,
            return_periodicity=True)

        # pitch, periodicity = alternate_crepe.predict(
        #     y_tensor,
        #     self.sample_rate,
        #     self.hop_length,
        #     model='tiny',
        #     batch_size=32,
        #     device=self.device,
        #     decoder=torchcrepe.decode.weighted_argmax,
        #     return_periodicity=True,
        # )

        return pitch.squeeze(0).cpu()

    def extract_rms(self, y):
        rms = librosa.feature.rms(
            y=y,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )[0]
        return torch.tensor(rms, dtype=torch.float32)

    def extract_spectral_centroid(self, y):
        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=self.sample_rate,
            n_fft=self.win_length,
            hop_length=self.hop_length
        )[0]
        return torch.tensor(centroid, dtype=torch.float32)

    def align_lengths(self, *features):
        min_len = min(f.shape[0] for f in features)
        return [f[:min_len] for f in features]

    def annotate(self, path):
        y = self.load_audio(path)

        pitch = self.extract_pitch(y)
        rms = self.extract_rms(y)
        centroid = self.extract_spectral_centroid(y)

        pitch, rms, centroid = self.align_lengths(pitch, rms, centroid)

        return [pitch, rms, centroid]


def process_bucket(input_bucket_name, input_prefix, output_bucket_name):
    client = storage.Client()

    input_bucket = client.bucket(input_bucket_name)
    output_bucket = client.bucket(output_bucket_name)

    blobs = input_bucket.list_blobs(prefix=input_prefix)

    annotator = AudioAnnotator(sample_rate=44100, hop_length=10, win_length=1024, device="cuda")

    for blob in blobs:
        if not blob.name.endswith(".wav"):
            continue

        print(f"Processing {blob.name}")

        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            blob.download_to_filename(tmp.name)
            temp_path = tmp.name

        try:
            annotations = annotator.annotate(temp_path)

            # Convert to something storable
            annotations_np = annotations.numpy()

            # Save locally
            output_path = temp_path + ".npy"
            import numpy as np
            np.save(output_path, annotations_np)

            # Upload to output bucket
            output_blob_name = blob.name.replace(".wav", ".npy")
            output_blob = output_bucket.blob(output_blob_name)
            output_blob.upload_from_filename(output_path)

        finally:
            # Clean up temp files
            os.remove(temp_path)
            if os.path.exists(output_path):
                os.remove(output_path)


def process_gs_bucket():
    process_bucket(
        input_bucket_name="bucket_name",
        input_prefix="folder_name",
        output_bucket_name="output_bucket_name"
    )

directories = ["local_folder"]
def process_local_folders():
    all_audio_files = []

    for dir_path in directories:
        path_obj = Path(dir_path)
        all_audio_files.extend(path_obj.rglob("*.wav"))

    annotator = AudioAnnotator(sample_rate=44100, hop_length=10, win_length=1024, device="cuda")

    current_index = 0
    with open("annotations.csv", "a") as annotation_file:
        for file_path in all_audio_files:

            time = datetime.datetime.now()
            print(f"Annotating {file_path}: {time}")
            annotations = annotator.annotate(str(file_path))
            annotation_file.write(",".join(annotations) + "\n")
            print(f"Finshed at {datetime.datetime.now() - time}")

            current_index += 1