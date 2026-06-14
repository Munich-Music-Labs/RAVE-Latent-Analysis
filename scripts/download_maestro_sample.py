import argparse
import io
import sys
import zipfile

import pandas as pd
import requests
from google.cloud import storage
from google.cloud.exceptions import NotFound

MAESTRO_BASE = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0"
METADATA_URL = f"{MAESTRO_BASE}/maestro-v3.0.0.csv"
AUDIO_ZIP_URL = f"{MAESTRO_BASE}/maestro-v3.0.0.zip"
MIDI_ZIP_URL = f"{MAESTRO_BASE}/maestro-v3.0.0-midi.zip"

GCP_PROJECT = "ravelatents"
BUCKET_NAME = "rave-latents"
GCS_PREFIX = "maestro-sample"


class _RangeFile:
    """Seekable file-like object backed by HTTP Range requests.

    Lets zipfile parse the central directory and fetch individual entries
    without downloading the full archive.
    """

    def __init__(self, url: str):
        self.url = url
        r = requests.head(url, timeout=30)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        self._pos = 0

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = self.size + pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if self._pos >= self.size:
            return b""
        end = (self.size - 1) if n < 0 else min(self._pos + n - 1, self.size - 1)
        r = requests.get(self.url, headers={"Range": f"bytes={self._pos}-{end}"}, timeout=300)
        r.raise_for_status()
        data = r.content
        self._pos += len(data)
        return data

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True


def _build_zip_index(zf: zipfile.ZipFile) -> dict[str, str]:
    """Map CSV-relative path → full zip entry name by stripping the top-level dir."""
    index = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = info.filename.split("/", 1)
        key = parts[1] if len(parts) == 2 else parts[0]
        index[key] = info.filename
    return index


def get_gcs_client() -> storage.Client:
    return storage.Client(project=GCP_PROJECT)


def verify_bucket(client: storage.Client) -> storage.Bucket:
    try:
        bucket = client.get_bucket(BUCKET_NAME)
        print(f"GCS connection OK — bucket gs://{BUCKET_NAME} is accessible.")
        return bucket
    except NotFound:
        print(f"ERROR: Bucket gs://{BUCKET_NAME} not found. Check the bucket name and project.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not access gs://{BUCKET_NAME}: {e}")
        sys.exit(1)


def blob_exists(bucket: storage.Bucket, gcs_path: str) -> bool:
    return bucket.blob(gcs_path).exists()


def extract_and_upload(
    zf: zipfile.ZipFile,
    zip_index: dict[str, str],
    csv_filename: str,
    bucket: storage.Bucket,
    gcs_path: str,
) -> None:
    entry_name = zip_index.get(csv_filename)
    if entry_name is None:
        raise KeyError(f"'{csv_filename}' not found in zip")

    blob = bucket.blob(gcs_path)
    with zf.open(entry_name) as src, blob.open("wb") as dst:
        while chunk := src.read(8 * 1024 * 1024):
            dst.write(chunk)


def download_sample(limit: int | None = None) -> None:
    client = get_gcs_client()
    bucket = verify_bucket(client)

    print("Fetching MAESTRO metadata CSV...")
    response = requests.get(METADATA_URL, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    print(f"Total rows: {len(df)}")

    sampled = (
        df.groupby("split", group_keys=False)
        .sample(frac=0.1, random_state=42)
        .reset_index(drop=True)
    )
    print(f"Sampled rows (10% stratified by split): {len(sampled)}")
    print(sampled["split"].value_counts().to_string())

    manifest_path = f"{GCS_PREFIX}/manifest.csv"
    bucket.blob(manifest_path).upload_from_string(sampled.to_csv(index=False), content_type="text/csv")
    print(f"Manifest saved to gs://{BUCKET_NAME}/{manifest_path}")

    print("Opening MAESTRO zip archives via range requests (no full download)...")
    audio_zf = zipfile.ZipFile(_RangeFile(AUDIO_ZIP_URL))
    midi_zf = zipfile.ZipFile(_RangeFile(MIDI_ZIP_URL))
    audio_index = _build_zip_index(audio_zf)
    midi_index = _build_zip_index(midi_zf)
    print(f"Zip indices ready: {len(audio_index)} audio, {len(midi_index)} MIDI entries.")

    col_to_zip: dict[str, tuple[zipfile.ZipFile, dict[str, str]]] = {
        "audio_filename": (audio_zf, audio_index),
        "midi_filename": (midi_zf, midi_index),
    }

    rows = sampled.head(limit) if limit is not None else sampled
    file_cols = list(col_to_zip.keys())
    total = len(rows) * len(file_cols)
    done = 0

    for _, row in rows.iterrows():
        for col in file_cols:
            src_filename = row[col].lstrip("/")
            gcs_path = f"{GCS_PREFIX}/{src_filename}"
            zf, idx = col_to_zip[col]
            done += 1

            if blob_exists(bucket, gcs_path):
                print(f"[{done}/{total}] SKIP  gs://{BUCKET_NAME}/{gcs_path}")
                continue

            try:
                extract_and_upload(zf, idx, src_filename, bucket, gcs_path)
                print(f"[{done}/{total}] OK    gs://{BUCKET_NAME}/{gcs_path}")
            except Exception as e:
                print(f"[{done}/{total}] ERROR {gcs_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a MAESTRO sample to GCS.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of rows (e.g. 1 for smoke test).")
    args = parser.parse_args()
    download_sample(limit=args.limit)
