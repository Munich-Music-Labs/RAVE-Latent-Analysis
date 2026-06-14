# Script templates

One template per pattern. Copy the one matching the classified pattern,
fill in the constants block, and adapt `col_to_zip` / sampling to the dataset.

---

## Pattern A — Individual files via HTTP + CSV manifest

```python
# scripts/download_<dataset_name>.py
import argparse
import io
import sys

import pandas as pd
import requests
from google.cloud import storage
from google.cloud.exceptions import NotFound

# ── Constants ──────────────────────────────────────────────────────────────
METADATA_URL = "https://..."        # direct URL to CSV manifest
FILE_BASE    = "https://..."        # base URL prepended to CSV file paths

GCP_PROJECT  = "ravelatents"
BUCKET_NAME  = "rave-latents"
GCS_PREFIX   = "<dataset-name>"

SAMPLE_FRAC  = 0.10                 # adjust to stay within agreed subset-size


def get_gcs_client() -> storage.Client:
    return storage.Client(project=GCP_PROJECT)


def verify_bucket(client: storage.Client) -> storage.Bucket:
    try:
        bucket = client.get_bucket(BUCKET_NAME)
        print(f"GCS connection OK - bucket gs://{BUCKET_NAME} accessible.")
        return bucket
    except NotFound:
        print(f"ERROR: Bucket gs://{BUCKET_NAME} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not access gs://{BUCKET_NAME}: {e}")
        sys.exit(1)


def blob_exists(bucket: storage.Bucket, gcs_path: str) -> bool:
    return bucket.blob(gcs_path).exists()


def download_and_upload(
    url: str,
    bucket: storage.Bucket,
    gcs_path: str,
) -> None:
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    blob = bucket.blob(gcs_path)
    with blob.open("wb") as dst:
        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
            dst.write(chunk)


def download_sample(limit: int | None = None) -> None:
    client = get_gcs_client()
    bucket = verify_bucket(client)

    print("Fetching metadata CSV...")
    r = requests.get(METADATA_URL, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    print(f"Total rows: {len(df)}")

    sampled = (
        df.groupby("split", group_keys=False)
        .sample(frac=SAMPLE_FRAC, random_state=42)
        .reset_index(drop=True)
    )
    print(f"Sampled rows: {len(sampled)}")

    manifest_path = f"{GCS_PREFIX}/manifest.csv"
    bucket.blob(manifest_path).upload_from_string(
        sampled.to_csv(index=False), content_type="text/csv"
    )
    print(f"Manifest -> gs://{BUCKET_NAME}/{manifest_path}")

    # Adapt: replace "audio_filename" with the actual column name(s) in this CSV
    file_cols = ["audio_filename"]
    rows  = sampled.head(limit) if limit is not None else sampled
    total = len(rows) * len(file_cols)
    done  = 0

    for _, row in rows.iterrows():
        for col in file_cols:
            src_filename = row[col].lstrip("/")
            url      = f"{FILE_BASE}/{src_filename}"
            gcs_path = f"{GCS_PREFIX}/{src_filename}"
            done += 1

            if blob_exists(bucket, gcs_path):
                print(f"[{done}/{total}] SKIP  gs://{BUCKET_NAME}/{gcs_path}")
                continue
            try:
                download_and_upload(url, bucket, gcs_path)
                print(f"[{done}/{total}] OK    gs://{BUCKET_NAME}/{gcs_path}")
            except Exception as e:
                print(f"[{done}/{total}] ERROR {gcs_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    download_sample(limit=args.limit)
```

---

## Pattern B — Zip via HTTP Range requests (Magenta / GCS datasets)

```python
# scripts/download_<dataset_name>.py
import argparse
import io
import sys
import zipfile

import pandas as pd
import requests
from google.cloud import storage
from google.cloud.exceptions import NotFound

# ── Constants ──────────────────────────────────────────────────────────────
DATASET_BASE  = "https://storage.googleapis.com/magentadata/datasets/<name>/<version>"
METADATA_URL  = f"{DATASET_BASE}/<name>.csv"
AUDIO_ZIP_URL = f"{DATASET_BASE}/<name>.zip"
# MIDI_ZIP_URL = f"{DATASET_BASE}/<name>-midi.zip"  # only if MIDI in separate zip

GCP_PROJECT   = "ravelatents"
BUCKET_NAME   = "rave-latents"
GCS_PREFIX    = "<dataset-name>"

SAMPLE_FRAC   = 0.10


class _RangeFile:
    """Seekable file backed by HTTP Range requests.
    Lets zipfile parse the central directory without fetching the full archive.
    One HEAD for total size, then one GET per read call.
    """
    def __init__(self, url: str):
        self.url  = url
        r = requests.head(url, timeout=30)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        self._pos = 0

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:   self._pos = pos
        elif whence == 1: self._pos += pos
        elif whence == 2: self._pos = self.size + pos
        return self._pos

    def tell(self) -> int: return self._pos

    def read(self, n: int = -1) -> bytes:
        if self._pos >= self.size:
            return b""
        end = (self.size - 1) if n < 0 else min(self._pos + n - 1, self.size - 1)
        r = requests.get(
            self.url, headers={"Range": f"bytes={self._pos}-{end}"}, timeout=300
        )
        r.raise_for_status()
        data = r.content
        self._pos += len(data)
        return data

    def seekable(self) -> bool: return True
    def readable(self) -> bool: return True


def _build_zip_index(zf: zipfile.ZipFile) -> dict[str, str]:
    """Map CSV-relative paths to full zip entry names by stripping the
    top-level directory prefix.

    e.g.  "maestro-v3.0.0/2008/file.wav"  ->  key "2008/file.wav"
    """
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
        print(f"GCS connection OK - bucket gs://{BUCKET_NAME} accessible.")
        return bucket
    except NotFound:
        print(f"ERROR: Bucket gs://{BUCKET_NAME} not found.")
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
        raise KeyError(f"'{csv_filename}' not found in zip index")
    blob = bucket.blob(gcs_path)
    with zf.open(entry_name) as src, blob.open("wb") as dst:
        while chunk := src.read(8 * 1024 * 1024):
            dst.write(chunk)


def download_sample(limit: int | None = None) -> None:
    client = get_gcs_client()
    bucket = verify_bucket(client)

    print("Fetching metadata CSV...")
    r = requests.get(METADATA_URL, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    print(f"Total rows: {len(df)}")

    sampled = (
        df.groupby("split", group_keys=False)
        .sample(frac=SAMPLE_FRAC, random_state=42)
        .reset_index(drop=True)
    )
    print(f"Sampled rows ({int(SAMPLE_FRAC * 100)}%): {len(sampled)}")
    if "split" in sampled.columns:
        print(sampled["split"].value_counts().to_string())

    manifest_path = f"{GCS_PREFIX}/manifest.csv"
    bucket.blob(manifest_path).upload_from_string(
        sampled.to_csv(index=False), content_type="text/csv"
    )
    print(f"Manifest -> gs://{BUCKET_NAME}/{manifest_path}")

    print("Opening zip archive via range requests...")
    audio_zf    = zipfile.ZipFile(_RangeFile(AUDIO_ZIP_URL))
    audio_index = _build_zip_index(audio_zf)
    print(f"Zip index ready: {len(audio_index)} entries")

    # Verify index matches CSV before the loop
    sample_csv_path = df.iloc[0]["audio_filename"].lstrip("/")
    if sample_csv_path not in audio_index:
        print(f"ERROR: CSV path '{sample_csv_path}' not found in zip index.")
        print(f"First 3 index keys: {list(audio_index.keys())[:3]}")
        sys.exit(1)

    # Combined zip (E-GMD style): both columns resolved from same zip
    col_to_zip = {
        "audio_filename": (audio_zf, audio_index),
        "midi_filename":  (audio_zf, audio_index),
    }
    # Separate zips (MAESTRO style): uncomment + fill in midi_zf / midi_index
    # midi_zf    = zipfile.ZipFile(_RangeFile(MIDI_ZIP_URL))
    # midi_index = _build_zip_index(midi_zf)
    # col_to_zip = {
    #     "audio_filename": (audio_zf, audio_index),
    #     "midi_filename":  (midi_zf,  midi_index),
    # }

    rows      = sampled.head(limit) if limit is not None else sampled
    file_cols = list(col_to_zip.keys())
    total     = len(rows) * len(file_cols)
    done      = 0

    for _, row in rows.iterrows():
        for col in file_cols:
            src_filename = row[col].lstrip("/")
            gcs_path     = f"{GCS_PREFIX}/{src_filename}"
            zf, idx      = col_to_zip[col]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap rows processed. --limit 1 for smoke test.")
    args = parser.parse_args()
    download_sample(limit=args.limit)
```

---

## Pattern C1 — Tar, local extraction

```python
# scripts/download_<dataset_name>.py
import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import requests
from google.cloud import storage
from google.cloud.exceptions import NotFound

# ── Constants ──────────────────────────────────────────────────────────────
TAR_URL     = "https://..."      # URL of the tar.gz subset
GCP_PROJECT = "ravelatents"
BUCKET_NAME = "rave-latents"
GCS_PREFIX  = "<dataset-name>"

# File extensions to upload (ignore everything else in the tar)
UPLOAD_EXTENSIONS = {".wav", ".mid", ".midi"}


def get_gcs_client() -> storage.Client:
    return storage.Client(project=GCP_PROJECT)


def verify_bucket(client: storage.Client) -> storage.Bucket:
    try:
        bucket = client.get_bucket(BUCKET_NAME)
        print(f"GCS connection OK - bucket gs://{BUCKET_NAME} accessible.")
        return bucket
    except NotFound:
        print(f"ERROR: Bucket gs://{BUCKET_NAME} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not access gs://{BUCKET_NAME}: {e}")
        sys.exit(1)


def blob_exists(bucket: storage.Bucket, gcs_path: str) -> bool:
    return bucket.blob(gcs_path).exists()


def upload_file(local_path: Path, bucket: storage.Bucket, gcs_path: str) -> None:
    blob = bucket.blob(gcs_path)
    with local_path.open("rb") as src, blob.open("wb") as dst:
        while chunk := src.read(8 * 1024 * 1024):
            dst.write(chunk)


def download_sample(limit: int | None = None) -> None:
    client = get_gcs_client()
    bucket = verify_bucket(client)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tar_path = tmp_path / "archive.tar.gz"

        print(f"Downloading tar to {tar_path} ...")
        r = requests.get(TAR_URL, stream=True, timeout=600)
        r.raise_for_status()
        with tar_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
        print(f"Download complete: {tar_path.stat().st_size / 1e9:.2f} GB")

        print("Extracting...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmp_path)
        tar_path.unlink()   # free disk space immediately after extraction

        files = sorted(
            p for p in tmp_path.rglob("*")
            if p.is_file() and p.suffix in UPLOAD_EXTENSIONS
        )
        if limit is not None:
            files = files[:limit]

        total = len(files)
        print(f"Files to upload: {total}")

        # Upload manifest: relative paths of all selected files
        manifest_lines = "\n".join(str(f.relative_to(tmp_path)) for f in files)
        bucket.blob(f"{GCS_PREFIX}/manifest.csv").upload_from_string(
            "path\n" + manifest_lines, content_type="text/csv"
        )
        print(f"Manifest -> gs://{BUCKET_NAME}/{GCS_PREFIX}/manifest.csv")

        for done, local_path in enumerate(files, 1):
            rel      = local_path.relative_to(tmp_path)
            gcs_path = f"{GCS_PREFIX}/{rel}"

            if blob_exists(bucket, gcs_path):
                print(f"[{done}/{total}] SKIP  gs://{BUCKET_NAME}/{gcs_path}")
                continue
            try:
                upload_file(local_path, bucket, gcs_path)
                print(f"[{done}/{total}] OK    gs://{BUCKET_NAME}/{gcs_path}")
            except Exception as e:
                print(f"[{done}/{total}] ERROR {gcs_path}: {e}")

        print("Temp directory cleaned up automatically.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap files uploaded. --limit 5 for smoke test.")
    args = parser.parse_args()
    download_sample(limit=args.limit)
```

---

## Pattern C2 — Tar, GCS-native extraction (no local disk)

Use when local disk < 2× tar size. Requires a Cloud Run job or Dataproc cluster.

```bash
# Step 1: stream tar directly to GCS (no local temp file)
curl -L <tar_url> | gcloud storage cp - gs://rave-latents/<name>/staging/<archive>.tar.gz

# Step 2: submit Cloud Run job to extract
# (create a simple Python job that reads from GCS, extracts, writes back)
gcloud run jobs create extract-<name> \
  --image gcr.io/google.com/cloudsdktool/cloud-sdk \
  --command bash \
  --args="-c,gsutil cp gs://rave-latents/<name>/staging/<archive>.tar.gz - | tar -xz | gsutil -m cp -r - gs://rave-latents/<name>/" \
  --region europe-west2 \
  --project ravelatents

gcloud run jobs execute extract-<name> --region europe-west2 --project ravelatents

# Step 3: delete staging tar from GCS
gcloud storage rm gs://rave-latents/<name>/staging/<archive>.tar.gz
```

---

## uv / project setup

```toml
# pyproject.toml — scripts-only, no [build-system] block
[tool.uv]
package = false
```

```bash
uv add google-cloud-storage pandas requests
uv add --dev pytest
```

Do NOT add `[build-system]` — hatchling fails without a package directory.
Do NOT add custom PyTorch index — `download-r2.pytorch.org` fails DNS on some ISPs.

---

## Windows command reference

```bash
# Run (unbuffered — required for background runs)
uv run python -u scripts/download_<name>.py
uv run python -u scripts/download_<name>.py --limit 1

# GCS connectivity check (expect 400 = healthy)
curl -s --max-time 10 -o /dev/null -w "%{http_code}" https://storage.googleapis.com

# Bucket size — gcloud storage only, NOT gsutil (fails on Windows)
gcloud storage du --summarize gs://rave-latents
gcloud storage du --summarize gs://rave-latents/<prefix>

# List prefix
gcloud storage ls --recursive "gs://rave-latents/<prefix>/" | head -20
```

```powershell
# Monitor background run progress
$f = "C:\Users\spyke\AppData\Local\Temp\claude\<hash>\tasks\<id>.output"
$lines = Get-Content $f
$ok    = ($lines | Select-String " OK ").Count
$skip  = ($lines | Select-String "SKIP").Count
$err   = ($lines | Select-String "ERROR").Count
"OK: $ok  SKIP: $skip  ERROR: $err  Total: $($ok+$skip+$err)"
"Last: $($lines[-1])"

# Audit errors
Select-String "ERROR" $f | Select-Object -ExpandProperty Line
```
