# Test spec

## How to use this file

The unit tests below use `download_maestro_sample` as a concrete worked example —
a real module that already exists and can be imported. When writing tests for a new
dataset, replace every occurrence of `download_maestro_sample` with your actual
module name (e.g. `download_cocochorales`), and replace the constants
`BUCKET_NAME`, `GCS_PREFIX`, `GCP_PROJECT` with imports from that module.

**Never leave `<dataset_name>` as a placeholder — tests with unresolved placeholders
pass vacuously and catch nothing.**

---

## Unit tests

File: `tests/test_download_maestro_sample.py`
(rename to `test_download_<your_dataset>.py` for new datasets)

```python
# tests/test_download_maestro_sample.py
import io
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# conftest.py adds scripts/ to sys.path — do not duplicate that here
from download_maestro_sample import (
    BUCKET_NAME,
    GCS_PREFIX,
    GCP_PROJECT,
    _build_zip_index,
    blob_exists,
    download_sample,
    extract_and_upload,
    verify_bucket,
)
from google.cloud.exceptions import NotFound


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_mock_bucket(blob_exists_value: bool = False) -> MagicMock:
    """Return a mock bucket whose blobs report exists() as given."""
    mock_blob = MagicMock()
    mock_blob.exists.return_value = blob_exists_value
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    return mock_bucket


def _make_zip(entries: list[str]) -> zipfile.ZipFile:
    """Return an in-memory ZipFile containing the given entry paths."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in entries:
            zf.writestr(name, b"fake audio data")
    buf.seek(0)
    return zipfile.ZipFile(buf)


# Minimal CSV that mirrors MAESTRO's structure:
# 10 train + 5 validation + 5 test rows = 20 rows total
SAMPLE_CSV = "\n".join([
    "split,audio_filename,midi_filename,duration",
    *[f"train,2008/train_{i:02d}.wav,2008/train_{i:02d}.midi,120" for i in range(10)],
    *[f"validation,2008/val_{i:02d}.wav,2008/val_{i:02d}.midi,90" for i in range(5)],
    *[f"test,2008/test_{i:02d}.wav,2008/test_{i:02d}.midi,95" for i in range(5)],
])


def _run_download(
    csv: str = SAMPLE_CSV,
    blob_already_exists: bool = False,
    limit: int | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Run download_sample with all external I/O mocked.

    Returns (mock_bucket, mock_extract_and_upload).
    mock_bucket.blob.return_value.upload_from_string holds manifest call args.
    mock_extract holds all per-file upload call args.
    """
    mock_bucket = _make_mock_bucket(blob_exists_value=blob_already_exists)

    # Make blob.open() return a usable context manager
    mock_dst = MagicMock()
    mock_blob_open = MagicMock()
    mock_blob_open.__enter__ = lambda s: mock_dst
    mock_blob_open.__exit__ = MagicMock(return_value=False)
    mock_bucket.blob.return_value.open.return_value = mock_blob_open

    # Build a zip index that matches the CSV paths in SAMPLE_CSV
    zip_entries = [
        f"maestro-v3.0.0/2008/train_{i:02d}.wav" for i in range(10)
    ] + [
        f"maestro-v3.0.0/2008/train_{i:02d}.midi" for i in range(10)
    ] + [
        f"maestro-v3.0.0/2008/val_{i:02d}.wav" for i in range(5)
    ] + [
        f"maestro-v3.0.0/2008/val_{i:02d}.midi" for i in range(5)
    ] + [
        f"maestro-v3.0.0/2008/test_{i:02d}.wav" for i in range(5)
    ] + [
        f"maestro-v3.0.0/2008/test_{i:02d}.midi" for i in range(5)
    ]
    mock_zf = _make_zip(zip_entries)

    with (
        patch("requests.get") as mock_requests_get,
        patch("download_maestro_sample.get_gcs_client"),
        patch("download_maestro_sample.verify_bucket", return_value=mock_bucket),
        patch("download_maestro_sample.zipfile.ZipFile", return_value=mock_zf),
        patch("download_maestro_sample.extract_and_upload") as mock_extract,
    ):
        mock_response = MagicMock()
        mock_response.text = csv
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response

        download_sample(limit=limit)

    return mock_bucket, mock_extract


# ── verify_bucket ──────────────────────────────────────────────────────────

def test_verify_bucket_returns_bucket_on_success():
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    result = verify_bucket(mock_client)
    assert result is mock_bucket
    mock_client.get_bucket.assert_called_once_with(BUCKET_NAME)


def test_verify_bucket_exits_on_not_found():
    mock_client = MagicMock()
    mock_client.get_bucket.side_effect = NotFound("bucket")
    with pytest.raises(SystemExit):
        verify_bucket(mock_client)


def test_verify_bucket_exits_on_generic_error():
    mock_client = MagicMock()
    mock_client.get_bucket.side_effect = Exception("permission denied")
    with pytest.raises(SystemExit):
        verify_bucket(mock_client)


# ── blob_exists ────────────────────────────────────────────────────────────

def test_blob_exists_returns_true_when_present():
    bucket = _make_mock_bucket(blob_exists_value=True)
    assert blob_exists(bucket, f"{GCS_PREFIX}/train/file.wav") is True


def test_blob_exists_returns_false_when_absent():
    bucket = _make_mock_bucket(blob_exists_value=False)
    assert blob_exists(bucket, f"{GCS_PREFIX}/train/file.wav") is False


# ── _build_zip_index ───────────────────────────────────────────────────────

def test_build_zip_index_strips_top_level_dir():
    zf = _make_zip([
        "maestro-v3.0.0/2008/audio.wav",
        "maestro-v3.0.0/2008/audio.midi",
    ])
    index = _build_zip_index(zf)
    assert "2008/audio.wav" in index
    assert index["2008/audio.wav"] == "maestro-v3.0.0/2008/audio.wav"
    assert "2008/audio.midi" in index


def test_build_zip_index_excludes_directory_entries():
    zf = _make_zip([
        "maestro-v3.0.0/",
        "maestro-v3.0.0/2008/",
        "maestro-v3.0.0/2008/audio.wav",
    ])
    index = _build_zip_index(zf)
    # Only the actual file should appear
    assert len(index) == 1
    assert "2008/audio.wav" in index


def test_build_zip_index_handles_no_top_level_dir():
    # Some zips have no top-level dir — entries start with the file path directly
    zf = _make_zip(["2008/audio.wav", "2008/audio.midi"])
    index = _build_zip_index(zf)
    assert "2008/audio.wav" in index
    assert index["2008/audio.wav"] == "2008/audio.wav"


# ── extract_and_upload ─────────────────────────────────────────────────────

def test_extract_and_upload_writes_chunks_to_blob():
    mock_src = MagicMock()
    mock_src.read.side_effect = [b"chunk_one", b"chunk_two", b""]

    mock_dst = MagicMock()
    mock_open_ctx = MagicMock()
    mock_open_ctx.__enter__ = lambda s: mock_dst
    mock_open_ctx.__exit__ = MagicMock(return_value=False)

    mock_zf_open_ctx = MagicMock()
    mock_zf_open_ctx.__enter__ = lambda s: mock_src
    mock_zf_open_ctx.__exit__ = MagicMock(return_value=False)

    mock_zf = MagicMock()
    mock_zf.open.return_value = mock_zf_open_ctx

    mock_blob = MagicMock()
    mock_blob.open.return_value = mock_open_ctx
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    zip_index = {"2008/audio.wav": "maestro-v3.0.0/2008/audio.wav"}
    extract_and_upload(
        mock_zf, zip_index, "2008/audio.wav",
        mock_bucket, f"{GCS_PREFIX}/2008/audio.wav"
    )

    assert mock_dst.write.call_count == 2
    mock_dst.write.assert_any_call(b"chunk_one")
    mock_dst.write.assert_any_call(b"chunk_two")


def test_extract_and_upload_raises_on_missing_entry():
    mock_zf     = MagicMock()
    mock_bucket = MagicMock()
    with pytest.raises(KeyError, match="not found in zip index"):
        extract_and_upload(
            mock_zf, {}, "missing/file.wav",
            mock_bucket, f"{GCS_PREFIX}/missing/file.wav"
        )


# ── download_sample ────────────────────────────────────────────────────────

def test_stratified_sample_preserves_split_proportions():
    mock_bucket, _ = _run_download()
    # Manifest is uploaded via upload_from_string on the manifest blob
    manifest_call = mock_bucket.blob.return_value.upload_from_string.call_args
    manifest_csv  = manifest_call[0][0]
    df = pd.read_csv(io.StringIO(manifest_csv))
    counts = df["split"].value_counts()
    # SAMPLE_CSV has 10 train / 5 val / 5 test — 10% -> at least 1 each
    assert counts.get("train", 0) >= 1
    assert counts.get("validation", 0) >= 1
    assert counts.get("test", 0) >= 1


def test_sampling_is_deterministic():
    _, mock_extract_1 = _run_download()
    _, mock_extract_2 = _run_download()
    calls_1 = [str(c) for c in mock_extract_1.call_args_list]
    calls_2 = [str(c) for c in mock_extract_2.call_args_list]
    assert calls_1 == calls_2


def test_limit_caps_rows_processed():
    _, mock_extract = _run_download(limit=1)
    # limit=1 -> 1 row -> 2 file calls (audio_filename + midi_filename)
    assert mock_extract.call_count == 2


def test_existing_blobs_are_skipped():
    _, mock_extract = _run_download(blob_already_exists=True)
    assert mock_extract.call_count == 0


# ── Edge cases ─────────────────────────────────────────────────────────────

def test_leading_slash_stripped_from_csv_path():
    csv_with_slashes = SAMPLE_CSV.replace(
        "2008/train_00.wav", "/2008/train_00.wav"
    ).replace(
        "2008/train_00.midi", "/2008/train_00.midi"
    )
    _, mock_extract = _run_download(csv=csv_with_slashes, limit=1)
    if mock_extract.call_count > 0:
        # 5th positional arg to extract_and_upload is gcs_path
        gcs_path = mock_extract.call_args_list[0][0][4]
        assert "//" not in gcs_path, f"Double slash in GCS path: {gcs_path}"


def test_error_in_one_file_does_not_abort_loop():
    mock_bucket = _make_mock_bucket(blob_exists_value=False)

    with (
        patch("requests.get") as mock_get,
        patch("download_maestro_sample.get_gcs_client"),
        patch("download_maestro_sample.verify_bucket", return_value=mock_bucket),
        patch("download_maestro_sample.zipfile.ZipFile"),
        patch(
            "download_maestro_sample._build_zip_index",
            return_value={"2008/train_00.wav": "maestro-v3.0.0/2008/train_00.wav"},
        ),
        patch(
            "download_maestro_sample.extract_and_upload",
            side_effect=[Exception("SSL error"), None, None, None],
        ) as mock_extract,
    ):
        mock_get.return_value.text = SAMPLE_CSV
        mock_get.return_value.raise_for_status = MagicMock()
        download_sample(limit=2)

    # 2 rows x 2 file cols = 4 calls total; first raises, rest continue
    assert mock_extract.call_count == 4


def test_manifest_uploaded_before_file_loop():
    """Manifest must be written before any extract_and_upload call."""
    call_order = []
    mock_bucket = _make_mock_bucket(blob_exists_value=False)
    mock_bucket.blob.return_value.upload_from_string.side_effect = (
        lambda *a, **kw: call_order.append("manifest")
    )

    with (
        patch("requests.get") as mock_get,
        patch("download_maestro_sample.get_gcs_client"),
        patch("download_maestro_sample.verify_bucket", return_value=mock_bucket),
        patch("download_maestro_sample.zipfile.ZipFile"),
        patch("download_maestro_sample._build_zip_index", return_value={
            "2008/train_00.wav": "maestro-v3.0.0/2008/train_00.wav"
        }),
        patch(
            "download_maestro_sample.extract_and_upload",
            side_effect=lambda *a, **kw: call_order.append("upload"),
        ),
    ):
        mock_get.return_value.text = SAMPLE_CSV
        mock_get.return_value.raise_for_status = MagicMock()
        download_sample(limit=1)

    assert call_order[0] == "manifest", "Manifest must be written before any file upload"


def test_manifest_content_type_is_text_csv():
    mock_bucket, _ = _run_download()
    upload_call = mock_bucket.blob.return_value.upload_from_string.call_args
    content_type = upload_call[1].get("content_type") or upload_call[0][1]
    assert content_type == "text/csv"


# ── Run ────────────────────────────────────────────────────────────────────
# uv run pytest tests/test_download_maestro_sample.py -v
```

---

## Integration tests — verify data landed in GCS {#integration}

Run manually after every completed download. Not automated via pytest.

### Test 1 — Object count matches manifest

```bash
# Expected: (manifest_rows x file_columns) + 1  (the +1 is the manifest itself)
MANIFEST_ROWS=$(gcloud storage cat "gs://rave-latents/<prefix>/manifest.csv" | wc -l)
OBJECT_COUNT=$(gcloud storage ls --recursive "gs://rave-latents/<prefix>/" | wc -l)

echo "Manifest data rows : $((MANIFEST_ROWS - 1))"
echo "Objects in GCS     : $((OBJECT_COUNT - 1))"
echo "Expected objects   : $(( (MANIFEST_ROWS - 1) * <file_columns> ))"
```

If `objects < expected`: ERROR files were not retried. Re-run the script.

### Test 2 — Total size is within agreed subset-size

```bash
gcloud storage du --summarize "gs://rave-latents/<prefix>/"
# Compare output against the GB agreed in the plan (SKILL.md Step 4)
```

A large discrepancy means sampling went wrong. Check the manifest CSV and re-run.

### Test 3 — Manifest is present and readable

```bash
gcloud storage cat "gs://rave-latents/<prefix>/manifest.csv" | head -5
# Expected: header row + at least one data row
```

If missing: script crashed before writing manifest. Check error log and re-run.

### Test 4 — No duplicate blobs

```bash
gcloud storage ls --recursive "gs://rave-latents/<prefix>/" | sort | uniq -d
# Expected: no output (idempotency should prevent duplicates)
```

### Test 5 — Spot-check: one WAV file is valid audio

```bash
gcloud storage cp "gs://rave-latents/<prefix>/<any_file>.wav" /tmp/check.wav

python3 -c "
import wave
with wave.open('/tmp/check.wav') as w:
    print(f'Frames: {w.getnframes()}, Rate: {w.getframerate()} Hz, Channels: {w.getnchannels()}')
"
# Expected: numeric output with no exception
# If wave.Error: file is corrupt or wrong format was uploaded
```

### Test 6 — No ERROR lines remain after cleanup re-run

```bash
# Run the script one final time
uv run python -u scripts/download_<name>.py 2>&1 | tee /tmp/final_run.log

# Check for ERRORs
grep "ERROR" /tmp/final_run.log
# Expected: no output

# All lines should be SKIP
grep -v "SKIP" /tmp/final_run.log | grep -v "GCS connection\|Manifest\|Sampled\|Total\|Zip index"
# Expected: no file-transfer lines other than SKIP
```
