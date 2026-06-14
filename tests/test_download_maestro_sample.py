import io
import sys
import zipfile
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest
import requests

from download_maestro_sample import (
    BUCKET_NAME,
    GCS_PREFIX,
    _build_zip_index,
    blob_exists,
    download_sample,
    extract_and_upload,
    verify_bucket,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_mock_bucket(blob_exists_value: bool = False) -> MagicMock:
    mock_blob = MagicMock()
    mock_blob.exists.return_value = blob_exists_value
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    return mock_bucket


def _make_csv(splits: dict[str, int]) -> str:
    """Build a minimal MAESTRO-shaped CSV with the given per-split row counts."""
    rows = ["canonical_composer,canonical_title,split,year,midi_filename,audio_filename,duration"]
    idx = 0
    for split, count in splits.items():
        for i in range(count):
            rows.append(
                f"Composer{idx},Title{idx},{split},2000,"
                f"{split}/file{i}.midi,{split}/file{i}.wav,120.0"
            )
            idx += 1
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# verify_bucket
# ---------------------------------------------------------------------------

def test_verify_bucket_returns_bucket_on_success():
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.get_bucket.return_value = mock_bucket

    result = verify_bucket(mock_client)

    assert result is mock_bucket
    mock_client.get_bucket.assert_called_once_with(BUCKET_NAME)


def test_verify_bucket_exits_on_not_found():
    from google.cloud.exceptions import NotFound

    mock_client = MagicMock()
    mock_client.get_bucket.side_effect = NotFound("bucket")

    with pytest.raises(SystemExit):
        verify_bucket(mock_client)


def test_verify_bucket_exits_on_generic_error():
    mock_client = MagicMock()
    mock_client.get_bucket.side_effect = Exception("permission denied")

    with pytest.raises(SystemExit):
        verify_bucket(mock_client)


# ---------------------------------------------------------------------------
# blob_exists
# ---------------------------------------------------------------------------

def test_blob_exists_returns_true_when_present():
    bucket = _make_mock_bucket(blob_exists_value=True)
    assert blob_exists(bucket, "maestro-sample/train/file.wav") is True


def test_blob_exists_returns_false_when_absent():
    bucket = _make_mock_bucket(blob_exists_value=False)
    assert blob_exists(bucket, "maestro-sample/train/file.wav") is False


# ---------------------------------------------------------------------------
# _build_zip_index
# ---------------------------------------------------------------------------

def test_build_zip_index_strips_top_level_dir():
    info_wav = MagicMock()
    info_wav.filename = "maestro-v3.0.0/2008/file.wav"
    info_wav.is_dir.return_value = False

    info_dir = MagicMock()
    info_dir.filename = "maestro-v3.0.0/"
    info_dir.is_dir.return_value = True

    mock_zf = MagicMock()
    mock_zf.infolist.return_value = [info_dir, info_wav]

    index = _build_zip_index(mock_zf)

    assert index == {"2008/file.wav": "maestro-v3.0.0/2008/file.wav"}


# ---------------------------------------------------------------------------
# extract_and_upload
# ---------------------------------------------------------------------------

def test_extract_and_upload_writes_chunks_to_blob():
    chunks = [b"hello ", b"world", b""]

    mock_src = MagicMock()
    mock_src.__enter__ = lambda s: s
    mock_src.__exit__ = MagicMock(return_value=False)
    mock_src.read.side_effect = chunks

    mock_dst = MagicMock()
    mock_dst.__enter__ = lambda s: s
    mock_dst.__exit__ = MagicMock(return_value=False)

    mock_blob = MagicMock()
    mock_blob.open.return_value = mock_dst

    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_zf = MagicMock()
    mock_zf.open.return_value = mock_src

    zip_index = {"2008/file.wav": "maestro-v3.0.0/2008/file.wav"}

    extract_and_upload(mock_zf, zip_index, "2008/file.wav", mock_bucket, "maestro-sample/2008/file.wav")

    mock_zf.open.assert_called_once_with("maestro-v3.0.0/2008/file.wav")
    mock_dst.write.assert_has_calls([call(b"hello "), call(b"world")])


def test_extract_and_upload_raises_on_missing_entry():
    mock_zf = MagicMock()
    with pytest.raises(KeyError, match="missing/file.wav"):
        extract_and_upload(mock_zf, {}, "missing/file.wav", MagicMock(), "some/path.wav")


# ---------------------------------------------------------------------------
# download_sample — stratified sampling & limit
# ---------------------------------------------------------------------------

def _run_download_sample(csv_text: str, limit=None, blob_already_exists=False):
    """Run download_sample with all external I/O mocked out."""
    mock_bucket = _make_mock_bucket(blob_exists_value=blob_already_exists)

    csv_response = MagicMock()
    csv_response.text = csv_text

    mock_zf = MagicMock()
    mock_zf.infolist.return_value = []

    with (
        patch("download_maestro_sample.get_gcs_client"),
        patch("download_maestro_sample.verify_bucket", return_value=mock_bucket),
        patch("download_maestro_sample.requests.get", return_value=csv_response),
        patch("download_maestro_sample._RangeFile"),
        patch("download_maestro_sample.zipfile.ZipFile", return_value=mock_zf),
        patch("download_maestro_sample._build_zip_index", return_value={}),
        patch("download_maestro_sample.extract_and_upload") as mock_upload,
    ):
        download_sample(limit=limit)

    return mock_bucket, mock_upload


def test_stratified_sample_proportions():
    # 20 train, 10 validation, 10 test → 10% = 2, 1, 1
    csv = _make_csv({"train": 20, "validation": 10, "test": 10})
    mock_bucket, _ = _run_download_sample(csv)

    manifest_call = mock_bucket.blob.return_value.upload_from_string.call_args
    manifest_df = pd.read_csv(io.StringIO(manifest_call[0][0]))

    counts = manifest_df["split"].value_counts()
    assert counts["train"] == 2
    assert counts["validation"] == 1
    assert counts["test"] == 1


def test_sampling_is_deterministic():
    csv = _make_csv({"train": 20, "validation": 10, "test": 10})

    _, mock_upload_1 = _run_download_sample(csv)
    _, mock_upload_2 = _run_download_sample(csv)

    # compare the csv_filename argument (3rd positional arg) of each call
    files_1 = [c.args[2] for c in mock_upload_1.call_args_list]
    files_2 = [c.args[2] for c in mock_upload_2.call_args_list]
    assert files_1 == files_2


def test_limit_caps_files_processed():
    csv = _make_csv({"train": 20, "validation": 10, "test": 10})
    _, mock_upload = _run_download_sample(csv, limit=1)

    # limit=1 → 1 row → 2 file uploads (audio + midi)
    assert mock_upload.call_count == 2


def test_existing_blobs_are_skipped():
    csv = _make_csv({"train": 20, "validation": 10, "test": 10})
    _, mock_upload = _run_download_sample(csv, blob_already_exists=True)

    assert mock_upload.call_count == 0
