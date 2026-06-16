# scripts/download_cocochorales.py
import argparse
import csv
import io
import subprocess
import sys

from google.cloud import storage
from google.cloud.exceptions import NotFound

# ── Constants ───────────────────────────────────────────────────────────────
GCP_PROJECT   = "ravelatents"
BUCKET_NAME   = "rave-latents"
GCS_PREFIX    = "cocochorales-sample"
SOURCE_BUCKET = "magentadata"
SOURCE_PREFIX = "datasets/cocochorales/cocochorales_full_v1_zipped"

# Metadata shards first so --limit 1 smoke test is fast (~2 MB vs ~4.8 GB).
# main_dataset: 2 train shards x ~4.8 GiB = ~9.6 GiB
# metadata:     2 train shards x ~2.3 MiB = negligible
SHARDS = [
    {"data_type": "metadata",     "split": "train", "shard": 1},
    {"data_type": "metadata",     "split": "train", "shard": 2},
    {"data_type": "main_dataset", "split": "train", "shard": 1},
    {"data_type": "main_dataset", "split": "train", "shard": 2},
]


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


def copy_shard(source_key: str, dest_key: str) -> None:
    # copy_blob() uses the copyTo API which times out for large cross-region
    # objects; gcloud storage cp uses the rewriteTo API which handles them.
    src = f"gs://{SOURCE_BUCKET}/{source_key}"
    dst = f"gs://{BUCKET_NAME}/{dest_key}"
    subprocess.run(
        f"gcloud storage cp {src} {dst}",
        shell=True, check=True, capture_output=True, text=True, encoding="utf-8",
    )


def download_sample(limit: int | None = None) -> None:
    client = get_gcs_client()
    bucket = verify_bucket(client)

    shards = SHARDS[:limit] if limit is not None else SHARDS
    total  = len(shards)

    manifest_rows = [
        {
            "data_type":   s["data_type"],
            "split":       s["split"],
            "shard_id":    s["shard"],
            "source_path": (
                f"gs://{SOURCE_BUCKET}/{SOURCE_PREFIX}"
                f"/{s['data_type']}/{s['split']}/{s['shard']}.tar.bz2"
            ),
            "dest_path": (
                f"gs://{BUCKET_NAME}/{GCS_PREFIX}"
                f"/{s['data_type']}/{s['split']}/{s['shard']}.tar.bz2"
            ),
        }
        for s in shards
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(manifest_rows[0].keys()))
    writer.writeheader()
    writer.writerows(manifest_rows)
    manifest_path = f"{GCS_PREFIX}/manifest.csv"
    bucket.blob(manifest_path).upload_from_string(
        buf.getvalue(), content_type="text/csv"
    )
    print(f"Manifest -> gs://{BUCKET_NAME}/{manifest_path}")

    for i, shard in enumerate(shards, 1):
        source_key = (
            f"{SOURCE_PREFIX}/{shard['data_type']}"
            f"/{shard['split']}/{shard['shard']}.tar.bz2"
        )
        dest_key = (
            f"{GCS_PREFIX}/{shard['data_type']}"
            f"/{shard['split']}/{shard['shard']}.tar.bz2"
        )

        if blob_exists(bucket, dest_key):
            print(f"[{i}/{total}] SKIP  gs://{BUCKET_NAME}/{dest_key}")
            continue
        try:
            copy_shard(source_key, dest_key)
            print(f"[{i}/{total}] OK    gs://{BUCKET_NAME}/{dest_key}")
        except Exception as e:
            print(f"[{i}/{total}] ERROR {dest_key}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap shards processed. --limit 1 for smoke test (copies one ~2 MB metadata shard)."
    )
    args = parser.parse_args()
    download_sample(limit=args.limit)
