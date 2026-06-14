---
name: dataset-downloader
description: Use when downloading audio/ML datasets to GCS. Covers research, pattern classification (zip/tar/direct/auth/hf), script generation, smoke testing, and GCS verification. Invoke directly with /dataset-downloader or Claude will load it automatically when dataset download is mentioned.
argument-hint: <dataset-name> <subset-size>
arguments: dataset_name subset_size
allowed-tools: Bash PowerShell Read Write Grep Glob WebSearch WebFetch
---

# dataset-downloader

Download a subset of an audio/ML dataset to `gs://rave-latents/` on GCP project `ravelatents`.

> **COST RULE — NON-NEGOTIABLE**: `subset-size` is MANDATORY. Never proceed without
> an explicit size cap. If missing, stop immediately and ask:
> *"What is the maximum GB or file count for this sample? Cost control requires
> an explicit cap before anything else happens."*

Proven on:
- MAESTRO v3.0.0 — 12.2 GB, 202 OK / 0 ERR
- E-GMD v1.0.0 — 10.3 GB, 7281 OK / 3 SSL ERR fixed on re-run

See [pitfalls.md](./pitfalls.md) for all known failure modes.
See [template.md](./template.md) for script templates (all patterns).
See [tests.md](./tests.md) for unit and integration test specs.

---

## Step 1 — Enforce cost gate

If `$subset_size` is empty, STOP. Do not continue until the user provides one.

---

## Step 2 — Research

Web-search in parallel. Report findings before writing any code.

1. Canonical download URL and hosting provider
2. **Hosting structure** — individual files via HTTP? zip? tar? auth required?
   Hugging Face? S3? This is the most important question (see pitfalls.md §4)
3. Metadata CSV/JSON listing file paths and splits
4. Total dataset size and file count
5. Archive internals — top-level directory prefix inside zip/tar?

Report to user:

| Field | Value |
|-------|-------|
| Source URL | |
| Hosting format | |
| Metadata file | URL or "none" |
| Total size | |
| Proposed sample | stays within `$subset_size` |

---

## Step 3 — Classify pattern

### Pattern A — Individual files via HTTP + CSV manifest
Files directly fetchable by URL. Sample CSV, `requests.get()` per file.

### Pattern B — Zip with HTTP Range support (Magenta/GCS datasets)
Files only inside a zip served from a Range-capable server.
`_RangeFile` + `zipfile.ZipFile` — fetches central directory only (~4 MB for 90 GB zip).
`_build_zip_index()` handles top-level prefix mismatch.

### Pattern C — Tar/multi-archive (e.g. CocoChorales)
Files in `.tar.gz` bundles. No Range support.

**Before writing any code**, calculate the tar subset size and ask:

> *"Downloading this subset requires approximately X GB of temporary local disk
> space. Your local machine will need that free during extraction.
>
> Alternatively, I can download the tar directly to GCS, extract it there,
> and delete the tar from GCS when done — no local disk used, but requires
> a short Dataproc or Cloud Run job. Which would you prefer?"*

**Option C1 — Local:** download tar → extract → upload selected files → delete local
**Option C2 — GCS-native:** stream tar to GCS → Cloud Run/Dataproc extraction job →
delete tar from GCS. Use when local disk is tight or dataset is > 10 GB.

### Pattern D — HTTP directory listing (no manifest)
Files individually accessible but no CSV — must scrape the index page to
discover filenames. Use `BeautifulSoup` to parse the listing, build a
synthetic manifest, then proceed as Pattern A.

### Pattern E — Authenticated download
Requires login, API key, or licence agreement (e.g. some speech/medical datasets).
Ask the user: *"This dataset requires authentication. Do you have credentials?
What form — API key, username/password, or licence acceptance?"*
Do not attempt download until auth method is confirmed.

### Pattern F — Hugging Face Hub
Use the `huggingface_hub` library:
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="org/dataset", repo_type="dataset",
                  local_dir="/tmp/hf_cache", ignore_patterns=["*.parquet"])
```
Then upload selected files to GCS and delete local cache.
Requires `uv add huggingface_hub`.

### Pattern G — Cloud-provider CLI (S3, other GCS buckets, Kaggle)
Use the provider's native CLI:
- S3: `aws s3 sync s3://bucket/prefix /tmp/local --exclude "*" --include "*.wav"`
- Kaggle: `kaggle datasets download -d org/dataset --path /tmp`
- Other GCS: `gcloud storage cp -r gs://src-bucket/prefix gs://rave-latents/dest/`

### None of the above
If the dataset doesn't fit any pattern above, stop and report:
> *"This dataset doesn't match a known download pattern. Here is what I found:
> [findings]. How would you like to proceed? Options: [list viable approaches]."*
Never guess or attempt a download with an unclassified pattern.

**Decision tree:**
```
Is there a metadata CSV with file paths?
  YES → Can you GET one file URL directly (not 404)?
          YES → Pattern A
          NO  → Is it a zip served from GCS/S3? → Pattern B
  NO  → Is it on Hugging Face? → Pattern F
      → Is it on S3/Kaggle? → Pattern G
      → Is it tar.gz bundles? → Pattern C (ask local vs GCS-native first)
      → Is it a directory listing? → Pattern D
      → Does it require login/key? → Pattern E
      → None match → stop and report
```

### Already classified

| Dataset | Pattern | Notes |
|---------|---------|-------|
| MAESTRO v3.0.0 | B | Separate audio zip + MIDI zip |
| E-GMD v1.0.0 | B | Combined zip (WAV + MIDI), ignore MIDI-only zip |
| CocoChorales | C | GitHub repo — use tiny shell script; ask local vs GCS-native |

---

## Step 4 — Confirm plan

Show this table and wait for explicit confirmation before generating code:

| Field | Value |
|-------|-------|
| Dataset | `$dataset_name` |
| Pattern | A / B / C1 / C2 / D / E / F / G |
| GCS prefix | `gs://rave-latents/$dataset_name/` |
| Files to download | N audio [+ N midi] |
| Estimated GB | X GB (within `$subset_size`) |
| Sampling method | stratified by `split` / random N rows |
| Local disk needed | X GB (Pattern C1 only) / none |
| Script path | `scripts/download_$dataset_name.py` |

---

## Step 5 — Generate script

Use the relevant template from [template.md](./template.md).

Critical rules — each fixed a real bug:

1. **ASCII only in `print()`** — non-ASCII crashes Windows cp1252 (pitfalls.md §1)
2. **Never download full archive** — Pattern B uses `_RangeFile` only (pitfalls.md §4)
3. **Strip zip prefix** with `_build_zip_index()`, verify first 3 entries vs CSV (pitfalls.md §5)
4. **Strip leading slashes**: `row[col].lstrip("/")` on every CSV path (pitfalls.md §13)
5. **Pandas sampling**: `GroupBy.sample()` not `groupby().apply()` (pitfalls.md §6)
6. **Always pass `project=`** to `storage.Client()` — wrong project silently misdirects
7. **Upload manifest before the file loop** — survives mid-run crashes
8. **8 MB streaming chunks** via `blob.open("wb")` — no temp files on disk
9. **`blob_exists()` before every upload** — full idempotency

---

## Step 6 — Smoke test

```bash
uv run python -u scripts/download_$dataset_name.py --limit 1
```

Required output:
```
GCS connection OK - bucket gs://rave-latents accessible.
[1/2] OK    gs://rave-latents/...
[2/2] OK    gs://rave-latents/...
```

Immediately re-run to verify idempotency:
```
[1/2] SKIP  gs://rave-latents/...
[2/2] SKIP  gs://rave-latents/...
```

Do not proceed to full download until both passes are clean.
Smoke test failure reference → pitfalls.md §smoke-test-failures

---

## Step 7 — Full download

```bash
uv run python -u scripts/download_$dataset_name.py
```

`-u` is required — stdout is buffered without it (pitfalls.md §7).

After completion, always run once more to catch transient SSL errors:
```bash
uv run python -u scripts/download_$dataset_name.py
```
Expect all SKIP and zero ERROR.

---

## Step 8 — Verify data landed in GCS

Run after every completed download. Full spec in tests.md §integration.

```bash
# Object count
gcloud storage ls --recursive "gs://rave-latents/$dataset_name/" | wc -l

# Total size
gcloud storage du --summarize "gs://rave-latents/$dataset_name/"

# Manifest present
gcloud storage ls "gs://rave-latents/$dataset_name/manifest.csv"

# Cross-check manifest rows vs object count
gcloud storage cat "gs://rave-latents/$dataset_name/manifest.csv" | wc -l
```

Expected: `(manifest_rows × file_columns) + 1` objects (the +1 is the manifest).
If count is lower: ERROR files not retried. Re-run the script.

---

## Completed datasets

| Dataset | Script | GCS prefix | Sample | Result |
|---------|--------|-----------|--------|--------|
| MAESTRO v3.0.0 | `download_maestro_sample.py` | `maestro-sample/` | 10% stratified | 202 OK / 54 SKIP / 0 ERR / 12.2 GB |
| E-GMD v1.0.0 | `download_e_gmd.py` | `e-gmd/` | 8% stratified | 7281 OK / 3 SSL ERR fixed on re-run / 10.3 GB |
| CocoChorales | `download_cocochorales.py` | `cocochorales-sample/` | tiny subset | pending |
