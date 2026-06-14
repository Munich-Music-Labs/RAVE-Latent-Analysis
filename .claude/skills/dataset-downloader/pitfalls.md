# Pitfalls reference

Load this file when debugging a failing download or generating a new script.
Every entry here caused a real failure on MAESTRO or E-GMD.

---

## §1 — Unicode crashes Windows terminals (cp1252)

**Symptom:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '→' in position 9
```
Windows terminals default to cp1252. Any non-ASCII in `print()` crashes immediately.

**Rule:** ASCII only in every `print()`. Replace `→` with `->`, `—` with `-`.

---

## §2 — `cp` via Bash tool silently fails on Windows paths

**Symptom:** `cp "C:\...\file.md" "C:\...\dest\"` exits 0, file not copied.

**Rule:** Use PowerShell `Copy-Item` for all file operations on Windows absolute paths.

```powershell
Copy-Item "C:\source\file.md" "C:\dest\file.md"
```

---

## §3 — Bash tool rejects PowerShell syntax

**Symptom:**
```
/usr/bin/bash: eval: line 1: syntax error near unexpected token '('
```
**Rule:** Use the PowerShell tool for any command with `$variable`, `Get-Content`,
`Select-String`, or PowerShell pipeline syntax.

---

## §4 — Individual file URLs return 404 from zip-only sources

**Symptom:** URL constructed from CSV path returns 404. Files only exist inside the archive.

**Rule:** Always fetch one sample file URL before writing the download loop.
If 404 → switch to Pattern B (`_RangeFile`).

---

## §5 — Zip entry names have a top-level dir prefix that CSV paths don't

**Symptom:** `zip_index["drummer1/session1/file.wav"]` → `None` → `KeyError`.

**Why:** Zip entry is `e-gmd-v1.0.0/drummer1/session1/file.wav`; CSV path is
`drummer1/session1/file.wav`. Direct lookup misses every file.

**Rule:** Always use `_build_zip_index()`. After building, verify:
```python
print(list(zip_index.keys())[:3])
print(df["audio_filename"].head(3).tolist())
```
Both should match structurally before starting the loop.

```python
def _build_zip_index(zf: zipfile.ZipFile) -> dict[str, str]:
    index = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = info.filename.split("/", 1)
        key = parts[1] if len(parts) == 2 else parts[0]
        index[key] = info.filename
    return index
```

---

## §6 — `groupby().apply()` silently drops the groupby column (pandas >= 3.0)

**Symptom:** `KeyError: 'split'` downstream even though column exists in original df.

**Rule:**
```python
# WRONG — drops 'split' column in pandas 3.0+
df.groupby("split").apply(lambda g: g.sample(frac=0.1, random_state=42))

# CORRECT
df.groupby("split", group_keys=False).sample(frac=0.1, random_state=42).reset_index(drop=True)
```

---

## §7 — Missing `-u` makes background runs produce no output

**Symptom:** Output file stays empty; script appears frozen.

**Why:** Without `-u`, Python fully buffers stdout for non-TTY processes.

**Rule:** Always `uv run python -u scripts/download_<name>.py`. No exceptions.

---

## §8 — `blob_exists()` crashes run if network drops mid-check

**Symptom:**
```
google.api_core.exceptions.RetryError: Timeout of 120.0s exceeded
```
**Prevention:** Check connectivity before starting:
```bash
curl -s --max-time 10 -o /dev/null -w "%{http_code}" https://storage.googleapis.com
```
Expect `400`. `000` or timeout = do not start.

**Known gap:** `blob_exists()` has no try/except in current scripts. Future fix:
catch `RetryError` at per-file level, treat as transient ERROR, continue loop.

---

## §9 — Transient SSL errors drop files silently (fixed by re-run)

**Symptom:**
```
SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol')
```
3 out of 7,284 E-GMD files failed this way. Script caught the exception, printed
ERROR, continued. Files were missing from GCS after the run.

**Rule:** After any run with ERROR lines, always do one cleanup re-run.
Script is idempotent — SKIPs all uploaded files, retries only missing ones.

---

## §10 — Smoke test SKIPs look wrong but are correct

**Symptom:** Full run after `--limit 1` shows `SKIP: 2` at the start.

**Why:** `--limit 1` uploaded 2 files. Full run correctly SKIPs them. Not a bug.

---

## §11 — `uv` VIRTUAL_ENV mismatch warning (non-fatal)

**Symptom:**
```
warning: `VIRTUAL_ENV=...` does not match the project environment path `.venv`
```
**Rule:** Ignore. Use `uv run --active` to suppress if it's distracting.

---

## §12 — VS Code Pylance import warnings (non-fatal)

**Symptom:** `Import "pandas" could not be resolved from source`

**Rule:** Ignore. Script runs correctly. To silence: point Pylance at the correct venv.

---

## §13 — Leading slashes in CSV paths cause double-slash GCS paths

**Symptom:** GCS path becomes `gs://rave-latents/prefix//2008/file.wav`.

**Rule:** Always strip:
```python
src_filename = row[col].lstrip("/")
gcs_path = f"{GCS_PREFIX}/{src_filename}"
```

---

## §14 — `gsutil du` fails on Windows

**Symptom:** `ERROR: (gsutil) python3.12: command not found`

**Rule:** Use `gcloud storage du --summarize gs://rave-latents` instead.

---

## §15 — E-GMD: combined zip; MAESTRO: separate audio and MIDI zips

**Structural note:**
- MAESTRO: `maestro-v3.0.0.zip` (audio) + `maestro-v3.0.0-midi.zip` (MIDI separate)
- E-GMD: `e-gmd-v1.0.0.zip` (WAV + MIDI both inside) — use this only, ignore MIDI-only zip

**Rule:** Before writing URL constants, verify whether audio and MIDI are bundled or split.

---

## §16 — Pattern C local disk exhaustion

**Symptom:** Extraction of tar fails mid-way with `No space left on device`.

**Prevention:** Before any Pattern C download, calculate tar size and confirm with user.
Offer GCS-native extraction (Pattern C2) when local disk is < 2× the tar size.

For C2, the extraction job outline:
```bash
# Stream tar to GCS (no local storage)
curl -L <tar_url> | gcloud storage cp - gs://rave-latents/<name>/<archive>.tar.gz

# Extract via Cloud Run job (or Dataproc if cluster already exists)
# Then delete the tar from GCS
gcloud storage rm gs://rave-latents/<name>/<archive>.tar.gz
```

---

## Smoke test failure reference {#smoke-test-failures}

| Error | Cause | Fix |
|-------|-------|-----|
| `404` on metadata CSV | Wrong URL | Web-search correct URL |
| `KeyError: 'split'` | No split col or pandas 3.0 bug | Check columns; use `GroupBy.sample()` — §6 |
| `KeyError: 'path/file.wav'` in zip | Top-level dir not stripped | Print `zf.infolist()[0].filename`; fix `_build_zip_index` — §5 |
| `KeyError` + leading slash | CSV path has leading `/` | Add `.lstrip("/")` — §13 |
| `KeyError: 'Content-Length'` | Server omits header on HEAD | Try GET with `stream=True` to get size |
| `403 Forbidden` on bucket | Wrong project or bucket name | Verify `GCP_PROJECT` and `BUCKET_NAME` |
| No output at all | Missing `-u` flag | Kill, restart with `uv run python -u ...` — §7 |
| `No space left on device` | Local disk full (Pattern C) | Switch to Pattern C2 (GCS-native) — §16 |
| `VIRTUAL_ENV` warning | uv env conflict | Ignore or use `--active` — §11 |
