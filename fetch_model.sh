#!/usr/bin/env bash
# Download google/gemma-4-31B-it-qat-q4_0-gguf at a pinned commit and verify
# every file against the SHA-256 published in the Hub's LFS metadata.
# Usage: fetch-model.sh [dest_dir]        (VISION=0 to skip the 1.2 GB mmproj)
set -euo pipefail
umask 077
export HF_HUB_DISABLE_TELEMETRY=1

REPO="google/gemma-4-31B-it-qat-q4_0-gguf"
REV="59dde24573e7e61570dba08b18a2e1fe246955ed"
DEST="${1:-$HOME/llm/models/gemma-4-31b-it-qat-q4_0}"
VENV="${VENV:-$HOME/llm/venv}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"
FILES=("gemma-4-31B_q4_0-it.gguf")
[[ "${VISION:-1}" == 1 ]] && FILES+=("gemma-4-31B-it-mmproj.gguf")

HF="$VENV/bin/hf"
PY="$VENV/bin/python"
die() { echo "error: $*" >&2; exit 1; }
[[ -x "$HF" && -x "$PY" ]] || die "venv missing at $VENV (python3 -m venv $VENV; $VENV/bin/pip install -U huggingface_hub)"
command -v sha256sum >/dev/null || die "sha256sum not found"

mkdir -p "$DEST"
free_gb=$(df -BG --output=avail "$DEST" | tail -n1 | tr -dc '0-9')
(( free_gb >= MIN_FREE_GB )) || die "${free_gb}G free under $DEST, need ${MIN_FREE_GB}G"

# 1. expected hashes from the Hub API, at the pinned revision (TLS only)
manifest=$(curl -fsS --proto '=https' --tlsv1.2 --max-time 30 \
    "https://huggingface.co/api/models/${REPO}/tree/${REV}") || die "cannot fetch Hub manifest"

expected=$(printf '%s' "$manifest" | "$PY" -c '
import json, sys
want = set(sys.argv[1:])
found = {}
for e in json.load(sys.stdin):
    p = e.get("path")
    if p in want:
        lfs = e.get("lfs") or {}
        oid, size = lfs.get("oid", ""), lfs.get("size")
        if len(oid) != 64 or not isinstance(size, int):
            sys.exit(f"no sha256/size in Hub metadata for {p}")
        found[p] = (oid, size)
missing = want - found.keys()
if missing:
    sys.exit("absent at pinned revision: " + ", ".join(sorted(missing)))
for p in sorted(found):
    print(found[p][0], found[p][1], p)
' "${FILES[@]}") || die "manifest check failed"

# 2. download exactly these files at exactly this revision
"$HF" download "$REPO" "${FILES[@]}" --revision "$REV" --local-dir "$DEST"

# 3. verify size + sha256, then lock the files read-only
cd "$DEST"
fail=0
: > SHA256SUMS.tmp
while read -r sha size path; do
    [[ -f "$path" ]] || { echo "FAIL missing $path"; fail=1; continue; }
    got_size=$(stat -c %s -- "$path")
    [[ "$got_size" == "$size" ]] || { echo "FAIL size $path $got_size != $size"; fail=1; continue; }
    got=$(sha256sum -- "$path" | cut -d' ' -f1)
    if [[ "$got" == "$sha" ]]; then
        echo "OK   $sha  $path"
        printf '%s  %s\n' "$sha" "$path" >> SHA256SUMS.tmp
    else
        echo "FAIL sha256 $path"
        fail=1
    fi
done <<< "$expected"

if (( fail )); then
    rm -f SHA256SUMS.tmp
    die "verification failed - do not load these files; delete and re-run"
fi
mv -f SHA256SUMS.tmp SHA256SUMS
printf '%s@%s\n' "$REPO" "$REV" > SOURCE
chmod 0400 "${FILES[@]}" SHA256SUMS SOURCE
echo "verified: $DEST"
