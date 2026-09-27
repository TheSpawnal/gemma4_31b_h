#!/usr/bin/env bash
# Turn coder3101/gemma-4-31B-it-heretic (BF16 safetensors) into a quantized GGUF
# that fits falkor, under your own control:
#   1. download the repo at a pinned commit
#   2. verify every large file against the SHA-256 in the Hub's LFS metadata
#   3. convert safetensors -> GGUF with llama.cpp's converter (text tower only)
#   4. quantize to the target type(s)
#   5. self-hash the OUTPUT (there is no upstream GGUF hash to trust) and lock it
#
# This is the honest path for a third-party weight edit: one upstream party,
# a quant recipe you chose, and a hash of the artifact you actually run.
#
# Env: REV QUANTS OUTTYPE KEEP_SRC KEEP_INTERMEDIATE MIN_FREE_GB VENV LLAMACPP ROOT
set -euo pipefail
umask 077
export HF_HUB_DISABLE_TELEMETRY=1

REPO="coder3101/gemma-4-31B-it-heretic"
REV="${REV:-9a1c7f5d851541f279cd104329e093cb363cd9e7}"   # pinned commit
ROOT="${ROOT:-$HOME/llm}"
VENV="${VENV:-$ROOT/venv}"
LLAMACPP="${LLAMACPP:-$ROOT/llama.cpp}"
SLUG="gemma-4-31b-it-heretic"
SRC="$ROOT/src/$SLUG"
OUT="$ROOT/models/$SLUG"

OUTTYPE="${OUTTYPE:-bf16}"            # conversion intermediate: bf16 (best) or q8_0 (smaller)
QUANTS="${QUANTS:-Q4_K_M}"           # space-separated: e.g. "Q4_K_M Q5_K_M"
KEEP_SRC="${KEEP_SRC:-0}"            # keep the 62 GB safetensors after conversion
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"   # keep the bf16/q8_0 GGUF after quantizing
MIN_FREE_GB="${MIN_FREE_GB:-150}"    # bf16 path keeps 62+62+18; q8_0 path needs ~95

# large files worth hash-verifying (the rest are tiny configs)
LFS_FILES=("model-00001-of-00002.safetensors" "model-00002-of-00002.safetensors" "tokenizer.json")
# everything the converter needs
ALL_FILES=("${LFS_FILES[@]}" "config.json" "generation_config.json"
    "model.safetensors.index.json" "preprocessor_config.json"
    "tokenizer_config.json" "chat_template.jinja")

HF="$VENV/bin/hf"
PY="$VENV/bin/python"
CONV="$LLAMACPP/convert_hf_to_gguf.py"
QUANT="$LLAMACPP/build/bin/llama-quantize"
die() { echo "error: $*" >&2; exit 1; }

case "$OUTTYPE" in bf16|q8_0) ;; *) die "OUTTYPE must be bf16 or q8_0";; esac
[[ -x "$HF" && -x "$PY" ]] || die "venv missing (python3 -m venv $VENV; pip install -U huggingface_hub -r $LLAMACPP/requirements.txt)"
[[ -f "$CONV" ]] || die "converter not found: $CONV"
[[ -x "$QUANT" ]] || die "llama-quantize not built: $QUANT"
"$PY" -c 'import gguf' 2>/dev/null || die "python 'gguf' missing (pip install -r $LLAMACPP/requirements.txt)"
command -v sha256sum >/dev/null || die "sha256sum not found"

mkdir -p "$SRC" "$OUT"
free_gb=$(df -BG --output=avail "$OUT" | tail -n1 | tr -dc '0-9')
(( free_gb >= MIN_FREE_GB )) || die "${free_gb}G free under $ROOT, need >= ${MIN_FREE_GB}G (set MIN_FREE_GB to override)"

# 1. expected hashes at the pinned revision (TLS only)
manifest=$(curl -fsS --proto '=https' --tlsv1.2 --max-time 30 \
    "https://huggingface.co/api/models/${REPO}/tree/${REV}?recursive=1") || die "cannot fetch Hub manifest"
expected=$(printf '%s' "$manifest" | "$PY" -c '
import json, sys
want = set(sys.argv[1:]); found = {}
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
' "${LFS_FILES[@]}") || die "manifest check failed"

# 2. download exactly these files at exactly this revision
echo "download: $REPO @ ${REV:0:12} (~63 GB)"
"$HF" download "$REPO" "${ALL_FILES[@]}" --revision "$REV" --local-dir "$SRC"

# 3. verify big files
echo "verify: sha256 of large files"
cd "$SRC"; fail=0
while read -r sha size path; do
    [[ -f "$path" ]] || { echo "FAIL missing $path"; fail=1; continue; }
    [[ "$(stat -c %s -- "$path")" == "$size" ]] || { echo "FAIL size $path"; fail=1; continue; }
    [[ "$(sha256sum -- "$path" | cut -d' ' -f1)" == "$sha" ]] \
        && echo "OK   $path" || { echo "FAIL sha256 $path"; fail=1; }
done <<< "$expected"
(( fail == 0 )) || die "source verification failed - delete $SRC and re-run"

# 4. convert (text tower; vision tensors are skipped for a text-only GGUF)
inter="$OUT/${SLUG}-${OUTTYPE}.gguf"
if [[ -f "$inter" ]]; then
    echo "convert: reuse existing $inter"
else
    echo "convert: safetensors -> $OUTTYPE GGUF"
    "$PY" "$CONV" "$SRC" --outfile "$inter" --outtype "$OUTTYPE"
fi
[[ "$KEEP_SRC" == 1 ]] || { echo "cleanup: removing safetensors"; rm -rf "$SRC"; }

# 5. quantize + self-hash each target
cd "$OUT"; : > SHA256SUMS.tmp
for q in $QUANTS; do
    final="$OUT/${SLUG}-${q}.gguf"
    echo "quantize: $OUTTYPE -> $q"
    "$QUANT" "$inter" "$final" "$q"
    sha=$(sha256sum -- "$(basename "$final")" | cut -d' ' -f1)
    printf '%s  %s\n' "$sha" "$(basename "$final")" >> SHA256SUMS.tmp
    echo "OK   $q  $sha"
done
[[ "$KEEP_INTERMEDIATE" == 1 ]] || rm -f "$inter"

mv -f SHA256SUMS.tmp SHA256SUMS
{ echo "source: ${REPO}@${REV}"
  echo "recipe: convert_hf_to_gguf.py --outtype ${OUTTYPE} | llama-quantize -> ${QUANTS}"
  echo "built:  $(date -u +%Y-%m-%dT%H:%M:%SZ) by convert-model.sh"
} > SOURCE
chmod 0400 SHA256SUMS SOURCE ./*.gguf 2>/dev/null || true
echo "done: $OUT"
ls -lh "$OUT"/*.gguf
