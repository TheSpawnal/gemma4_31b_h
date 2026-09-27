#!/usr/bin/env bash
# Launch llama-server for Gemma 4 31B QAT Q4_0.
# - loopback only, API key from a 0600 file
# - memory-capped transient systemd scope, no swap: an OOM kills the server, not the desktop
# - preflight: model hashes, free VRAM, GPU temp, RAM, port, cgroup delegation, flag support
# Env overrides: CTX THREADS PORT FIT_TARGET_MIB MEM_HIGH MEM_MAX VISION THINK VERIFY
set -euo pipefail
umask 077

ROOT="${ROOT:-$HOME/llm}"
BIN="${BIN:-$ROOT/llama.cpp/build/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/gemma-4-31b-it-qat-q4_0}"
MODEL="$MODEL_DIR/gemma-4-31B_q4_0-it.gguf"
MMPROJ="$MODEL_DIR/gemma-4-31B-it-mmproj.gguf"
KEYFILE="$ROOT/api.key"
HDRFILE="$ROOT/auth.hdr"
LOGDIR="$ROOT/logs"

HOST=127.0.0.1
PORT="${PORT:-8080}"
CTX="${CTX:-16384}"
THREADS="${THREADS:-8}"              # physical cores; decode is bandwidth-bound, SMT does not help
FIT_TARGET_MIB="${FIT_TARGET_MIB:-1536}" # VRAM left free for the desktop; 512 if display runs on the iGPU
MEM_HIGH="${MEM_HIGH:-18G}"
MEM_MAX="${MEM_MAX:-20G}"
VISION="${VISION:-0}"                # 1 = load mmproj, kept on CPU (no VRAM cost)
THINK="${THINK:-false}"              # server default; per-request override via chat_template_kwargs
VERIFY="${VERIFY:-1}"

MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-9000}"
MIN_AVAIL_RAM_MIB="${MIN_AVAIL_RAM_MIB:-16000}"
MAX_START_GPU_C="${MAX_START_GPU_C:-65}"

die() { echo "abort: $*" >&2; exit 1; }

# ---- input validation
[[ "$THINK" == true || "$THINK" == false ]] || die "THINK must be true|false"
[[ "$VISION" == 0 || "$VISION" == 1 ]] || die "VISION must be 0|1"
for v in PORT CTX THREADS FIT_TARGET_MIB MIN_FREE_VRAM_MIB MIN_AVAIL_RAM_MIB MAX_START_GPU_C; do
    [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v must be an integer"
done
[[ "$MEM_HIGH" =~ ^[0-9]+[KMG]$ && "$MEM_MAX" =~ ^[0-9]+[KMG]$ ]] || die "MEM_HIGH/MEM_MAX like 18G"

# ---- tools and files
for c in nvidia-smi systemd-run ss sha256sum; do command -v "$c" >/dev/null || die "$c not found"; done
[[ -x "$BIN" ]] || die "llama-server not found: $BIN"
[[ -r "$MODEL" ]] || die "model not found: $MODEL"
[[ "$VISION" == 0 || -r "$MMPROJ" ]] || die "mmproj not found: $MMPROJ"

# ---- flag support of this llama.cpp build
help=$("$BIN" --help 2>&1 || true)
has() { grep -qF -- "$1" <<< "$help"; }
for f in --fit-target --no-mmap --api-key-file --chat-template-kwargs --metrics --jinja; do
    has "$f" || die "this llama-server build lacks $f (update llama.cpp)"
done
if [[ "$VISION" == 1 ]]; then
    has --no-mmproj-offload || die "this llama-server build lacks --no-mmproj-offload"
fi

# ---- integrity
if [[ "$VERIFY" == 1 ]]; then
    [[ -r "$MODEL_DIR/SHA256SUMS" ]] || die "no SHA256SUMS in $MODEL_DIR (run fetch-model.sh)"
    echo "verify: sha256"
    (cd "$MODEL_DIR" && sha256sum --quiet --strict -c SHA256SUMS) || die "model hash mismatch"
fi

# ---- host state
[[ -z "$(ss -Hltn "sport = :$PORT")" ]] || die "port $PORT already in use"

IFS=', ' read -r vram_free gpu_temp < <(nvidia-smi -i 0 \
    --query-gpu=memory.free,temperature.gpu --format=csv,noheader,nounits)
[[ "${vram_free:-}" =~ ^[0-9]+$ && "${gpu_temp:-}" =~ ^[0-9]+$ ]] || die "cannot read GPU state"
(( vram_free >= MIN_FREE_VRAM_MIB )) || die "${vram_free} MiB VRAM free < ${MIN_FREE_VRAM_MIB} (close games / GPU apps)"
(( gpu_temp <= MAX_START_GPU_C )) || die "GPU already at ${gpu_temp}C"
others=$(nvidia-smi -i 0 --query-compute-apps=pid,process_name --format=csv,noheader || true)
[[ -z "$others" ]] || echo "note: other CUDA processes on GPU: $others"

avail_mib=$(( $(awk '/^MemAvailable:/{print $2}' /proc/meminfo) / 1024 ))
(( avail_mib >= MIN_AVAIL_RAM_MIB )) || die "${avail_mib} MiB RAM available < ${MIN_AVAIL_RAM_MIB}"

uid=$(id -u)
ctl="/sys/fs/cgroup/user.slice/user-${uid}.slice/user@${uid}.service/cgroup.controllers"
{ [[ -r "$ctl" ]] && grep -qw memory "$ctl"; } || die "memory controller not delegated to user systemd (MemoryMax would be ignored)"

# ---- credentials
mkdir -p "$LOGDIR"
if [[ ! -s "$KEYFILE" ]]; then
    head -c 32 /dev/urandom | od -An -tx1 -v | tr -d ' \n' > "$KEYFILE"
    echo >> "$KEYFILE"
    echo "key:    generated $KEYFILE"
fi
[[ "$(stat -c %a "$KEYFILE")" =~ ^[46]00$ ]] || die "$KEYFILE must be mode 600 or 400"
key=""
IFS= read -r key < "$KEYFILE" || [[ -n "$key" ]] || die "empty key file"
printf 'Authorization: Bearer %s\n' "$key" > "$HDRFILE"
unset key

# ---- server arguments
LOG="$LOGDIR/llama-server-$(date +%Y%m%dT%H%M%S).log"
args=(
    -m "$MODEL" --alias gemma-4-31b-it-qat
    --host "$HOST" --port "$PORT" --api-key-file "$KEYFILE"
    -c "$CTX" -np 1
    -t "$THREADS" -tb "$THREADS"
    -fa on -ctk q8_0 -ctv q8_0
    --fit on --fit-target "$FIT_TARGET_MIB"
    --no-mmap
    --jinja --chat-template-kwargs "{\"enable_thinking\":$THINK}"
    --temp 1.0 --top-p 0.95 --top-k 64
    --metrics
)
has --log-file && args+=(--log-file "$LOG")
has --log-timestamps && args+=(--log-timestamps)
[[ "$VISION" == 1 ]] && args+=(--mmproj "$MMPROJ" --no-mmproj-offload)

echo "start:  ctx $CTX thr $THREADS fit-target ${FIT_TARGET_MIB}M mem ${MEM_HIGH}/${MEM_MAX} swap 0 vision $VISION think $THINK"
echo "host:   vram free ${vram_free}M gpu ${gpu_temp}C ram avail ${avail_mib}M"
echo "api:    http://$HOST:$PORT  (curl -H @$HDRFILE)"
echo "log:    $LOG"

exec systemd-run --user --scope --quiet --collect --unit="llm-gemma31b-$PORT" \
    -p MemoryHigh="$MEM_HIGH" -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 \
    -- "$BIN" "${args[@]}"
