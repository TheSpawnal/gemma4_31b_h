# Local LLM Deployment Protocol — Gemma 4 31B Heretic on RTX 5070

Host: `falkor` (Ryzen 7 9700X, RTX 5070 12 GB, 64 GB DDR5-6000, Ubuntu 26.04)
Model: `coder3101/gemma-4-31B-it-heretic` 
Scripts: `convert-model.sh`, `llm-serve.sh`, `llmwatch.py`

---

## What this model is (specs from the page)

This is not a ready-to-run model. It is a **weight edit** of Google's stock instruction-tuned model, published as full-precision safetensors. Read this section before touching the hardware — it changes the pipeline and the threat model.

### Lineage and architecture (inherited from the base, unchanged)

Base chain: `google/gemma-4-31B` → `google/gemma-4-31B-it` → this abliteration. The architecture is the stock Gemma 4 31B dense model:

| Property | Value |
| --- | --- |
| Total parameters | 30.7B |
| Layers | 60 |
| Sliding window | 1024 tokens |
| Context length | 256K tokens |
| Vocabulary | 262K |
| Attention | hybrid local/global, final layer global, unified KV + p-RoPE on global layers |
| Modalities | text, image (no audio on 31B) |

### Distribution format — the part that matters for deployment

| Property | Value |
| --- | --- |
| Format | Safetensors (2 shards: 49.9 GB + 12.6 GB = **62.5 GB**) |
| Tensor type | **BF16** |
| GGUF provided? | **No** |

Two consequences:

1. **Safetensors is a security plus.** It stores tensors only — no executable code, unlike Python pickle (`.bin`) weights. Downloading it cannot run code on your box. That is why the pipeline below tolerates a third-party file at all.
2. **BF16 will not run on falkor, in any configuration.** 30.7B params x 2 bytes = 62.5 GB of weights. Your total memory is 12 GB VRAM + 32 GB RAM = 44 GB. The model does not fit even fully offloaded to the CPU. You **must** convert to GGUF and quantize before it can run. That is Phase 3.

### What "heretic" means

The card states: a decensored version of `google/gemma-4-31B-it`, made with **Heretic v1.2.0** using the **Arbitrary-Rank Ablation (ARA)** method with row-norm preservation.

Abliteration is a directional weight edit, not a finetune. It identifies the direction(s) in the residual stream that correlate with refusal (estimated from paired harmful/harmless prompts) and projects those directions out of the layers' write matrices, so the model stops emitting refusals. Row-norm preservation keeps each edited weight row's magnitude, which limits how far the rest of the model's statistics drift — that is why the KL divergence stays low.

Reported abliteration parameters, and what each means:

| Parameter | Value | Reading |
| --- | --- | --- |
| start_layer_index | 1 | edit begins at decoder layer 1 |
| end_layer_index | 59 | edit runs to layer 59 — nearly all 60 layers |
| preserve_good_behavior_weight | 0.8438 | high weight on keeping benign behavior intact (conservative edit) |
| steer_bad_behavior_weight | 0.0002 | essentially zero — it *removes* refusal, it does not *inject* harmful steering |
| overcorrect_relative_weight | 1.0760 | slight overcorrection factor |
| neighbor_count | 15 | kNN size used to estimate the refusal direction |

That `steer_bad_behavior_weight ~ 0` is the important nuance: the model was not trained toward harmful output. It had its ability to *decline* ablated. The difference matters for how you evaluate it.

### Reported performance

| Metric | This model | Base `gemma-4-31B-it` | Reading |
| --- | --- | --- | --- |
| KL divergence | 0.0434 | 0 (by definition) | token distribution stays very close to base — capability loss is small |
| Refusals | 15/100 | 99/100 | base refused 99% of a harmful test set; this refuses 15% |

So it is substantially decensored but not fully compliant, and it is still "mostly the base model" in raw capability. Ecosystem: served by Featherless AI; 13 community GGUF quants, 3 adapters, 3 finetunes exist; ~7,900 downloads last month.

### Honest note before you build

Standing up inference infrastructure is neutral engineering, and an uncensored local model is a legitimate instrument for red-team and alignment research. Two things are yours to own, not the tooling's:

- **It will comply with requests the base refuses, including genuinely harmful ones.** Keep it isolated: loopback only, no tool/agent/shell wiring, and treat every output as untrusted. The launcher enforces the network isolation; the rest is operational discipline.
- **The weights were modified by an individual.** Safetensors means no code execution, and KL 0.043 says "mostly base," but that is not proof the *only* change was refusal removal. Phase 6 includes a behavioral diff against the official base so you verify that yourself rather than trusting the label.

---

## Speed and memory on falkor

Each generated token reads every weight once, so decode speed is set by memory bandwidth.

| Half | Size (Q4_K_M) | Bandwidth | Time per token |
| --- | --- | --- | --- |
| GPU | ~8.5–9 GB | 672 GB/s | ~14 ms |
| CPU | ~9–10 GB | ~55–60 GB/s effective | ~150–170 ms |

The 9700X is a single-CCD chip; its CCD-to-IO-die read link (~32 B/clock, ~64 GB/s peak at FCLK 2000) is the real ceiling for the CPU half, not DDR5-6000's nominal 96 GB/s.

- **Realistic:** 4–5 tokens/s.
- Each layer moved onto the GPU (~0.25–0.3 GB) saves ~4–5 ms/token.
- Thinking mode can emit ~1,500 hidden tokens — 5–6 minutes before the first visible word. The launcher keeps thinking **off by default**; enable it per request.

### Which quant

There is **no QAT build** for this model (abliteration was done on BF16), so you cannot match the quality-per-bit of the official QAT GGUF. Pick from standard k-quants:

| Quant | Size | Notes |
| --- | --- | --- |
| **Q4_K_M** | ~18.5 GB | default. Best speed/quality balance. |
| **Q5_K_M** | ~21.7 GB | preserves the (already perturbed) weights more faithfully; the ~44 GB budget holds it fine, at slightly lower t/s. |

Because abliteration has already moved the weights, aggressive 4-bit quantization compounds the distortion. If you notice degraded coherence at Q4_K_M, step up to Q5_K_M before blaming the abliteration.

---

## Phase 0 — Hardware baseline (before any model)

```bash
sudo apt update
sudo apt install -y lm-sensors nvme-cli smartmontools rasdaemon stress-ng
sudo systemctl enable --now rasdaemon

nvidia-smi                                   # header "CUDA Version" must be >= 13.0 (R580+)
cat /proc/driver/nvidia/version              # must say "Open Kernel Module"
nvidia-smi -q -d TEMPERATURE,POWER | grep -Ei 'slowdown|shutdown|target|limit'
sudo lspci -vv -d 10de: | grep -E 'LnkCap:|LnkSta:'
sensors
sudo nvme smart-log /dev/nvme0 | grep -Ei 'temperature|critical_warning|media_errors|percentage_used'
sudo ras-mc-ctl --summary
journalctl -k -b | grep -Ei 'xid|mce|machine check|hardware error' || echo "kernel log clean"
df -h ~
```

What to check:

- **PCIe link.** `LnkSta` usually drops to a lower speed at idle (normal power saving). The width must be x16.
- **Sensors.** `sensors` should list `k10temp` (Tctl, Tccd1) and `nvme`. If it also lists `spd5118`, that is your per-DIMM DDR5 temperature; if missing, try `sudo modprobe spd5118`, else the board does not expose it — skip.
- **Disk.** This model needs a lot of transient space (see Phase 3). Check `df -h ~` shows enough before starting.

### CPU stability matters more here than usual

llama.cpp's CPU layers on Zen 5 run AVX-512 dot-product kernels — the kind of load where a marginal Curve Optimizer offset produces silent wrong results or machine-check errors instead of clean crashes. If per-core CO validation is still pending, do it now. CPUs 0–7 are one thread per physical core:

```bash
for c in 0 1 2 3 4 5 6 7; do taskset -c $c stress-ng --cpu 1 --cpu-method fft --verify -t 3m --metrics-brief; done
stress-ng --cpu 8 --cpu-method matrixprod --verify -t 20m --metrics-brief
sudo ras-mc-ctl --errors
```

For a true AVX-512 torture test, y-cruncher's stress mode (from numberworld.org) is the reference. Any MCE means backing the CO off before trusting the model.

---

## Phase 1 — Toolchain

The NVIDIA driver already comes from Ubuntu, so get the toolkit from Ubuntu too. The Ubuntu and NVIDIA-repository install methods cannot be combined; mixing them breaks library paths and `nvcc` lookups.

```bash
sudo apt install -y build-essential cmake ninja-build git pkg-config python3-venv nvidia-cuda-toolkit
nvcc --version
```

The "CUDA Version" in the `nvidia-smi` header should be at least the `nvcc` release. If older, update the driver through Ubuntu (`sudo ubuntu-drivers list`, install the newest `-open` package). If `nvcc` rejects your GCC, add `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14` to the configure step below.

---

## Phase 2 — Build llama.cpp and the conversion venv

The conversion step needs both the built binaries and llama.cpp's Python converter, so a recent checkout that supports the `gemma4` architecture is required.

```bash
mkdir -p ~/llm && cd ~/llm
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
TAG=$(git tag -l 'b*' --sort=-v:refname | head -n1)
git -c advice.detachedHead=false checkout "$TAG"
git rev-parse HEAD | tee ../llama.cpp.pinned

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j 8 --target llama-server llama-cli llama-bench llama-quantize
./build/bin/llama-server --version
./build/bin/llama-server --list-devices      # expect RTX 5070, compute capability 12.0

# conversion venv: hub client + the converter's Python deps
python3 -m venv ~/llm/venv
~/llm/venv/bin/pip install -U huggingface_hub nvidia-ml-py psutil
~/llm/venv/bin/pip install -r ~/llm/llama.cpp/requirements.txt
~/llm/venv/bin/python -c 'import gguf; print("gguf ok")'
```

Why these options:

- **`-j 8`, not 16.** Parallel `nvcc` jobs each use 1–2 GB of RAM; 16 can OOM a 32 GB box mid-build.
- **`GGML_NATIVE=ON`** builds for your exact CPU (AVX-512) and auto-detects the GPU arch.
- **`llama-quantize` target added** — Phase 3 needs it.
- **`LLAMA_CURL=OFF`** removes the built-in downloader so models arrive only through the verified pipeline.

---

## Phase 3 — Convert and quantize (the heart of this variant)

Because the repo ships BF16 safetensors and no GGUF, you build the runnable artifact yourself. This is also the more secure path: **one** upstream party, a quantization recipe you chose, and a hash of the file you actually run.

```bash
mkdir -p ~/llm/bin && cp convert-model.sh llm-serve.sh llmwatch.py ~/llm/bin/ && chmod 700 ~/llm/bin/*
~/llm/bin/convert-model.sh                       # default: Q4_K_M
# or both quants:      QUANTS="Q4_K_M Q5_K_M" ~/llm/bin/convert-model.sh
```

`convert-model.sh`:

1. Pins the download to commit `9a1c7f5`.
2. Fetches expected SHA-256 + size of the large files (both safetensors shards and `tokenizer.json`) from the Hub API at that commit.
3. Downloads the shards and config/tokenizer files at that revision.
4. Verifies size and hash of the large files; aborts on any mismatch.
5. Converts safetensors → GGUF (`convert_hf_to_gguf.py --outtype bf16`), text tower only.
6. Quantizes to the target type(s) with `llama-quantize`.
7. **Self-hashes the output** (there is no upstream GGUF hash to trust), writes `SHA256SUMS` and `SOURCE` (repo@commit + recipe), and locks the GGUFs read-only.

### Disk budget

The BF16 path is disk-heavy. Default preflight requires **150 GB** free under `~/llm`:

| Stage | On disk |
| --- | --- |
| safetensors download | 62.5 GB |
| + bf16 GGUF (during conversion) | +62.5 GB → ~125 GB peak |
| safetensors deleted, then quantize | 62.5 GB + 18.5 GB → ~81 GB |
| final (Q4_K_M) | ~18.5 GB |

To cut the peak to ~95 GB on a smaller drive, use a Q8_0 intermediate (near-lossless): `OUTTYPE=q8_0 ~/llm/bin/convert-model.sh`. Keep the safetensors with `KEEP_SRC=1`, keep the intermediate with `KEEP_INTERMEDIATE=1`, override the check with `MIN_FREE_GB=...`.

Keep `llmwatch.py` running during the download and conversion to watch NVMe temperature over the sustained multi-GB writes.

### Auth

The license is Apache 2.0, so no token should be needed. On a 401, create a read-only fine-grained token, `hf auth login`, then `hf auth logout` afterwards.

### The community-GGUF shortcut (and why it is second choice)

There are 13 community quants of this model. Using one skips Phase 3 but adds a **second** untrusted party (whoever quantized it) on top of the abliteration author, and you would be hash-verifying transport of someone else's artifact, not content you produced. If you take that route anyway, download at a pinned commit, verify against that repo's own hashes, and still run the Phase 6 behavioral diff.

---

## Phase 4 — One-time guardrails (root)

### Power limit (optional, cheap insurance)

Decode is bandwidth-bound, so a power cap barely changes speed; it trims prompt-processing bursts and heat. Read the range first, then pick a value inside it:

```bash
nvidia-smi -q -d POWER | grep -Ei 'default|min|max|current'
sudo systemctl enable --now nvidia-persistenced
sudo tee /etc/systemd/system/nvidia-powerlimit.service >/dev/null <<'EOF'
[Unit]
Description=NVIDIA GPU power limit
Wants=nvidia-persistenced.service
After=nvidia-persistenced.service
[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -i 0 -pl 200
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now nvidia-powerlimit.service
nvidia-smi --query-gpu=power.limit --format=csv
```

Without `nvidia-persistenced` the driver can drop the power limit when the last GPU client exits. If the daemon is absent, add `ExecStart=/usr/bin/nvidia-smi -i 0 -pm 1` before the `-pl` line. At 250 W the 5070's 12V-2x6 connector runs modest current per pin — still confirm it is fully seated with no sharp bend.

### Firewall

```bash
sudo ufw default deny incoming && sudo ufw default allow outgoing && sudo ufw enable && sudo ufw status verbose
```

The server binds 127.0.0.1 regardless; UFW is the second wall. Never `--host 0.0.0.0`. For remote use, tunnel: `ssh -L 8080:127.0.0.1:8080 falkor`. For a decensored model this isolation is not optional — it is the containment boundary.

### Optional VRAM gain

Drive the display from the iGPU (BIOS: Integrated Graphics = Enabled) to free ~0.5–1 GB of VRAM from the compositor (2–4 more layers on GPU); then set `FIT_TARGET_MIB=512`. Games would then render through PRIME offload — treat it as an "LLM session" mode.

---

## Phase 5 — First light (three terminals)

### T1 — the monitor

```bash
~/llm/venv/bin/python ~/llm/bin/llmwatch.py --csv ~/llm/logs/watch-$(date +%F).csv --kill
```

One line per second, in four groups: GPU (util, VRAM, temp, power/limit, clocks, fan, P-state, PCIe gen/width + rx/tx, throttle reasons); CPU (load, clocks, Tctl/Tccd); RAM (usage, available, swap growth, memory/IO pressure); storage + the llama-server process (DIMM/NVMe temps, disk throughput, RSS, CPU, threads, major page faults, live tokens/s from `/metrics`).

Alerts go to stderr only on state change. GPU Xid errors are captured via NVML events. With `--kill`, five consecutive critical samples SIGTERM the server. Native cross-check: `nvidia-smi dmon -s pucvmet`.

### T2 — benchmark, then serve

```bash
M=~/llm/models/gemma-4-31b-it-heretic/gemma-4-31b-it-heretic-Q4_K_M.gguf
~/llm/llama.cpp/build/bin/llama-bench -m "$M" -ngl 24,28,32 -fa on -t 8 -p 512 -n 128 -r 3 --mmap 0
~/llm/bin/llm-serve.sh                 # QUANT=Q5_K_M to serve the larger quant
```

The launcher preflights: hashes (`SHA256SUMS`), free VRAM, starting GPU temp, available RAM, port conflicts, that this llama.cpp build supports every flag it uses, and that systemd can cap your session's memory. It then starts the server in a scope capped at 18G soft / 20G hard with swap forbidden — if memory runs out the kernel kills the server, not your desktop.

Placement is automatic (`--fit` on by default; `--fit-target` sets VRAM headroom to leave). In the load log, confirm: layers offloaded (~30–34 of 60), CUDA0 vs CPU/CPU_REPACK buffer sizes, KV + compute buffer sizes, and `AVX512 = 1`.

```bash
systemctl --user status llm-heretic-8080.scope    # Memory: high 18G, max 20G, swap max 0B
ss -ltnp | grep 8080                              # 127.0.0.1 only
```

### T3 — first contact

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/chat/completions -H @$HOME/llm/auth.hdr -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"In 3 sentences: why is LLM decoding memory-bandwidth bound?"}],"max_tokens":300}' \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"]); print(r.get("timings"))'
```

The key is read from a 0600 file and sent as a header, so it never appears in `ps`. Enable thinking for one request with `"chat_template_kwargs":{"enable_thinking":true}`. Web UI at http://127.0.0.1:8080; paste the key from `~/llm/api.key`.

---

## Phase 6 — Soak test and behavioral verification

### Thermal/stability soak

Run three long generations (`max_tokens` 2000) over 20–30 minutes while watching T1.

| Signal | Healthy | Act when |
| --- | --- | --- |
| GPU temp | < 75 °C | WARN/CRIT (from the card's own slowdown threshold, shown at startup) |
| GPU `thr` | `-` or `pwrcap` | `sw_therm`: fix airflow. `HW_THERM`/`HW_SLOW`/`HW_PBRAKE`: stop — hardware protection firing, covers sensors GeForce does not report. |
| PCIe | x16 | x8/x4 under load: reseat the card, check BIOS |
| Tctl | < 85 °C (PPT-limited) | ≥ 92 °C, or any MCE |
| DIMM | < 60 °C | ≥ 75 °C: improve DIMM airflow |
| NVMe | < 60 °C | ≥ 75 °C |
| PSI mem full | ~0 | > 5 % sustained: thrashing, lower CTX or close apps |
| swap+ | 0 | growing: something outside the scope is swapping |
| VRAM free | about the fit target | one jump on the first long prompt is normal; shrinking across requests is a leak |
| tg | 4–6 t/s | much lower: check offload count, CPU clocks, `thr` |

```bash
journalctl -k --since "-45 min" | grep -Ei 'xid|nvrm|mce|hardware error' || echo "kernel log clean"
sudo ras-mc-ctl --errors
```

### Behavioral diff against the base — verify the label

The claim is "only refusal was removed, KL 0.043 from base." Check that yourself rather than trusting it. Quantize the **official** `google/gemma-4-31B-it` the same way (or reuse the official QAT GGUF), serve it on a second port, and compare both models on the same neutral prompt set (coding, math, factual recall, summarization). If capability on benign tasks is materially worse than base, or the outputs diverge in ways unrelated to refusal, the edit did more than advertised — stop trusting it. This is the real research payload of running an abliterated model: measure what changed, do not just run it.

---

## Phase 7 — Operating rules

- **Isolation is the containment boundary.** Loopback only; no tool/agent/function-calling wiring, no shell access, no outbound network from anything consuming its output. Treat every response as untrusted text.
- **Never game and serve at the same time.** CUDA on Linux does not page VRAM; the loser gets an OOM or driver fault.
- **Stop the server before suspending.** A live CUDA context across suspend/resume is a classic Xid/hang source.
- **After kernel/driver updates,** `nvidia-smi` must work before launching. After a CUDA toolkit update, rebuild llama.cpp.
- **Upgrade llama.cpp deliberately:** new tag, rebuild, re-`llama-bench`, compare.
- **Raising context costs speed.** With `CTX=32768`, `--fit` gives up GPU layers to hold the larger KV cache. Compare the offload line between runs.
- **Keep the source hashes.** `SHA256SUMS` and `SOURCE` in the model dir record exactly which commit and recipe produced the file you run. Do not delete them.

### Next moves, in order of payoff

1. Run the Phase 6 behavioral diff — it is the point of the exercise.
2. Move the display to the iGPU for more GPU layers.
3. Compare Q4_K_M vs Q5_K_M for coherence at your typical prompt length.
4. Benchmark the base `google/gemma-4-31B-it` QAT build alongside it as your capability yardstick.

---

## Script reference

### `convert-model.sh`

Downloads `coder3101/gemma-4-31B-it-heretic` at the pinned commit, verifies the large files against the Hub's LFS SHA-256, converts safetensors → GGUF (text tower), quantizes to the target type(s), self-hashes the output, and locks it read-only. Env: `REV QUANTS OUTTYPE KEEP_SRC KEEP_INTERMEDIATE MIN_FREE_GB VENV LLAMACPP ROOT`.

### `llm-serve.sh`

Hardened launcher. Loopback only; API key from a 0600 file passed as a header (never in `ps`); memory-capped transient systemd scope with swap disabled. Preflight covers model hashes, free VRAM, GPU temp, RAM, port, cgroup delegation, and flag support. Env: `QUANT CTX THREADS PORT FIT_TARGET_MIB MEM_HIGH MEM_MAX VISION THINK VERIFY`. Default `QUANT=Q4_K_M`.

### `llmwatch.py`

Model-agnostic hardware telemetry (unchanged from the base protocol). Read-only except the optional kill switch. `stdout` one line/interval; `stderr` WARN/CRIT/CLEAR transitions, Xid events, kill actions; `--csv` full field log. Key options: `--kill`, `--kill-after N`, `--interval`, per-signal thresholds. Deps: `nvidia-ml-py`, `psutil`.

---

## Verification status

`convert-model.sh` and `llm-serve.sh` passed syntax checks and were exercised in a sandbox against stubbed `hf`/converter/quantizer, NVML, systemd, and `/metrics`: source hash verification, tamper abort, quant selection, self-hashing and read-only locking, keep-flag handling, kill switch, Xid capture, permission and port checks, and the flag-compatibility guard. `llmwatch.py` is unchanged from the base protocol. The first real run on `falkor` is the true test — keep `--kill` armed for it.
