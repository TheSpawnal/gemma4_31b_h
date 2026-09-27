# gemma4_31b_h
gemma 4 31b h ARA local deployment 
# Local LLM Deployment Protocol — Gemma 4 31B on RTX 5070

Host: `falkor` (Ryzen 7 9700X, RTX 5070 12 GB, 64 GB DDR5-6000, Ubuntu 26.04)
Model: `google/gemma-4-31B-it-qat-q4_0-gguf` @ commit `59dde24573e7e61570dba08b18a2e1fe246955ed`
Scripts: `fetch-model.sh`, `llm-serve.sh`, `llmwatch.py`

---

## The hard numbers

The RTX 5070 has **12 GB of VRAM**. The 32 GB DDR5 is system RAM; the GPU cannot use it as its own memory. Gemma 4 31B is a dense model: 30.7B parameters across 60 layers, with a 1024-token sliding window and a 256K-token context. At 4-bit the weights are larger than VRAM, so the model runs split: roughly half the layers on the GPU and the rest on the CPU, read from DDR5.

The chosen file is Google's own QAT build. Quantization-Aware Training keeps quality close to bfloat16 while cutting the memory needed to load the model. The repo holds a 17.7 GB model file and a 1.2 GB vision projector (mmproj). Three advantages: the publisher is the `google` org itself (supply chain), QAT gives better quality at 4-bit, and Q4_0 has the cheapest dequantization. llama.cpp also repacks Q4_0 into interleaved layouts for fast SIMD on the CPU half.

### Speed estimate

Each generated token reads every weight once.

| Half | Size | Bandwidth | Time per token |
| --- | --- | --- | --- |
| GPU | ~8.5–9 GB | 672 GB/s | ~14 ms |
| CPU | ~9 GB | ~55–60 GB/s effective | ~150–165 ms |

The 9700X is a single-CCD chip. Its CCD-to-IO-die link reads about 32 B per clock, so at FCLK 2000 you get roughly 64 GB/s peak (about 55–60 GB/s in practice). DDR5-6000's theoretical 96 GB/s does not apply.

- **Ceiling:** about 6 tokens/s.
- **Realistic:** 4–5 tokens/s.
- Each layer moved onto the GPU (~0.25–0.3 GB) saves about 4–5 ms per token.

Practical consequence: thinking mode can produce ~1,500 hidden reasoning tokens, which at this speed is 5–6 minutes before the first visible word. The launcher keeps thinking **off by default**; switch it on per request.

### A faster sibling worth benchmarking later

The 26B A4B MoE has 25.2B total parameters but only 3.8B active, and runs almost as fast as a 4B model. On this machine it should generate several times faster than the 31B. The cost is a few benchmark points; the gap is larger on Codeforces and long-context tasks. Install the 31B as planned, then compare.

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

- **PCIe link.** `LnkSta` usually drops to a lower speed at idle (normal power saving). The width must be x16. The monitor shows the live generation under load.
- **Sensors.** `sensors` should list `k10temp` (Tctl, Tccd1) and `nvme`. On some boards it also lists `spd5118`, one entry per DIMM, which is the DDR5 temperature. If missing, try `sudo modprobe spd5118` and rerun `sensors`. If still missing, the board does not expose it — skip it.

### CPU stability matters more here than usual

llama.cpp's CPU layers on Zen 5 run AVX-512 dot-product kernels. That is the kind of load where a marginal Curve Optimizer offset produces silent wrong results or machine-check errors instead of clean crashes. If per-core CO validation is still pending, do it now. CPUs 0–7 are one thread per physical core:

```bash
for c in 0 1 2 3 4 5 6 7; do taskset -c $c stress-ng --cpu 1 --cpu-method fft --verify -t 3m --metrics-brief; done
stress-ng --cpu 8 --cpu-method matrixprod --verify -t 20m --metrics-brief
sudo ras-mc-ctl --errors
```

For a true AVX-512 torture test, y-cruncher's stress mode (Linux build from numberworld.org) is the reference. Any MCE in the output means backing the CO off before trusting the model's answers.

---

## Phase 1 — Toolchain

The NVIDIA driver already comes from Ubuntu, so get the toolkit from Ubuntu too. The archive's `nvidia-cuda-toolkit` trails upstream but uses the same driver Ubuntu provides. The Ubuntu and NVIDIA-repository install methods cannot be combined: mixing packages from both leads to conflicting library paths and broken `nvcc` lookups.

```bash
sudo apt install -y build-essential cmake ninja-build git pkg-config python3-venv nvidia-cuda-toolkit
nvcc --version
```

The "CUDA Version" in the `nvidia-smi` header should be at least the `nvcc` release. If older, update the driver through Ubuntu: run `sudo ubuntu-drivers list` and install the newest `-open` package. Do not use NVIDIA's repo for this. If `nvcc` rejects your GCC version, add `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14` to the configure step below.

---

## Phase 2 — Build llama.cpp at a pinned tag

```bash
mkdir -p ~/llm && cd ~/llm
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
TAG=$(git tag -l 'b*' --sort=-v:refname | head -n1)
git -c advice.detachedHead=false checkout "$TAG"
git rev-parse HEAD | tee ../llama.cpp.pinned

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j 8 --target llama-server llama-cli llama-bench
./build/bin/llama-server --version
./build/bin/llama-server --list-devices      # expect RTX 5070, compute capability 12.0
```

Why these options:

- **`-j 8`, not 16.** Parallel `nvcc` jobs each use 1–2 GB of RAM; 16 of them can push a 32 GB machine out of memory while compiling.
- **`GGML_NATIVE=ON`** builds for your exact CPU (AVX-512) and auto-detects the GPU architecture.
- **`LLAMA_CURL=OFF`** leaves the built-in downloader out, so model files can only arrive through the verified pipeline below. If CMake says the variable is unused, that is harmless.

---

## Phase 3 — Get the model (pinned and hash-verified)

```bash
python3 -m venv ~/llm/venv
~/llm/venv/bin/pip install -U huggingface_hub nvidia-ml-py psutil
mkdir -p ~/llm/bin && cp fetch-model.sh llm-serve.sh llmwatch.py ~/llm/bin/ && chmod 700 ~/llm/bin/*
~/llm/bin/fetch-model.sh
```

`fetch-model.sh`:

- Pins the download to commit `59dde24`.
- Fetches the expected SHA-256 and size of each file from the Hub API at that same commit.
- Downloads only those two files.
- Checks size and hash, then writes `SHA256SUMS` and `SOURCE` and locks the files read-only (0400).

Keep the monitor running during the download to watch NVMe temperature over a sustained 19 GB write. The license is Apache 2.0, so no token should be needed. On a 401, create a read-only fine-grained token, run `hf auth login`, then `hf auth logout` afterwards.

Never load `.bin`/pickle weights from anyone. GGUF contains no executable code, but GGUF parsers have had memory-safety bugs in the past. Official files plus an up-to-date llama.cpp is the defense.

---

## Phase 4 — One-time guardrails (root)

### Power limit (optional, cheap insurance)

Generation speed is limited by memory bandwidth, so capping power barely changes it — it trims prompt-processing bursts and gaming heat. Read the allowed range first, then pick a value inside it:

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

- **Why the persistence daemon:** without it, the driver can tear down its state (including the power limit) when the last GPU client exits. If `nvidia-persistenced` is absent, add `ExecStart=/usr/bin/nvidia-smi -i 0 -pm 1` before the `-pl` line instead.
- **Power connector:** at 250 W the 5070's 12V-2x6 connector carries modest current per pin, far from 5090 territory. Still check it is fully clicked in with no sharp bend near the plug.

### Firewall

```bash
sudo ufw default deny incoming && sudo ufw default allow outgoing && sudo ufw enable && sudo ufw status verbose
```

The server binds to 127.0.0.1 regardless; UFW is the second wall. Never use `--host 0.0.0.0`. For remote access, use an SSH tunnel: `ssh -L 8080:127.0.0.1:8080 falkor`.

### Optional VRAM gain

Plug the monitor into the motherboard so the iGPU drives the display (you may need to set Integrated Graphics to Enabled in the BIOS). That frees roughly 0.5–1 GB of VRAM from the compositor — 2–4 more layers on the GPU; then set `FIT_TARGET_MIB=512`. The catch: games would then render through PRIME offload, adding another variable to a Wayland gaming setup. Treat it as an "LLM session" mode.

---

## Phase 5 — First light (three terminals)

### T1 — the monitor

Arm the kill switch from the start:

```bash
~/llm/venv/bin/python ~/llm/bin/llmwatch.py --csv ~/llm/logs/watch-$(date +%F).csv --kill
```

One line per second, in four groups:

- **GPU:** utilization, VRAM, temperature, power/limit, clocks, fan, P-state, PCIe gen/width and rx/tx, throttle reasons.
- **CPU:** load, clocks, Tctl/Tccd.
- **RAM:** usage, available memory, swap growth, memory and I/O pressure (PSI).
- **Storage and the llama-server process:** DIMM and NVMe temperatures, disk throughput, then the process's RSS, CPU, threads, major page faults, and live tokens/s from `/metrics`.

Alerts go to stderr only when a state changes. GPU Xid errors are captured through NVML events. With `--kill`, five consecutive critical samples send SIGTERM to llama-server. For a native cross-check, `nvidia-smi dmon -s pucvmet` in a spare terminal shows the same GPU counters.

### T2 — controlled benchmark first, then the server

```bash
M=~/llm/models/gemma-4-31b-it-qat-q4_0/gemma-4-31B_q4_0-it.gguf
~/llm/llama.cpp/build/bin/llama-bench -m "$M" -ngl 24,28,32 -fa on -t 8 -p 512 -n 128 -r 3 --mmap 0
~/llm/bin/llm-serve.sh
```

The benchmark shows how generation speed changes as layers move to the GPU. The launcher then runs these checks before starting:

- hashes, free VRAM, starting GPU temperature, available RAM, port conflicts
- that the llama.cpp build supports every flag it uses
- that systemd lets your user sessions cap memory, so the cap below is actually enforced

It starts the server in a scope capped at 18G (soft) and 20G (hard) with swap forbidden. If memory runs out, the kernel kills the server instead of freezing the desktop.

Placement is automatic: `--fit` is enabled by default and sizes placement to fit memory; `--fit-target` sets how much free memory to leave on the GPU. In the load log, look for:

- the number of layers offloaded (expect roughly 30–34 of 60)
- the CUDA0 and CPU/CPU_REPACK model buffer sizes
- the KV cache and compute buffer sizes
- `AVX512 = 1` in the system info line

Then confirm the scope and the binding:

```bash
systemctl --user status llm-gemma31b-8080.scope    # Memory line must show high 18G, max 20G, swap max 0B
ss -ltnp | grep 8080                               # 127.0.0.1 only
```

### T3 — first contact

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/chat/completions -H @$HOME/llm/auth.hdr -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"In 3 sentences: why is LLM decoding memory-bandwidth bound?"}],"max_tokens":300}' \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"]); print(r.get("timings"))'
```

The key is sent as a header read from a 0600 file, so it never appears in `ps` output. To enable thinking for one request, add `"chat_template_kwargs":{"enable_thinking":true}` to the JSON — the same mechanism the launcher uses to set the default. The web UI is at http://127.0.0.1:8080; paste the key from `~/llm/api.key` into its settings.

---

## Phase 6 — Soak test and reading the signals

Run three long generations (`max_tokens` 2000) over 20–30 minutes while watching T1.

| Signal | Healthy | Act when |
| --- | --- | --- |
| GPU temp | < 75 °C | WARN/CRIT, derived from the card's own slowdown threshold (shown at startup) |
| GPU `thr` | `-` or `pwrcap` | `sw_therm`: fix airflow. `HW_THERM`/`HW_SLOW`/`HW_PBRAKE`: stop — hardware protection is firing, and it also covers sensors GeForce does not report (memory junction). |
| PCIe | x16 | x8/x4 under load: reseat the card, check BIOS |
| Tctl | < 85 °C (PPT-limited) | ≥ 92 °C, or any MCE |
| DIMM | < 60 °C | ≥ 75 °C: improve airflow over the DIMMs |
| NVMe | < 60 °C | ≥ 75 °C |
| PSI mem full | ~0 | > 5 % sustained: thrashing, lower CTX or close apps |
| swap+ | 0 | growing: something outside the scope is swapping |
| VRAM free | about the fit target | One jump during the first long prompt is normal (CUDA pool growth). Shrinking across requests is a leak. |
| tg | 4–6 t/s | much lower: check offload count, CPU clocks, `thr` |

Afterwards:

```bash
journalctl -k --since "-45 min" | grep -Ei 'xid|nvrm|mce|hardware error' || echo "kernel log clean"
sudo ras-mc-ctl --errors
```

---

## Phase 7 — Operating rules

- **Never game and serve at the same time.** CUDA on Linux does not page VRAM; whichever process loses gets an out-of-memory error or a driver fault.
- **Stop the server before suspending.** A live CUDA context across suspend/resume is a classic source of Xid errors and hangs.
- **After kernel or driver updates,** `nvidia-smi` must work before launching. After a CUDA toolkit update, rebuild llama.cpp.
- **Upgrade llama.cpp deliberately:** new tag, rebuild, rerun `llama-bench`, compare.
- **Raising context costs speed.** With `CTX=32768`, `--fit` gives up GPU layers to hold the bigger KV cache, so generation slows. Compare the offload line between runs.
- **If the model is ever connected to tools or agents,** treat its output as untrusted input and sandbox anything it can execute.

### Next moves, in order of payoff

1. Move the display to the iGPU for more layers on the GPU.
2. Run this same protocol on the 26B-A4B QAT.
3. Try MTP speculative decoding with Google's drafter. The assistant model must be a QAT checkpoint with the same precision as the target model.
4. Enable vision with `VISION=1`. The projector stays on the CPU and costs no VRAM.

---

## Script reference

### `llmwatch.py`

Live hardware telemetry for local LLM inference (NVIDIA GPU, AMD CPU, Linux). Read-only except for the optional kill switch.

| Stream | Content |
| --- | --- |
| stdout | one status line per interval |
| stderr | WARN / CRIT / CLEAR transitions, Xid events, kill actions |
| `--csv` | every sampled field, one row per interval |

Key options: `--kill` (SIGTERM llama-server after N consecutive CRIT samples, SIGKILL 10 s later if needed), `--kill-after N`, `--interval`, `--gpu-index`, `--url`/`--key-file` (loopback only), `--no-metrics`, and per-signal `--*-warn`/`--*-crit` thresholds. Dependencies: `nvidia-ml-py`, `psutil`.

### `llm-serve.sh`

Hardened launcher for `llama-server`. Loopback only; API key from a 0600 file, passed as a header (never in `ps`); memory-capped transient systemd scope with swap disabled. Preflight covers model hashes, free VRAM, GPU temperature, RAM, port, cgroup delegation, and flag support.

Env overrides: `CTX THREADS PORT FIT_TARGET_MIB MEM_HIGH MEM_MAX VISION THINK VERIFY`.

### `fetch-model.sh`

Pinned, hash-verified download of `google/gemma-4-31B-it-qat-q4_0-gguf` at commit `59dde24`. Verifies size and SHA-256 against the Hub's LFS metadata, then locks files read-only. `VISION=0` skips the 1.2 GB mmproj.

---

## Verification status

All three scripts passed syntax checks and were exercised in a sandbox against stubbed NVML, systemd and `/metrics` endpoints: kill switch, Xid capture, hash-mismatch abort, permission and port checks, and the flag-compatibility guard. The first real run on `falkor` is the true test — keep `--kill` armed for it.
