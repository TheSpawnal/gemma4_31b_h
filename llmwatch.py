#!/usr/bin/env python3
"""llmwatch - live hardware telemetry for local LLM inference (NVIDIA GPU, AMD CPU, Linux).

stdout : one status line per interval
stderr : WARN / CRIT / CLEAR transitions, XID events, kill actions
--csv  : every sampled field, one row per interval
--kill : SIGTERM llama-server after N consecutive CRIT samples (SIGKILL 10 s later if needed)

Deps: nvidia-ml-py, psutil. Read-only except for the optional kill switch.
"""
import argparse
import csv
import http.client
import os
import signal
import sys
import time
import urllib.parse
import urllib.request

try:
    import psutil
    import pynvml as nv
except ImportError as exc:
    sys.exit(f"missing dependency: {exc.name} (pip install nvidia-ml-py psutil)")

MIB = 1024 * 1024
GIB = 1024 * MIB

# NVML clock event (throttle) reason bits - stable ABI values
THR_BITS = (
    (0x001, "idle"), (0x002, "app"), (0x004, "pwrcap"), (0x008, "HW_SLOW"),
    (0x010, "sync"), (0x020, "sw_therm"), (0x040, "HW_THERM"), (0x080, "HW_PBRAKE"),
    (0x100, "disp"),
)
THR_CRIT = 0x008 | 0x040 | 0x080
THR_WARN = 0x020

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


def nvq(fn, *args):
    if fn is None:
        return None
    try:
        return fn(*args)
    except nv.NVMLError:
        return None


def fmt(v, spec="{:.0f}"):
    return "NA" if v is None else spec.format(v)


def fu(v, unit, spec="{:.0f}"):
    return "NA" if v is None else spec.format(v) + unit


def thr_str(mask):
    if mask is None:
        return "NA"
    names = [n for b, n in THR_BITS if mask & b and n != "idle"]
    return ",".join(names) if names else "-"


class Gpu:
    def __init__(self, index):
        nv.nvmlInit()
        h = self.h = nv.nvmlDeviceGetHandleByIndex(index)
        self.name = _s(nv.nvmlDeviceGetName(h))
        self.driver = _s(nv.nvmlSystemGetDriverVersion())
        self.slowdown = nvq(nv.nvmlDeviceGetTemperatureThreshold, h,
                            nv.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)
        self.shutdown = nvq(nv.nvmlDeviceGetTemperatureThreshold, h,
                            nv.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)
        self.pl_min = self.pl_max = None
        cons = nvq(nv.nvmlDeviceGetPowerManagementLimitConstraints, h)
        if cons:
            self.pl_min, self.pl_max = cons[0] / 1000, cons[1] / 1000
        pdef = nvq(nv.nvmlDeviceGetPowerManagementDefaultLimit, h)
        self.pl_def = pdef / 1000 if pdef is not None else None
        self.pcie_max_gen = nvq(nv.nvmlDeviceGetMaxPcieLinkGeneration, h)
        self.pcie_max_w = nvq(nv.nvmlDeviceGetMaxPcieLinkWidth, h)
        self._thr = (getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None)
                     or getattr(nv, "nvmlDeviceGetCurrentClocksThrottleReasons", None))
        self._fi_memtemp = getattr(nv, "NVML_FI_DEV_MEMORY_TEMP", 82)
        self.has_memtemp = self._memtemp() is not None
        self._wait = (getattr(nv, "nvmlEventSetWait_v2", None)
                      or getattr(nv, "nvmlEventSetWait", None))
        self.es = None
        try:
            es = nv.nvmlEventSetCreate()
            nv.nvmlDeviceRegisterEvents(h, getattr(nv, "nvmlEventTypeXidCriticalError", 0x8), es)
            self.es = es
        except (nv.NVMLError, AttributeError):
            self.es = None

    def _memtemp(self):
        try:
            fv = nv.nvmlDeviceGetFieldValues(self.h, [self._fi_memtemp])[0]
        except (nv.NVMLError, AttributeError, IndexError, TypeError):
            return None
        if fv.nvmlReturn != 0:
            return None
        v = fv.value
        val = {0: v.dVal, 1: v.uiVal, 2: v.ulVal, 3: v.ullVal, 4: v.sllVal}.get(fv.valueType)
        return val if val else None

    def _poll_xid(self):
        if self.es is None or self._wait is None:
            return []
        out = []
        for _ in range(32):
            try:
                ev = self._wait(self.es, 0)
            except nv.NVMLError as e:
                if getattr(e, "value", None) != nv.NVML_ERROR_TIMEOUT:
                    self.es = None
                break
            out.append(int(ev.eventData))
        return out

    def sample(self):
        h = self.h
        d = {}
        u = nvq(nv.nvmlDeviceGetUtilizationRates, h)
        d["gpu_util"] = u.gpu if u else None
        d["gpu_mem_util"] = u.memory if u else None
        m = nvq(nv.nvmlDeviceGetMemoryInfo, h)
        d["vram_used_mib"] = m.used / MIB if m else None
        d["vram_free_mib"] = m.free / MIB if m else None
        d["vram_total_mib"] = m.total / MIB if m else None
        d["gpu_temp"] = nvq(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU)
        d["gpu_memtemp"] = self._memtemp() if self.has_memtemp else None
        p = nvq(nv.nvmlDeviceGetPowerUsage, h)
        d["gpu_power_w"] = p / 1000 if p is not None else None
        pl = nvq(nv.nvmlDeviceGetEnforcedPowerLimit, h)
        d["gpu_plimit_w"] = pl / 1000 if pl is not None else None
        d["sm_mhz"] = nvq(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_SM)
        d["mem_mhz"] = nvq(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_MEM)
        d["fan_pct"] = nvq(nv.nvmlDeviceGetFanSpeed, h)
        d["pstate"] = nvq(nv.nvmlDeviceGetPerformanceState, h)
        d["pcie_gen"] = nvq(nv.nvmlDeviceGetCurrPcieLinkGeneration, h)
        d["pcie_width"] = nvq(nv.nvmlDeviceGetCurrPcieLinkWidth, h)
        # NVML reports KB/s, GPU-relative (rx = host -> device)
        rx = nvq(nv.nvmlDeviceGetPcieThroughput, h, nv.NVML_PCIE_UTIL_RX_BYTES)
        tx = nvq(nv.nvmlDeviceGetPcieThroughput, h, nv.NVML_PCIE_UTIL_TX_BYTES)
        d["pcie_rx_mbs"] = rx / 1024 if rx is not None else None
        d["pcie_tx_mbs"] = tx / 1024 if tx is not None else None
        d["thr_mask"] = nvq(self._thr, h)
        d["xid"] = self._poll_xid()
        return d

    def close(self):
        if self.es is not None:
            nvq(nv.nvmlEventSetFree, self.es)
        nvq(nv.nvmlShutdown)


def read_temps():
    fn = getattr(psutil, "sensors_temperatures", None)
    t = fn() if fn else {}
    k10 = t.get("k10temp", [])
    tctl = next((e.current for e in k10 if e.label == "Tctl"), None)
    ccds = [e.current for e in k10 if e.label.startswith("Tccd")]
    dimms = [e.current for e in t.get("spd5118", [])]
    nvme = [e.current for e in t.get("nvme", []) if e.label in ("Composite", "")]
    return tctl, (max(ccds) if ccds else None), dimms, (max(nvme) if nvme else None), sorted(t)


def read_psi(res):
    out = {}
    try:
        with open(f"/proc/pressure/{res}") as f:
            for line in f:
                kind, *fields = line.split()
                kv = dict(x.split("=", 1) for x in fields)
                out[kind] = float(kv["avg10"])
    except (OSError, KeyError, ValueError):
        pass
    return out.get("some"), out.get("full")


def read_majflt(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
    except OSError:
        return None
    rest = data[data.rfind(")") + 2:].split()
    return int(rest[9])  # field 12: majflt


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def find_proc(name, pid):
    if pid:
        try:
            return psutil.Process(pid)
        except psutil.Error:
            return None
    uid = os.getuid()
    for p in psutil.process_iter(["name", "uids"]):
        try:
            if p.info["name"] == name and p.info["uids"] and p.info["uids"].real == uid:
                return p
        except psutil.Error:
            continue
    return None


# Explicit empty ProxyHandler: never route the bearer token through an http(s)_proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def scrape(url, key):
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with _OPENER.open(req, timeout=0.5) as r:
            body = r.read(1 << 20).decode("utf-8", "replace")
    except (OSError, ValueError, http.client.HTTPException):
        return None
    out = {}
    for line in body.splitlines():
        if not line or line[0] == "#":
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        try:
            out[parts[0].split("{", 1)[0]] = float(parts[1])
        except ValueError:
            pass
    return out


def rate(cur, prev, num, den):
    if not cur or not prev or num not in cur or den not in cur:
        return None
    dn, dd = cur[num] - prev.get(num, 0.0), cur[den] - prev.get(den, 0.0)
    return dn / dd if dn > 0 and dd > 0 else None


def load_key(path):
    if not path:
        return None
    st = os.stat(path)
    if st.st_mode & 0o077:
        print(f"note: {path} is group/world accessible (chmod 600)", file=sys.stderr)
    with open(path) as f:
        return f.readline().strip() or None


class Alerts:
    def __init__(self):
        self.active = {}

    def update(self, conds, ts):
        for k, (lvl, msg) in conds.items():
            if self.active.get(k) != lvl:
                print(f"{ts} {lvl} {k} {msg}", file=sys.stderr, flush=True)
        for k in self.active:
            if k not in conds:
                print(f"{ts} CLEAR {k}", file=sys.stderr, flush=True)
        self.active = {k: v[0] for k, v in conds.items()}


def level(v, warn, crit):
    if v is None:
        return None
    if v >= crit:
        return "CRIT"
    if v >= warn:
        return "WARN"
    return None


def terminate(proc, ts, reason):
    try:
        print(f"{ts} KILL SIGTERM pid {proc.pid} ({reason})", file=sys.stderr, flush=True)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except psutil.TimeoutExpired:
        print(f"{ts} KILL SIGKILL pid {proc.pid}", file=sys.stderr, flush=True)
        proc.kill()
    except psutil.Error:
        pass


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--interval", type=float, default=1.0)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--csv", metavar="PATH")
    ap.add_argument("--proc", default="llama-server", help="process name to track")
    ap.add_argument("--pid", type=int)
    ap.add_argument("--url", default="http://127.0.0.1:8080/metrics", help="llama-server /metrics")
    ap.add_argument("--key-file", default=os.path.expanduser("~/llm/api.key"))
    ap.add_argument("--no-metrics", action="store_true", help="do not scrape llama-server")
    ap.add_argument("--kill", action="store_true", help="arm kill switch")
    ap.add_argument("--kill-after", type=int, default=5, help="consecutive CRIT samples")
    ap.add_argument("--gpu-warn", type=float)
    ap.add_argument("--gpu-crit", type=float)
    ap.add_argument("--cpu-warn", type=float, default=85)
    ap.add_argument("--cpu-crit", type=float, default=92)
    ap.add_argument("--dimm-warn", type=float, default=65)
    ap.add_argument("--dimm-crit", type=float, default=75)
    ap.add_argument("--nvme-warn", type=float, default=65)
    ap.add_argument("--nvme-crit", type=float, default=75)
    ap.add_argument("--psi-warn", type=float, default=5, help="memory PSI full avg10 %%")
    ap.add_argument("--psi-crit", type=float, default=20)
    ap.add_argument("--swap-warn", type=float, default=256, help="swap growth MiB")
    ap.add_argument("--swap-crit", type=float, default=1024)
    ap.add_argument("--vram-warn", type=float, default=512, help="min free VRAM MiB")
    a = ap.parse_args()
    if a.interval < 0.2:
        ap.error("interval must be >= 0.2 s")
    host = urllib.parse.urlparse(a.url).hostname
    if not a.no_metrics and host not in LOOPBACK:
        ap.error("--url must point to loopback (the API key is sent with it)")
    return a


CSV_FIELDS = [
    "ts", "gpu_util", "gpu_mem_util", "vram_used_mib", "vram_free_mib", "vram_total_mib",
    "gpu_temp", "gpu_memtemp", "gpu_power_w", "gpu_plimit_w", "sm_mhz", "mem_mhz", "fan_pct",
    "pstate", "pcie_gen", "pcie_width", "pcie_rx_mbs", "pcie_tx_mbs", "thr_mask",
    "cpu_pct", "cpu_max_core_pct", "cpu_mhz_avg", "cpu_mhz_max", "tctl", "tccd_max",
    "load1", "ram_used_gib", "ram_avail_gib", "swap_delta_mib", "psi_mem_some", "psi_mem_full",
    "psi_io_some", "dimm_max", "nvme_temp", "disk_rd_mbs", "disk_wr_mbs",
    "proc_pid", "proc_rss_gib", "proc_cpu_pct", "proc_threads", "proc_majflt_s",
    "tg_tps", "pp_tps", "busy",
]


def main():
    a = parse_args()
    gpu = Gpu(a.gpu_index)

    if a.gpu_crit is None:
        a.gpu_crit = min(gpu.slowdown - 3, 90) if gpu.slowdown else 88
    if a.gpu_warn is None:
        a.gpu_warn = a.gpu_crit - 7

    key = None
    if not a.no_metrics:
        try:
            key = load_key(a.key_file)
        except OSError:
            key = None

    _, _, dimms0, _, sensors = read_temps()
    vm0 = psutil.virtual_memory()
    swap0 = psutil.swap_memory().used
    print(f"llmwatch | {gpu.name} drv {gpu.driver} | pl def {fmt(gpu.pl_def)}W "
          f"range {fmt(gpu.pl_min)}-{fmt(gpu.pl_max)}W | slowdown {fmt(gpu.slowdown)}C "
          f"shutdown {fmt(gpu.shutdown)}C | pcie max G{fmt(gpu.pcie_max_gen)}x{fmt(gpu.pcie_max_w)} "
          f"| memtemp {'yes' if gpu.has_memtemp else 'NA'} | xid {'on' if gpu.es else 'journal only'}")
    print(f"llmwatch | {cpu_model()} {psutil.cpu_count(False)}c/{psutil.cpu_count()}t | "
          f"ram {vm0.total / GIB:.1f}G | swap {swap0 / MIB:.0f}M used at start | "
          f"sensors {','.join(sensors) or 'none'} | dimm sensors {len(dimms0)}")
    print(f"thresholds | gpu {a.gpu_warn:.0f}/{a.gpu_crit:.0f}C cpu {a.cpu_warn:.0f}/{a.cpu_crit:.0f}C "
          f"dimm {a.dimm_warn:.0f}/{a.dimm_crit:.0f}C nvme {a.nvme_warn:.0f}/{a.nvme_crit:.0f}C "
          f"psi_full {a.psi_warn:.0f}/{a.psi_crit:.0f}% swap+ {a.swap_warn:.0f}/{a.swap_crit:.0f}M "
          f"vram_free<{a.vram_warn:.0f}M | kill {'armed after ' + str(a.kill_after) if a.kill else 'off'}",
          flush=True)

    csv_f = writer = None
    if a.csv:
        new = not os.path.exists(a.csv) or os.path.getsize(a.csv) == 0
        csv_f = open(a.csv, "a", newline="", buffering=1)
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()

    alerts = Alerts()
    proc = None
    prev_flt = prev_metrics = None
    last_tg = last_pp = None
    crit_streak = 0
    psutil.cpu_percent(percpu=True)
    io_prev = psutil.disk_io_counters()
    t_prev = time.monotonic()
    next_t = t_prev + a.interval

    try:
        while True:
            time.sleep(max(0.0, next_t - time.monotonic()))
            next_t += a.interval
            now = time.monotonic()
            dt = max(now - t_prev, 1e-3)
            t_prev = now
            ts = time.strftime("%H:%M:%S")

            g = gpu.sample()
            per = psutil.cpu_percent(percpu=True)
            fr = psutil.cpu_freq(percpu=True) or []
            mhz = [f.current for f in fr if f and f.current]
            tctl, tccd, dimms, nvme_t, _ = read_temps()
            vm = psutil.virtual_memory()
            swap_delta = (psutil.swap_memory().used - swap0) / MIB
            psi_ms, psi_mf = read_psi("memory")
            psi_is, _ = read_psi("io")
            io = psutil.disk_io_counters()
            rd = wr = None
            if io and io_prev:
                rd = (io.read_bytes - io_prev.read_bytes) / dt / 1e6
                wr = (io.write_bytes - io_prev.write_bytes) / dt / 1e6
            io_prev = io

            if proc is None or not proc.is_running():
                proc = find_proc(a.proc, a.pid)
                prev_flt = None
                if proc:
                    try:
                        proc.cpu_percent(None)
                    except psutil.Error:
                        proc = None
            p_rss = p_cpu = p_thr = p_flt = None
            if proc:
                try:
                    with proc.oneshot():
                        p_rss = proc.memory_info().rss / GIB
                        p_cpu = proc.cpu_percent(None)
                        p_thr = proc.num_threads()
                    flt = read_majflt(proc.pid)
                    if flt is not None and prev_flt is not None:
                        p_flt = (flt - prev_flt) / dt
                    prev_flt = flt
                except psutil.Error:
                    proc = None

            busy = None
            if not a.no_metrics and proc:
                cur = scrape(a.url, key)
                if cur:
                    tg = (rate(cur, prev_metrics, "llamacpp:tokens_predicted_total",
                               "llamacpp:tokens_predicted_seconds_total")
                          or cur.get("llamacpp:predicted_tokens_seconds") or None)
                    pp = (rate(cur, prev_metrics, "llamacpp:prompt_tokens_total",
                               "llamacpp:prompt_seconds_total")
                          or cur.get("llamacpp:prompt_tokens_seconds") or None)
                    last_tg = tg or last_tg
                    last_pp = pp or last_pp
                    busy = cur.get("llamacpp:requests_processing")
                    prev_metrics = cur

            # ---- alert evaluation ----
            conds = {}

            def chk(name, v, warn, crit, unit="C"):
                lv = level(v, warn, crit)
                if lv:
                    conds[name] = (lv, f"{v:.1f}{unit}")

            chk("gpu_temp", g["gpu_temp"], a.gpu_warn, a.gpu_crit)
            chk("cpu_tctl", tctl, a.cpu_warn, a.cpu_crit)
            chk("dimm_temp", max(dimms) if dimms else None, a.dimm_warn, a.dimm_crit)
            chk("nvme_temp", nvme_t, a.nvme_warn, a.nvme_crit)
            chk("mem_psi_full", psi_mf, a.psi_warn, a.psi_crit, "%")
            chk("swap_growth", swap_delta, a.swap_warn, a.swap_crit, "MiB")
            m = g["thr_mask"]
            if m is not None and m & THR_CRIT:
                conds["gpu_hw_throttle"] = ("CRIT", thr_str(m))
            elif m is not None and m & THR_WARN:
                conds["gpu_sw_thermal"] = ("WARN", thr_str(m))
            if g["vram_free_mib"] is not None and g["vram_free_mib"] < a.vram_warn:
                conds["vram_low"] = ("WARN", f"{g['vram_free_mib']:.0f}MiB free")
            if (g["gpu_util"] or 0) >= 50 and g["pcie_width"] and gpu.pcie_max_w \
                    and g["pcie_width"] < gpu.pcie_max_w:
                conds["pcie_width"] = ("WARN", f"x{g['pcie_width']} < x{gpu.pcie_max_w} under load")
            for x in g["xid"]:
                print(f"{ts} CRIT xid {x}", file=sys.stderr, flush=True)
            alerts.update(conds, ts)

            crit_now = any(lv == "CRIT" for lv, _ in conds.values())
            crit_streak = crit_streak + 1 if crit_now else 0
            if a.kill and proc and crit_streak >= a.kill_after:
                reason = ",".join(k for k, (lv, _) in conds.items() if lv == "CRIT")
                terminate(proc, ts, reason)
                proc = None
                crit_streak = 0

            # ---- output ----
            cpu_avg = sum(per) / len(per) if per else None
            cpu_max = max(per) if per else None
            mhz_avg = sum(mhz) / len(mhz) if mhz else None
            mhz_max = max(mhz) if mhz else None
            dimm_max = max(dimms) if dimms else None
            memt = f" mem {fu(g['gpu_memtemp'], 'C')}" if gpu.has_memtemp else ""
            dimm_s = "/".join(f"{d:.0f}" for d in dimms) + "C" if dimms else "NA"
            ghz = lambda v: None if v is None else v / 1000
            line = (
                f"{ts} gpu {fu(g['gpu_util'], '%')} vram {fmt(g['vram_used_mib'])}/"
                f"{fu(g['vram_total_mib'], 'M')} {fu(g['gpu_temp'], 'C')}{memt} "
                f"{fmt(g['gpu_power_w'])}/{fu(g['gpu_plimit_w'], 'W')} sm {fmt(g['sm_mhz'])} "
                f"mclk {fmt(g['mem_mhz'])} fan {fu(g['fan_pct'], '%')} P{fmt(g['pstate'])} "
                f"pcie G{fmt(g['pcie_gen'])}x{fmt(g['pcie_width'])} "
                f"rx {fmt(g['pcie_rx_mbs'])} tx {fu(g['pcie_tx_mbs'], 'MB/s')} thr {thr_str(m)}"
                f" | cpu {fu(cpu_avg, '%')} max {fu(cpu_max, '%')} "
                f"{fmt(ghz(mhz_avg), '{:.2f}')}/{fu(ghz(mhz_max), 'GHz', '{:.2f}')} "
                f"tctl {fu(tctl, 'C')} ccd {fu(tccd, 'C')}"
                f" | ram {vm.used / GIB:.1f}/{vm.total / GIB:.1f}G avail {vm.available / GIB:.1f}G "
                f"swap+{swap_delta:.0f}M psi m {fmt(psi_ms, '{:.1f}')}/{fmt(psi_mf, '{:.1f}')} "
                f"io {fmt(psi_is, '{:.1f}')}"
                f" | dimm {dimm_s} nvme {fu(nvme_t, 'C')} disk r {fmt(rd)} w {fu(wr, 'MB/s')}"
                f" | llm "
                + (f"rss {fu(p_rss, 'G', '{:.1f}')} cpu {fu(p_cpu, '%')} thr {fmt(p_thr)} "
                   f"majflt {fu(p_flt, '/s')} tg {fmt(last_tg, '{:.2f}')} pp {fmt(last_pp, '{:.1f}')} t/s"
                   if proc else "not running")
            )
            print(line, flush=True)

            if writer:
                writer.writerow({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    **{k: g[k] for k in (
                        "gpu_util", "gpu_mem_util", "vram_used_mib", "vram_free_mib",
                        "vram_total_mib", "gpu_temp", "gpu_memtemp", "gpu_power_w",
                        "gpu_plimit_w", "sm_mhz", "mem_mhz", "fan_pct", "pstate", "pcie_gen",
                        "pcie_width", "pcie_rx_mbs", "pcie_tx_mbs", "thr_mask")},
                    "cpu_pct": cpu_avg, "cpu_max_core_pct": cpu_max, "cpu_mhz_avg": mhz_avg,
                    "cpu_mhz_max": mhz_max, "tctl": tctl, "tccd_max": tccd,
                    "load1": os.getloadavg()[0], "ram_used_gib": vm.used / GIB,
                    "ram_avail_gib": vm.available / GIB, "swap_delta_mib": swap_delta,
                    "psi_mem_some": psi_ms, "psi_mem_full": psi_mf, "psi_io_some": psi_is,
                    "dimm_max": dimm_max, "nvme_temp": nvme_t, "disk_rd_mbs": rd,
                    "disk_wr_mbs": wr, "proc_pid": proc.pid if proc else None,
                    "proc_rss_gib": p_rss, "proc_cpu_pct": p_cpu, "proc_threads": p_thr,
                    "proc_majflt_s": p_flt, "tg_tps": last_tg, "pp_tps": last_pp, "busy": busy,
                })
    except KeyboardInterrupt:
        pass
    finally:
        if csv_f:
            csv_f.close()
        gpu.close()


if __name__ == "__main__":
    main()
