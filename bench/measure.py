#!/usr/bin/env python3
"""Phase 0 benchmark: VRAM, tokens/s and time-to-first-token for the main-model candidates.

For each (model, num_ctx) pair:
  1. Unload everything and record idle VRAM / WSL RAM as the baseline.
  2. Two warm-up requests: the first loads the model at num_ctx and calibrates the
     prompt size, the second brings the GPU up to boost clocks.
  3. N measured requests with a prompt filling ~85% of the context.
     A random nonce at the start of each prompt defeats Ollama's prefix cache,
     so every run pays the full prefill cost.
  4. Optionally load the helper model next to it and check both still fit on the GPU.

Standard library only. Usage:
  python3 bench/measure.py
  python3 bench/measure.py --models gpt-oss:20b --ctx 8192 --runs 2 --with-helper
"""

import argparse
import json
import random
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

OLLAMA = "http://127.0.0.1:11434"
HELPER = "qwen3:1.7b"
FILL_FRACTION = 0.85  # share of num_ctx the prompt occupies; leaves room for the reply
EXTRA_OPTIONS = {}    # extra Ollama options from the command line, e.g. num_batch

# Physics-flavoured vocabulary for the filler text. Content doesn't matter for speed,
# only token count does; varied words keep the tokenizer from compressing it unusually.
WORDS = (
    "the a of in and to is that for with as by on field energy momentum tensor metric "
    "frame observer clock interval proper time dilation contraction velocity boost "
    "Lorentz invariant covariant derivative manifold curvature geodesic Christoffel "
    "symbol connection Riemann Ricci scalar Einstein equation stress source vacuum "
    "Lagrangian Hamiltonian action principle variation symmetry Noether current charge "
    "conserved operator eigenvalue spectrum Hilbert space state amplitude probability "
    "wavefunction commutator spin rotation group representation matrix vector basis "
    "component index contraction trace determinant integral boundary condition solution "
    "therefore hence we obtain consider assume define compute follows result shows"
).split()


def api(path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(OLLAMA + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gpu_used_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0])


def mem_available_mib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable not found")


class GpuPeak:
    """Samples nvidia-smi in a background thread and keeps the maximum."""

    def __init__(self, interval=0.25):
        self.interval, self.peak = interval, 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, gpu_used_mib())
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def unload_all():
    for m in api("/api/ps").get("models", []):
        api("/api/generate", {"model": m["name"], "keep_alive": 0})
    for _ in range(60):
        if not api("/api/ps").get("models"):
            time.sleep(1.0)  # let the driver release memory
            return
        time.sleep(0.5)
    raise RuntimeError("models did not unload")


def ps_split(model):
    """(total_bytes, vram_bytes, context_length) for a loaded model, from /api/ps."""
    for m in api("/api/ps").get("models", []):
        if m["name"] == model or m["model"] == model:
            return m["size"], m["size_vram"], m.get("context_length")
    return None


def think_setting(model):
    # gpt-oss cannot disable reasoning, only lower it; Qwen3.x accepts a boolean.
    return "low" if model.startswith("gpt-oss") else False


def filler(n_words, rng):
    lines, words = [], [rng.choice(WORDS) for _ in range(n_words)]
    for i in range(0, len(words), 18):
        lines.append(" ".join(words[i:i + 18]).capitalize() + ".")
    return "\n".join(" ".join(lines[i:i + 6]) for i in range(0, len(lines), 6))


def chat_stream(model, prompt, num_ctx, num_predict):
    """One streamed /api/chat call. Returns wall-clock TTFT and Ollama's final stats."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "think": think_setting(model),
        "keep_alive": "10m",
        "options": {"num_ctx": num_ctx, "num_predict": num_predict,
                    "temperature": 0, "seed": 1, **EXTRA_OPTIONS},
    }
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            msg = chunk.get("message", {})
            if ttft is None and (msg.get("content") or msg.get("thinking")):
                ttft = time.perf_counter() - t0
            if chunk.get("done"):
                return ttft, chunk
    raise RuntimeError("stream ended without done=true")


def build_prompt(n_words, rng):
    return (f"[session {uuid.uuid4()}]\n"
            "Below is an excerpt from lecture notes.\n\n"
            f"{filler(n_words, rng)}\n\n"
            "Summarize the excerpt above in detail, section by section.")


def bench_one(model, num_ctx, runs, num_predict, with_helper, rng):
    print(f"\n=== {model} @ num_ctx={num_ctx}")
    unload_all()
    base_gpu, base_avail = gpu_used_mib(), mem_available_mib()

    # Warm-up doubles as calibration: ~1.4 tokens per word is a first guess,
    # then scale by the token count Ollama actually reports.
    target = int(num_ctx * FILL_FRACTION) - num_predict
    n_words = int(target / 1.4)
    t0 = time.perf_counter()
    _, stats = chat_stream(model, build_prompt(n_words, rng), num_ctx, 1)
    load_s = stats.get("load_duration", 0) / 1e9
    n_words = int(n_words * target / stats["prompt_eval_count"])
    print(f"  warm-up {time.perf_counter() - t0:.1f}s (load {load_s:.1f}s), "
          f"calibrated to {n_words} words")
    # Second warm-up at full size with some decoding: the GPU idles at low clocks (P8)
    # and needs a few seconds of load to boost, which otherwise skews run 1.
    chat_stream(model, build_prompt(n_words, rng), num_ctx, 64)

    samples = []
    with GpuPeak() as peak:
        for i in range(runs):
            ttft, s = chat_stream(model, build_prompt(n_words, rng), num_ctx, num_predict)
            r = {
                "ttft_s": ttft,
                "prompt_tokens": s["prompt_eval_count"],
                "prefill_tok_s": s["prompt_eval_count"] / (s["prompt_eval_duration"] / 1e9),
                "gen_tokens": s["eval_count"],
                "decode_tok_s": s["eval_count"] / (s["eval_duration"] / 1e9),
            }
            samples.append(r)
            print(f"  run {i + 1}: TTFT {r['ttft_s']:.2f}s  prompt {r['prompt_tokens']} tok "
                  f"@ {r['prefill_tok_s']:.0f} tok/s  decode {r['gen_tokens']} tok "
                  f"@ {r['decode_tok_s']:.1f} tok/s")

    size, size_vram, ctx_reported = ps_split(model)
    result = {
        "model": model,
        "num_ctx": num_ctx,
        "ctx_reported_by_ollama": ctx_reported,
        "load_s": load_s,
        "baseline_gpu_mib": base_gpu,
        "peak_gpu_mib": peak.peak,
        "model_gpu_mib": peak.peak - base_gpu,
        "ollama_size_mib": size // 2**20,
        "ollama_vram_mib": size_vram // 2**20,
        "gpu_fraction": size_vram / size,
        "wsl_ram_used_mib": base_avail - mem_available_mib(),
        "runs": samples,
        "median": {k: statistics.median(s[k] for s in samples)
                   for k in ("ttft_s", "prefill_tok_s", "decode_tok_s", "prompt_tokens")},
    }

    if with_helper:
        # Chat mode: main model + helper resident together.
        _, hs = chat_stream(HELPER, "Classify this question: what is proper time?", 4096, 32)
        main_after, helper_split = ps_split(model), ps_split(HELPER)
        result["with_helper"] = {
            "gpu_used_mib": gpu_used_mib(),
            "main_still_loaded": main_after is not None,
            "main_gpu_fraction": main_after[1] / main_after[0] if main_after else None,
            "helper_gpu_fraction": helper_split[1] / helper_split[0] if helper_split else None,
            "helper_decode_tok_s": hs["eval_count"] / (hs["eval_duration"] / 1e9),
        }
        print(f"  + helper: {result['with_helper']}")

    m = result["median"]
    print(f"  -> median TTFT {m['ttft_s']:.2f}s, decode {m['decode_tok_s']:.1f} tok/s, "
          f"GPU +{result['model_gpu_mib']} MiB, on-GPU share {result['gpu_fraction']:.0%}, "
          f"WSL RAM +{result['wsl_ram_used_mib']} MiB")
    return result


def print_table(results):
    print("\n| Model | Ctx | Prompt tok | TTFT (s) | Prefill tok/s | Decode tok/s "
          "| GPU MiB | On GPU | WSL RAM MiB |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        m = r["median"]
        print(f"| {r['model']} | {r['num_ctx'] // 1024}K | {m['prompt_tokens']:.0f} "
              f"| {m['ttft_s']:.2f} | {m['prefill_tok_s']:.0f} | {m['decode_tok_s']:.1f} "
              f"| {r['model_gpu_mib']} | {r['gpu_fraction']:.0%} | {r['wsl_ram_used_mib']} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["gpt-oss:20b", "qwen3.6:35b-a3b"])
    ap.add_argument("--ctx", nargs="+", type=int, default=[8192, 32768])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--gen", type=int, default=256, help="tokens to generate per run")
    ap.add_argument("--with-helper", action="store_true",
                    help=f"also load {HELPER} and check both fit on the GPU")
    ap.add_argument("--num-batch", type=int,
                    help="prompt batch size; larger batches cut weight uploads for CPU-offloaded experts")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = ap.parse_args()
    if args.num_batch:
        EXTRA_OPTIONS["num_batch"] = args.num_batch

    rng = random.Random(0)
    results = []
    for model in args.models:
        for ctx in args.ctx:
            try:
                results.append(bench_one(model, ctx, args.runs, args.gen, args.with_helper, rng))
            except Exception as e:  # keep going: one OOM shouldn't lose the other numbers
                print(f"  FAILED: {e}")
                results.append({"model": model, "num_ctx": ctx, "error": str(e)})
    unload_all()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"measure_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({"ollama_version": api("/api/version")["version"],
                                "options": EXTRA_OPTIONS, "results": results}, indent=2))
    print_table([r for r in results if "error" not in r])
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
