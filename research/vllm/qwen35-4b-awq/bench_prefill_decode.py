#!/usr/bin/env python3
"""Qwen3.5-4B-AWQ single-GPU prefill/decode bench (vLLM offline).

- prefill: prompt_len in [128, 512, 2048, 4096, 8192], max_tokens=1
- decode:  prompt 128, out_len in [128, 512]
- sample:  realistic text prompt
"""
import json
import random
import time

from vllm import LLM, SamplingParams

MODEL = "kumar2235/Qwen3.5-4B-AWQ"
MAX_LEN = 8192


def make_ids(n, seed):
    rnd = random.Random(seed)
    return [rnd.randrange(500, 200000) for _ in range(n)]


print("loading...", flush=True)
t0 = time.perf_counter()
llm = LLM(
    model=MODEL,
    max_model_len=MAX_LEN,
    enforce_eager=True,
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    enable_prefix_caching=False,  # else repeated prompts hit KV cache
)
print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)


def run(n, seed_base, max_tokens, reps=3):
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    dts, out = [], None
    for r in range(reps):
        # fresh random prompt every rep: no cache reuse
        t0 = time.perf_counter()
        out = llm.generate({"prompt_token_ids": make_ids(n, seed_base + r)},
                           sp, use_tqdm=False)
        dts.append(time.perf_counter() - t0)
    return min(dts), out[0]


# warmup (compile triton kernels etc.)
run(128, 0, 16, reps=1)

res = {"model": MODEL, "max_model_len": MAX_LEN,
       "prefill": [], "decode": []}

# ---- prefill ----
for plen in [128, 512, 2048, 4096, 8000]:
    dt, out = run(plen, 1000 + plen, 1, reps=3)
    ttft = None
    try:
        m = out.metrics
        if m is not None and m.first_token_time is not None \
                and m.arrival_time is not None:
            ttft = m.first_token_time - m.arrival_time
    except Exception:
        ttft = None
    rate = plen / dt
    res["prefill"].append({"prompt_len": plen, "wall_s": round(dt, 4),
                           "ttft_s": ttft, "prefill_tok_s": round(rate, 1)})
    print(f"prefill plen={plen:5d} wall={dt:.3f}s ttft={ttft} "
          f"rate={rate:.1f} tok/s", flush=True)

p128 = res["prefill"][0]["prefill_tok_s"]

# ---- decode ----
for olen in [128, 512]:
    dt, out = run(128, 9999, olen, reps=2)
    n_out = len(out.outputs[0].token_ids)
    prefill_t = 128 / p128
    denom = max(n_out - 1, 1)
    tpot = (dt - prefill_t) / denom
    res["decode"].append({"prompt_len": 128, "out_len": n_out,
                          "wall_s": round(dt, 3),
                          "tpot_ms": round(tpot * 1000, 2),
                          "decode_tok_s": round(1 / tpot, 1)})
    print(f"decode out={n_out} wall={dt:.2f}s tpot={tpot * 1000:.2f}ms "
          f"tok/s={1 / tpot:.1f}", flush=True)

# ---- realistic sample ----
sp = SamplingParams(temperature=0.7, max_tokens=256)
t0 = time.perf_counter()
out = llm.generate(["Explain machine learning in one paragraph."],
                   sp, use_tqdm=False)[0]
dt = time.perf_counter() - t0
txt = out.outputs[0].text
res["sample"] = {"wall_s": round(dt, 2),
                 "out_len": len(out.outputs[0].token_ids),
                 "text": txt[:2000]}
print(f"sample wall={dt:.2f}s out_len={res['sample']['out_len']}",
      flush=True)
print(txt[:500], flush=True)

with open("results.json", "w") as f:
    json.dump(res, f, indent=2)
print("saved results.json", flush=True)
