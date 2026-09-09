#!/usr/bin/env python3
"""decode 诊断: eager vs cudagraph, batch scaling, clock 采样"""
import random, time, subprocess, json
from vllm import LLM, SamplingParams

MODEL = "kumar2235/Qwen3.5-4B-AWQ"

def make_ids(n, seed):
    rnd = random.Random(seed)
    return [rnd.randrange(500, 200000) for _ in range(n)]

def load(eager):
    llm = LLM(model=MODEL, max_model_len=8192, enforce_eager=eager,
              max_num_seqs=128,
              gpu_memory_utilization=0.9, trust_remote_code=True,
              enable_prefix_caching=False)
    return llm

def tpot(llm, plen=128, olen=512, reps=3):
    sp = SamplingParams(max_tokens=olen, temperature=0.0)
    dts = []
    for r in range(reps):
        t0 = time.perf_counter()
        out = llm.generate({"prompt_token_ids": make_ids(plen, r)}, sp, use_tqdm=False)
        dts.append(time.perf_counter() - t0)
    dt = min(dts)
    n = len(out[0].outputs[0].token_ids)
    return dt, n

res = {}
# warmup eager kernel compile quickly
llm = load(False)
sp = SamplingParams(max_tokens=8, temperature=0.0)
llm.generate({"prompt_token_ids": make_ids(64, 1)}, sp, use_tqdm=False)

# --- cudagraph single ---
dt, n = tpot(llm)
res['cudagraph_single'] = {"wall": dt, "out": n, "tpot_ms": dt/(max(n-1,1))*1000}
print(f"cudagraph single: wall={dt:.2f}s out={n} tpot={(dt/(n-1))*1000:.1f}ms", flush=True)

# --- batch scaling (cudagraph) ---
for bs in [1, 2, 4, 8]:
    reqs = [{"prompt_token_ids": make_ids(128, 5000+bs*100+i)} for i in range(bs)]
    spb = SamplingParams(max_tokens=256, temperature=0.0)
    t0 = time.perf_counter()
    outs = llm.generate(reqs, spb, use_tqdm=False)
    dt = time.perf_counter() - t0
    tot = sum(len(o.outputs[0].token_ids) for o in outs)
    res[f'batch{bs}'] = {"wall": dt, "total_out": tot,
                         "agg_tok_s": tot/dt, "per_req_tok_s": tot/dt/bs}
    print(f"batch={bs}: wall={dt:.2f}s total_out={tot} agg={tot/dt:.1f} tok/s "
          f"per_req={tot/dt/bs:.1f}", flush=True)

# --- clock sample during decode (launch async-ish) ---
import threading
stop = threading.Event()
samples = []
def gen_loop():
    while not stop.is_set():
        llm.generate({"prompt_token_ids": make_ids(128, 777)}, spb, use_tqdm=False)
threading.Thread(target=gen_loop, daemon=True).start()
time.sleep(1)
for _ in range(8):
    r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,utilization.gpu,utilization.memory",
                        "--format=csv,noheader,nounits", "-i", "1"], capture_output=True, text=True)
    samples.append(r.stdout.strip())
    time.sleep(0.4)
stop.set()
time.sleep(0.5)
res['clock_samples'] = samples
print("clocks:", samples, flush=True)

json.dump(res, open("decode_diag.json", "w"), indent=2)
print("saved decode_diag.json")
