import json, random, time
from vllm import LLM, SamplingParams

def ids(n, seed):
    r = random.Random(seed); return [r.randrange(500, 200000) for _ in range(n)]

llm = LLM(model="kumar2235/Qwen3.5-4B-AWQ", max_model_len=8192,
          max_num_seqs=128, gpu_memory_utilization=0.9,
          trust_remote_code=True, enable_prefix_caching=False)
# warmup
llm.generate({"prompt_token_ids": ids(64,1)}, SamplingParams(max_tokens=8,temperature=0.0), use_tqdm=False)

res = {"model":"Qwen3.5-4B-AWQ", "gpu":"GPU1 x8", "config":"CUDA Graph on"}

# ---- prefill (max_tokens=1, wall=TTFT incl decode of 1 token) ----
pre = []
for plen in [128, 512, 2048, 4096, 8000]:
    dts=[]
    for r in range(5):
        t0=time.perf_counter()
        llm.generate({"prompt_token_ids": ids(plen, 1000+plen+r)}, SamplingParams(max_tokens=1,temperature=0.0), use_tqdm=False)
        dts.append(time.perf_counter()-t0)
    dt=min(dts)
    pre.append({"plen":plen, "ttft_s":round(dt,4), "tok_s":round(plen/dt,0)})
    print(f"prefill plen={plen:5d} TTFT={dt:.4f}s  {plen/dt:,.0f} tok/s", flush=True)
res["prefill"]=pre

# ---- decode: 单流 TPOT (prompt 128) ----
dts=[]
for r in range(3):
    t0=time.perf_counter()
    out=llm.generate({"prompt_token_ids": ids(128,555+r)}, SamplingParams(max_tokens=512,temperature=0.0), use_tqdm=False)
    dts.append(time.perf_counter()-t0)
dt=min(dts); n=len(out[0].outputs[0].token_ids)
# 扣除128 token prefill(用上面plen=128的TTFT)
pf = pre[0]["ttft_s"]/plen if False else (pre[0]["ttft_s"])
tpot=(dt-pf)/(n-1)
res["decode"]={"prompt":128,"out":n,"wall_s":round(dt,2),"tpot_ms":round(tpot*1000,2),"tok_s":round(1/tpot,1)}
print(f"decode single: wall={dt:.2f}s out={n} TPOT={tpot*1000:.2f}ms = {1/tpot:.1f} tok/s", flush=True)

# ---- decode batch 缩放 ----
bs=[]
for B in [2,4,8,16]:
    reqs=[{"prompt_token_ids": ids(128, 7000+B*37+i)} for i in range(B)]
    t0=time.perf_counter()
    outs=llm.generate(reqs, SamplingParams(max_tokens=128,temperature=0.0), use_tqdm=False)
    dt=time.perf_counter()-t0
    tot=sum(len(o.outputs[0].token_ids) for o in outs)
    bs.append({"batch":B, "wall_s":round(dt,2), "agg_tok_s":round(tot/dt,0), "per_req_tok_s":round(tot/dt/B,1)})
    print(f"batch={B:2d} agg={tot/dt:,.0f} tok/s  per={tot/dt/B:.1f}", flush=True)
res["decode_batch"]=bs

json.dump(res, open("final_params.json","w"), indent=2)
print("saved final_params.json")
