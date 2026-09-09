import json, torch, time
from vllm import LLM, SamplingParams
def _main():

    def mem():
        u = torch.cuda.mem_get_info()
        return u[1]/2**30, (u[1]-u[0])/2**30

    print("=== baseline (no model) ===", flush=True)
    print(f"total={mem()[0]:.2f}GiB used={mem()[1]:.2f}GiB", flush=True)

    t0=time.time()
    llm = LLM(model="kumar2235/Qwen3.5-4B-AWQ", max_model_len=8192,
              max_num_seqs=128, gpu_memory_utilization=0.9,
              trust_remote_code=True, enable_prefix_caching=False)
    print(f"=== after load ({time.time()-t0:.1f}s) ===", flush=True)
    t,u = mem()
    print(f"total={t:.2f}GiB used={u:.2f}GiB", flush=True)

    # 1 token / 1k / 8k prompt 后的 KV 增量
    import random
    def ids(n, seed=7):
        r=random.Random(seed); return [r.randrange(500,200000) for _ in range(n)]

    for plen, olen in [(1,1),(1000,1),(6000,1000),(8000,4)]:
        t0=time.time()
        out=llm.generate({"prompt_token_ids": ids(plen)}, SamplingParams(max_tokens=olen,temperature=0.0), use_tqdm=False)
        dt=time.time()-t0
        t,u = mem()
        print(f"plen={plen} olen={olen}: wall={dt:.2f}s used={u:.2f}GiB", flush=True)

    # KV cache 容量情况从日志里已见: 102400 tokens
    # 实际: 8k+4 tok 后还剩多少
    t,u = mem()
    print(f"FINAL used={u:.2f}GiB  free={t-u:.2f}GiB", flush=True)


if __name__ == "__main__":
    _main()
