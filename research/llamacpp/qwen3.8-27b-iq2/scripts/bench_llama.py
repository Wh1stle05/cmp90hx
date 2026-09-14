#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_llama.py —— llama.cpp / Ollama 通用测速脚本（prefill / decode / 投机解码）

设计目标：和 research/vllm/qwen35-4b-awq/REPORT.md 同口径
  * prefill：输入长度 128 / 512 / 2048 / 4096 / 8000，max_tokens=1，测 TTFT 与吞吐
  * decode ：固定短 prompt，输出 128 / 512 token，测 TPOT(tok/s)
  * spec   ：三类任务（逐字复述 / 代码改写 / 开放问答），对比投机解码前后

关键防坑（照抄 vLLM 报告里的两条教训）：
  1. 关闭 prefix cache（llama.cpp: `cache_prompt:false`）+ 每次请求带随机 nonce，避免"命中缓存"的假数据；
     并把引擎回报的 cache 命中数 cache_n 打进结果表，命中不为 0 会直接标红。
  2. 用 /tokenize + /detokenize 构造"精确 token 长度"的 prompt（Ollama 无此接口时退化为近似，并回报实际长度）。

用法：
  python3 bench_llama.py all      --base-url http://127.0.0.1:8080/v1 --label baseline
  python3 bench_llama.py prefill  --lens 128,512,2048,4096,8000 --reps 3
  python3 bench_llama.py decode   --out 128,512 --prompt-len 128 --reps 3
  python3 bench_llama.py spec     --reps 3
  # Ollama:
  python3 bench_llama.py all --base-url http://127.0.0.1:11434/v1 --model qwen3.8-27b

常用参数：--reps N（默认 3，取 min）、--json out.json、--no-monitor（不采样显存/功耗）
"""
import argparse, csv, json, re, statistics, subprocess, sys, threading, time, urllib.error, urllib.request, uuid

FILLER = ("The on-disk row cache keeps embedding rows on disk and fetches only the rows that a token "
          "needs, because the address of every row is derived from the token itself and is known "
          "before the layer runs, so the fetch can be issued early and overlapped with compute. ")

CODE = ('def summarize(rows, limit):\n'
        '    v1 = 0\n'
        '    out = []\n'
        '    for r in rows:\n'
        '        v1 += r["n"]\n'
        '        if v1 > limit:\n'
        '            break\n'
        '        out.append({"id": r["id"], "n": r["n"]})\n'
        '    return "total=%d" % v1, out\n')

COPY_SRC = ("The cache keeps a bounded number of pages resident, evicts in least-recently-used order, "
            "and never holds the whole table in memory. When the table is large, the working set of a "
            "single sequence is tiny, so the hit rate stays high and the cost per token stays close to "
            "the cost of a normal embedding lookup. ")


# ---------------------------------------------------------------- HTTP helpers
class Client:
    def __init__(self, base_url, model=None, timeout=1800):
        self.base = base_url.rstrip("/")
        self.root = self.base[:-3] if self.base.endswith("/v1") else self.base
        self.timeout = timeout
        self.model = model or self._detect_model()
        self.has_tokenize = self._probe_tokenize()

    def _req(self, url, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _detect_model(self):
        try:
            d = self._req(self.base + "/models")
            items = d.get("data") or d.get("models") or []
            if items:
                return items[0].get("id") or items[0].get("model")
        except Exception as e:
            print(f"  (warn) 无法自动获取模型名: {e}")
        return "default"

    def _probe_tokenize(self):
        try:
            self._req(self.root + "/tokenize", {"content": "hi"})
            return True
        except Exception:
            return False

    def tokens(self, text):
        if not self.has_tokenize:
            return None
        return len(self._req(self.root + "/tokenize", {"content": text})["tokens"])

    def detok(self, ids):
        return self._req(self.root + "/detokenize", {"tokens": ids})["content"]

    TMPL_OVERHEAD = None

    def warmup(self):
        """全局预热一次，并标定 chat 模板固定开销（prompt_n - 文本token数）"""
        try:
            probe = "warmup probe"
            d, _ = self.chat(probe, max_tokens=1)
            t = timings_of(d)
            n_txt = self.tokens(probe)
            if t["prompt_n"] and n_txt:
                type(self).TMPL_OVERHEAD = t["prompt_n"] - n_txt
        except Exception:
            type(self).TMPL_OVERHEAD = 0

    def chat(self, prompt, max_tokens, temperature=0.0, extra=None):
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": temperature, "cache_prompt": False}
        if extra:
            body.update(extra)
        t0 = time.time()
        d = self._req(self.base + "/chat/completions", body)
        return d, time.time() - t0


# ---------------------------------------------------------------- prompt 构造
def build_prompt(client, target_tokens):
    """构造 token 数尽量等于 target_tokens 的 prompt；返回 (text, 实际 token 数)"""
    nonce = f"[run {uuid.uuid4().hex[:8]}] "
    if not client.has_tokenize:
        # 退化：约 1 token ≈ 4 字符英文（Ollama 无 /tokenize）
        body = nonce + FILLER * max(1, int(target_tokens * 4 / len(FILLER)))
        return body, client.tokens(body) or -1
    n_nonce = client.tokens(nonce)
    lo, hi = 1, 8
    while client.tokens(nonce + FILLER * hi) < target_tokens:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if client.tokens(nonce + FILLER * mid) < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    text = nonce + FILLER * max(1, lo - 1)
    ids = client._req(client.root + "/tokenize", {"content": text})["tokens"]
    need = target_tokens - 1
    if len(ids) < need:
        ids = client._req(client.root + "/tokenize", {"content": nonce + FILLER * (lo + 4)})["tokens"]
    text = client.detok(ids[:need])
    return text, client.tokens(text)


def build_prompt_engine(cli, target_engine_tokens):
    """构造让【引擎侧 prompt_n】≈ target 的 user 文本（扣掉 chat 模板固定开销）"""
    overhead = cli.TMPL_OVERHEAD
    if overhead is None:
        cli.warmup(); overhead = cli.TMPL_OVERHEAD or 0
    return build_prompt(cli, max(8, target_engine_tokens - overhead))


# ---------------------------------------------------------------- GPU 采样
class GpuMon(threading.Thread):
    def __init__(self, enabled=True):
        super().__init__(daemon=True)
        self.on = enabled
        self.peak_vram = 0
        self.max_power = 0.0
        self.stop = False

    def run(self):
        while not self.stop and self.on:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used,power.draw", "--format=csv,noheader"],
                    text=True).strip().split("\n")[0]
                v, p = [x.strip() for x in out.split(",")]
                self.peak_vram = max(self.peak_vram, int(v.split()[0]))
                self.max_power = max(self.max_power, float(p.split()[0]))
            except Exception:
                pass
            time.sleep(0.3)


def timings_of(d):
    t = d.get("timings") or {}
    return {
        "prompt_n": t.get("prompt_n") or d.get("usage", {}).get("prompt_tokens"),
        "gen_n": t.get("predicted_n") or d.get("usage", {}).get("completion_tokens"),
        "prompt_ms": t.get("prompt_ms"),
        "prompt_tps": t.get("prompt_per_second"),
        "gen_tps": t.get("predicted_per_second"),
        "cache_n": t.get("cache_n"),
        "draft_n": t.get("draft_n"),
        "draft_accepted": t.get("draft_n_accepted"),
    }


# ---------------------------------------------------------------- 三个测试
def test_prefill(cli, lens, reps, out, mon):
    rows = []
    cli.warmup()
    print(f"\n=== prefill（max_tokens=1，{reps} 次取 min；cache_prompt=false）===")
    print(f"{'目标长度':>8} {'引擎prompt_n':>12} {'TTFT引擎(s)':>11} {'TTFTwall(s)':>11} "
          f"{'吞吐(tok/s)':>12} {'cache_n':>8}")
    for L in lens:
        samples = []
        for i in range(reps):
            prompt, n = build_prompt_engine(cli, L)
            d, wall = cli.chat(prompt, max_tokens=1)
            t = timings_of(d)
            ttft = (t["prompt_ms"] / 1000) if t["prompt_ms"] else wall
            tps = t["prompt_tps"] or (n / ttft)
            samples.append({"target": L, "prompt_n": t["prompt_n"] or n, "ttft_s": round(ttft, 4),
                            "ttft_wall_s": round(wall, 4), "prompt_ms": t["prompt_ms"],
                            "prefill_tps": round(tps, 1), "cache_n": t["cache_n"]})
        best = dict(min(samples, key=lambda x: x["ttft_s"]))
        best["samples"] = samples
        rows.append(best)
        flag = " ⚠缓存命中!" if (best["cache_n"] or 0) > 0 else ""
        print(f"{L:>8} {str(best['prompt_n']):>12} {best['ttft_s']:>11.3f} {best['ttft_wall_s']:>11.3f} "
              f"{best['prefill_tps']:>12.1f} {str(best['cache_n']):>8}{flag}")
    out["prefill"] = rows
    return rows


DECODE_PROMPT = "请写一段约 150 字的短文，主题是秋天的山间清晨。不要复述题目，直接写正文。"


def test_decode(cli, prompt_len, outs, reps, out, mon):
    rows = []
    cli.warmup()
    print(f"\n=== decode（自由生成，避免复述类任务触发投机；{reps} 次取 min）===")
    print(f"{'输出长度':>8} {'TPOT(ms)':>9} {'tok/s(TPOT)':>12} {'引擎gen_tps':>12} {'cache_n':>8}")
    for O in outs:
        samples = []
        for i in range(reps):
            if prompt_len > 0:
                tail, _ = build_prompt_engine(cli, prompt_len)
                prompt = f"[{uuid.uuid4().hex[:8]}] " + DECODE_PROMPT + "\n\n参考资料：\n" + tail
            else:
                prompt = f"[{uuid.uuid4().hex[:8]}] " + DECODE_PROMPT
            n = cli.tokens(prompt) or -1
            d, wall = cli.chat(prompt, max_tokens=O)
            t = timings_of(d)
            gen = t["gen_n"] or O
            pre_s = (t["prompt_ms"] / 1000) if t["prompt_ms"] else wall * 0.1
            tpot = (wall - pre_s) / max(gen, 1)
            samples.append({"out": O, "prompt_n": t["prompt_n"] or n, "prompt_target": prompt_len, "gen_n": gen,
                            "wall_s": round(wall, 3), "tpot_ms": round(tpot * 1000, 2),
                            "gen_tps_tpot": round(1.0 / tpot, 1) if tpot > 0 else None,
                            "gen_tps_engine": round(t["gen_tps"], 1) if t["gen_tps"] else None,
                            "cache_n": t["cache_n"], "draft_n": t["draft_n"],
                            "draft_accepted": t["draft_accepted"]})
        best = dict(max(samples, key=lambda x: x["gen_tps_tpot"] or 0))
        best["samples"] = samples
        rows.append(best)
        print(f"{O:>8} {best['tpot_ms']:>9.2f} {best['gen_tps_tpot']:>12.1f} "
              f"{str(best['gen_tps_engine']):>12} {str(best['cache_n']):>8}")
    out["decode"] = rows
    return rows


SPEC_TASKS = [
    ("逐字复述", lambda n: "一字不差地重复输出下面这段文字，不要加任何解释、不要翻译、不要改标点：\n\n" + COPY_SRC * 3),
    ("代码改写", lambda n: "把下面 Python 代码里的变量名 v1 全部改名为 total，其余一字不改，原样输出完整代码：\n\n```python\n" + CODE + "```"),
    ("开放问答", lambda n: "请用中文详细解释什么是梯度下降，并举一个生活中的类比。"),
]


def test_spec(cli, reps, out, mon, max_tokens=256):
    rows = []
    cli.warmup()
    print(f"\n=== 投机解码（输出 {max_tokens} token，{reps} 次取最优）===")
    print("    注：投机收益取决于任务是否可被 n-gram 命中；输出越长越接近真实比例，可用 --spec-max-tokens 调整")
    print(f"{'任务':<10} {'tok/s':>8} {'草稿数':>8} {'接受数':>8} {'接受率':>8} {'耗时(s)':>8}")
    for name, mk in SPEC_TASKS:
        samples = []
        for i in range(reps):
            prompt = mk(0)
            d, wall = cli.chat(prompt, max_tokens=max_tokens)
            t = timings_of(d)
            dn, da = t["draft_n"] or 0, t["draft_accepted"] or 0
            samples.append({"task": name, "wall_s": round(wall, 2), "gen_n": t["gen_n"],
                            "gen_tps": round(t["gen_tps"], 2) if t["gen_tps"] else None,
                            "draft_n": dn, "draft_accepted": da,
                            "accept_rate": round(da / dn, 3) if dn else None})
        best = dict(max(samples, key=lambda x: x["gen_tps"] or 0))
        best["samples"] = samples
        rows.append(best)
        acc = f"{best['accept_rate']*100:.1f}%" if best["accept_rate"] else "-"
        draft = f"{best['draft_accepted']}/{best['draft_n']}" if best["draft_n"] else "无草稿"
        print(f"{name:<10} {str(best['gen_tps']):>8} {str(best['draft_n']):>8} "
              f"{str(best['draft_accepted']):>8} {acc:>8} {best['wall_s']:>8}")
        if not best["draft_n"]:
            print(f"{'':<10}   （服务端未开投机解码，或该任务没命中草稿池）draft={draft}")
    out["spec"] = rows
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["all", "prefill", "decode", "spec"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default="run")
    ap.add_argument("--lens", default="128,512,2048,4096,8000")
    ap.add_argument("--out", default="128,512")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--spec-max-tokens", type=int, default=256)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-monitor", action="store_true")
    a = ap.parse_args()

    cli = Client(a.base_url, a.model)
    print(f"base_url = {a.base_url}")
    print(f"model    = {cli.model}")
    print(f"/tokenize 可用 = {cli.has_tokenize}" + ("" if cli.has_tokenize else "（长度将按字符估算，表中回报实际 token 数）"))

    mon = GpuMon(enabled=not a.no_monitor)
    if not a.no_monitor:
        mon.start()

    out = {"label": a.label, "base_url": a.base_url, "model": cli.model,
           "has_tokenize": cli.has_tokenize, "reps": a.reps, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if a.mode in ("all", "prefill"):
        test_prefill(cli, [int(x) for x in a.lens.split(",")], a.reps, out, mon)
    if a.mode in ("all", "decode"):
        test_decode(cli, a.prompt_len, [int(x) for x in a.out.split(",")], a.reps, out, mon)
    if a.mode in ("all", "spec"):
        test_spec(cli, a.reps, out, mon, a.spec_max_tokens)

    if not a.no_monitor:
        mon.stop = True
        time.sleep(0.4)
        out["gpu"] = {"peak_vram_mib": mon.peak_vram, "max_power_w": round(mon.max_power, 1)}
        print(f"\nGPU：峰值显存 {mon.peak_vram} MiB，峰值功耗 {mon.max_power:.0f} W")

    if a.json:
        json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=2)
        print(f"结果已写入 {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
