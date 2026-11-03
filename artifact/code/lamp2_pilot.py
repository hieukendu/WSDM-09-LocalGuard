"""LaMP-2 pilot (declared in PLAN.md, round-3 plan): reader Qwen2.5-14B-Instruct Q4_K_M on the shared
endpoint :18140 (own backend), first 60 evaluation users, no profile vs similarity top-5.
Go if the accuracy gain >= 0.05 and its 95% bootstrap lower bound > 0."""
import json
import os

import numpy as np

from common import RES, SEED, K, LLMCache, generate, load_proc, reader_prompt, utility, NUM_PREDICT
from run import pmap

MODEL = "qwen2.5-14b-ep18140"
os.environ.setdefault("LLAMACPP_PORT_qwen2_5_14b_ep18140", "18140")
os.environ.setdefault("LLAMACPP_BACKEND", "llamacpp-6f3a9f3d-endpoint18140-Qwen2.5-14B-Instruct-Q4_K_M-official-split")
users = [u for u in load_proc("lamp2")["users"] if u["split"] == "eval"][:60]
cache = LLMCache("reader_lamp2_qwen25_14b_ep18140")
items = [(u, ()) for u in users] + [(u, tuple(range(K))) for u in users]
outs = list(pmap(lambda t: generate(cache, MODEL, reader_prompt("lamp2", t[0], t[1]), num_predict=NUM_PREDICT["lamp2"]), items))
n = len(users)
u0 = np.array([utility("lamp2", users[i], outs[i]) for i in range(n)])
u5 = np.array([utility("lamp2", users[i], outs[n + i]) for i in range(n)])
rng = np.random.default_rng(SEED)
d = u5 - u0
bs = [d[rng.integers(0, n, n)].mean() for _ in range(5000)]
res = dict(n=n, acc_none=float(u0.mean()), acc_top5=float(u5.mean()), gain=[float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
res["go"] = bool(res["gain"][0] >= 0.05 and res["gain"][1] > 0)
res["examples"] = [o.strip()[:30] for o in outs[:5] + outs[n:n + 5]]
json.dump(res, open(os.path.join(RES, "lamp2_pilot.json"), "w"), indent=1)
print(res)
