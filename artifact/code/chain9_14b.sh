#!/bin/bash
# Stronger reader qwen2.5:14b (Ollama 0.30.8, Q4_K_M, temperature 0, seed 0): LaMP-3 table on the
# first 300 evaluation users for the kept rules, then a 40-user LaMP-QA pilot (no profile vs top-5)
# judged by the main judge (qwen2.5:7b, Ollama) after all 14B calls are done.
cd "$(dirname "$0")"
PY=../venv/bin/python
export WORKERS=2
N_EVAL_USE=450 $PY run.py reader lamp3 qwen2.5:14b ../results/lamp3_contexts_14b.json
N_EVAL_USE=450 $PY run.py score lamp3 qwen2.5:14b
echo LAMP3_14B_DONE
$PY run.py reader lampqa qwen2.5:14b ../results/lampqa_contexts_14b_pilot.json
mv ../results/lampqa_reader_outputs_qwen25_14b.json ../results/lampqa_pilot_outputs_qwen25_14b.json
$PY - <<'PY'
import json, os
from common import *
users = {u["uid"]: u for u in load_proc("lampqa")["users"]}
jc = LLMCache("judge_lampqa")
o = json.load(open(os.path.join(RES, "lampqa_pilot_outputs_qwen25_14b.json")))
sc = {uid: {k: utility("lampqa", users[uid], v, jc) for k, v in d.items()} for uid, d in o.items()}
json.dump(sc, open(os.path.join(RES, "lampqa_pilot_utility_qwen25_14b.json"), "w"))
import numpy as np
g = np.array([d[k] - d[""] for d in sc.values() for k in d if k])
print("pilot top5 gain", g.mean(), "n", len(g))
PY
echo QA_PILOT_DONE
