#!/bin/bash
# LaMP-2 extension after the pilot passed (PLAN.md round-3 plan): 150 evaluation users.
# Auditors: qwen2.5:7b (Ollama) and gemma2:9b (bundled llama.cpp on the laptop), as for the other tasks.
# Reader: Qwen2.5-14B-Instruct on the shared endpoint :18140 (own backend) for every set of the kept rules.
cd "$(dirname "$0")"
PY=../venv/bin/python
B=~/.ollama/models/blobs
SRV=/opt/homebrew/Cellar/ollama/0.30.8/libexec/lib/ollama/llama-server
export N_TRAIN_USE=120 N_EVAL_USE=150 N_READER=150
$PY run.py detect lamp2
WORKERS=3 $PY run.py audit lamp2
$PY - <<'PY'
import json, os
from common import *
from run import audit_ids, tag
c = LLMCache('audit_lamp2_' + tag(AUDITOR))
out = {}
for u in load_proc('lamp2')['users']:
    if u['split'] == 'train':
        for ci in audit_ids(u, False):
            out.setdefault(u['uid'], {})[str(ci)] = audit_record(c, AUDITOR, 'lamp2', u['cands'][ci])
json.dump(out, open(os.path.join(RES, 'lamp2_audit_train.json'), 'w'))
PY
$PY gate.py lamp2 > ../logs/gate_lamp2.log
$PY - <<'PY'
import json, os
from common import load_proc, RES
rules = json.load(open(os.path.join(RES, 'lamp2_rules.json')))
ctx = {}
for u in [x for x in load_proc('lamp2')['users'] if x['split'] == 'eval']:
    r = rules[u['uid']]['sets']; seen = set(); L = []
    for m in ('sim', 'flt', 'cert_sim'):
        for s in r[m]:
            k = tuple(sorted(s))
            if k not in seen:
                seen.add(k); L.append(list(k))
    ctx[u['uid']] = L
json.dump(ctx, open(os.path.join(RES, 'lamp2_contexts_ep14b.json'), 'w'))
print('contexts', sum(len(v) for v in ctx.values()))
PY
LLAMACPP_PORT_qwen2_5_14b_ep18140=18140 WORKERS=4 \
LLAMACPP_BACKEND="llamacpp-6f3a9f3d-endpoint18140-Qwen2.5-14B-Instruct-Q4_K_M-official-split" \
  $PY run.py reader lamp2 qwen2.5-14b-ep18140 ../results/lamp2_contexts_ep14b.json
$PY run.py score lamp2 qwen2.5-14b-ep18140
echo LAMP2_READER_DONE
$SRV -m $B/sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373 \
  --port 18091 --host 127.0.0.1 -c 8192 -np 4 -ngl 999 --jinja --no-webui > ../logs/llamaserver_gemma.log 2>&1 &
until curl -s http://127.0.0.1:18091/health | grep -q ok; do sleep 5; done
LLAMACPP_PORT_gemma2_9b=18091 WORKERS=4 $PY run.py audit lamp2 gemma2:9b evaltop
pkill -f "llama-server -m .*ff1d1fc78170d787.* --port 18091"
$PY analyze.py lamp2 qwen25-14b-ep18140 > ../logs/analyze_lamp2.log 2>&1
echo CHAIN11_DONE
