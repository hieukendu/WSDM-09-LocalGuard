#!/bin/bash
# LaMP-2 extension, laptop part only (PLAN.md amendment 2026-10-08): the steps of chain11_lamp2.sh
# that use laptop models. Auditors: qwen2.5:7b (Ollama) and gemma2:9b (bundled llama.cpp).
# The reader step (Qwen2.5-14B on the shared endpoint :18140) and the analysis stay in
# chain11_lamp2.sh and run when the endpoint is back; nothing here calls port 18140.
cd "$(dirname "$0")"
PY=../venv/bin/python
B=~/.ollama/models/blobs
SRV=/opt/homebrew/Cellar/ollama/0.30.8/libexec/lib/ollama/llama-server
export N_TRAIN_USE=120 N_EVAL_USE=150 N_READER=150
[ -f ../results/lamp2_detect.json ] || $PY run.py detect lamp2
WORKERS=3 $PY run.py audit lamp2; echo AUDIT_QWEN_DONE
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
print('train audit users', len(out))
PY
$PY gate.py lamp2 > ../logs/gate_lamp2.log; echo GATE_DONE
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
$SRV -m $B/sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373 \
  --port 18091 --host 127.0.0.1 -c 8192 -np 3 -ngl 999 --jinja --no-webui > ../logs/llamaserver_gemma.log 2>&1 &
until curl -s http://127.0.0.1:18091/health | grep -q ok; do sleep 5; done
LLAMACPP_PORT_gemma2_9b=18091 WORKERS=3 $PY run.py audit lamp2 gemma2:9b evaltop; echo AUDIT_GEMMA_DONE
pkill -f "llama-server -m .*ff1d1fc78170d787.* --port 18091"
echo CHAIN11A_DONE
