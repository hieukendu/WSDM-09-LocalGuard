#!/bin/bash
# Stages after audit + counterfactual collection: scoring, gate fitting, reader runs, oracle, analysis.
# usage: pipeline_rest.sh TASK
set -e
cd "$(dirname "$0")"
T=$1; PY=../venv/bin/python; R=../results
until [ -f $R/${T}_cf_outputs_qwen25_7b.json ] && [ -f $R/${T}_detect.json ]; do sleep 30; done
if [ $T = lampqa ]; then $PY run.py judge $T; else $PY run.py score $T; fi
# the gate needs auditor labels of training users only; wait until the audit cache covers them
until $PY -c "
import json,sys
from common import *
from run import audit_ids, tag
c=LLMCache('audit_${T}_'+tag(AUDITOR))
tr=[u for u in load_proc('$T')['users'] if u['split']=='train']
ok=all(c.get(AUDITOR, AUDIT_PROMPT.format(text=record_text('$T',u['cands'][ci])), {'temperature':0.0,'num_predict':40,'seed':0,'fmt':'json'}) is not None for u in tr for ci in audit_ids(u, False))
sys.exit(0 if ok else 1)"; do sleep 60; done
$PY - <<PYEOF
import json, os
from common import *
from run import audit_ids, tag
c=LLMCache('audit_${T}_'+tag(AUDITOR))
out={}
for u in load_proc('$T')['users']:
    if u['split']=='train':
        for ci in audit_ids(u, False):
            out.setdefault(u['uid'],{})[str(ci)]=audit_record(c, AUDITOR, '$T', u['cands'][ci])
json.dump(out, open(os.path.join(RES,'${T}_audit_train.json'),'w'))
PYEOF
$PY gate.py $T
$PY run.py reader $T
if [ $T = lampqa ]; then $PY run.py judge $T; else $PY run.py score $T; fi
until [ -f $R/${T}_audit.json ]; do sleep 30; done
$PY analyze.py $T > ../logs/analyze_$T.log 2>&1 || true
$PY oracle.py $T 80
$PY analyze.py $T > ../logs/analyze_$T.log 2>&1
echo PIPELINE_DONE
