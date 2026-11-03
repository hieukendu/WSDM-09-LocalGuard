#!/bin/bash
# Extends reader calls from the first 150 to all 300 audited evaluation users (statistical power).
# Deterministic: the first 150 users' rules and outputs are cache hits.
set -e
cd "$(dirname "$0")"
T=$1; PY=../venv/bin/python
export N_READER=300
$PY gate.py $T > ../logs/gate300_$T.log
WORKERS=3 $PY run.py audit $T
WORKERS=3 $PY run.py reader $T
if [ $T = lampqa ]; then WORKERS=3 $PY run.py judge $T; else $PY run.py score $T; fi
$PY analyze.py $T > ../logs/analyze_$T.log 2>&1
echo EXTEND_DONE
