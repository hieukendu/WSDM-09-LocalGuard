#!/bin/bash
# Reader, auditor and judge calls of qwen2.5:7b for all 300 evaluation users, on the same backend as
# every other qwen2.5:7b cell (Ollama 0.30.8, Q4_K_M, temperature 0, seed 0).
cd "$(dirname "$0")"
PY=../venv/bin/python
export WORKERS=3 N_READER=300
unset LLAMACPP_PORT_qwen2_5_7b
for T in lamp3 lamp7 lampqa; do
  $PY gate.py $T > ../logs/gate300_$T.log
  $PY run.py audit $T
  $PY run.py reader $T
  if [ $T = lampqa ]; then $PY run.py judge $T; else $PY run.py score $T; fi
  echo EXTENDED $T
done
echo QWEN_DONE
