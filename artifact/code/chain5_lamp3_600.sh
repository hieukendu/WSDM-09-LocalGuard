#!/bin/bash
# Statistical power for the trade-off task: LaMP-3 extended to 600 evaluation users (200 appended
# by prep_extend.py), same backends as before (qwen2.5:7b on Ollama, gemma2:9b on llama.cpp).
cd "$(dirname "$0")"
PY=../venv/bin/python
B=~/.ollama/models/blobs
SRV=/opt/homebrew/Cellar/ollama/0.30.8/libexec/lib/ollama/llama-server
until grep -q QWEN_DONE ../logs/chain4_qwen.log && grep -q CPP_DONE ../logs/chain4_cpp.log; do sleep 60; done
cp ../data/proc/lamp3.json ../data/proc/lamp3_400eval_backup.json
cp ../data/proc/lamp3_emb.npy ../data/proc/lamp3_400eval_backup_emb.npy
$PY prep_extend.py lamp3 200
export N_EVAL_USE=600 N_READER=600 WORKERS=3
$PY run.py detect lamp3
$PY gate.py lamp3 > ../logs/gate600_lamp3.log
$PY run.py audit lamp3
$PY run.py reader lamp3
$PY run.py score lamp3
$SRV -m $B/sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373 \
  --port 18091 --host 127.0.0.1 -c 8192 -np 4 -ngl 999 --jinja --no-webui > ../logs/llamaserver_gemma.log 2>&1 &
until curl -s http://127.0.0.1:18091/health | grep -q ok; do sleep 5; done
LLAMACPP_PORT_gemma2_9b=18091 WORKERS=4 $PY run.py audit lamp3 gemma2:9b evaltop
pkill -f "llama-server -m .*ff1d1fc78170d787.* --port 18091"
echo LAMP3_600_DONE
