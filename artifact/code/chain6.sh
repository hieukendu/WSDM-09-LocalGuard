#!/bin/bash
# Reordered qwen2.5:7b work (Ollama): finish LaMP-7, then LaMP-3 to 600 evaluation users (the
# trade-off task), then LaMP-QA to 300; gemma2:9b (llama.cpp) for the appended LaMP-3 users last.
cd "$(dirname "$0")"
PY=../venv/bin/python
B=~/.ollama/models/blobs
SRV=/opt/homebrew/Cellar/ollama/0.30.8/libexec/lib/ollama/llama-server
while pgrep -f "run.py reader lamp7" > /dev/null; do sleep 20; done
export WORKERS=3
N_READER=300 $PY run.py reader lamp7 && $PY run.py score lamp7 && echo "EXTENDED lamp7"
# LaMP-3: 600 evaluation users
cp ../data/proc/lamp3.json ../data/proc/lamp3_400eval_backup.json
cp ../data/proc/lamp3_emb.npy ../data/proc/lamp3_400eval_backup_emb.npy
$PY prep_extend.py lamp3 200
export N_EVAL_USE=600
N_READER=600 $PY run.py detect lamp3
N_READER=600 $PY gate.py lamp3 > ../logs/gate600_lamp3.log
N_READER=600 $PY run.py audit lamp3
N_READER=600 $PY run.py reader lamp3
$PY run.py score lamp3
echo "EXTENDED lamp3 600"
unset N_EVAL_USE
# LaMP-QA: 300 evaluation users
N_READER=300 $PY gate.py lampqa > ../logs/gate300_lampqa.log
N_READER=300 $PY run.py audit lampqa
N_READER=300 $PY run.py reader lampqa
$PY run.py judge lampqa
echo "EXTENDED lampqa"
until grep -q CPP_DONE ../logs/chain4_cpp.log; do sleep 60; done
$SRV -m $B/sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373 \
  --port 18091 --host 127.0.0.1 -c 8192 -np 4 -ngl 999 --jinja --no-webui > ../logs/llamaserver_gemma.log 2>&1 &
until curl -s http://127.0.0.1:18091/health | grep -q ok; do sleep 5; done
N_EVAL_USE=600 LLAMACPP_PORT_gemma2_9b=18091 WORKERS=4 $PY run.py audit lamp3 gemma2:9b evaltop
pkill -f "llama-server -m .*ff1d1fc78170d787.* --port 18091"
echo CHAIN6_DONE
