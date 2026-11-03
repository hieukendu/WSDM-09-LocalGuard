#!/bin/bash
# gemma2:9b second-auditor labels for the 200 appended LaMP-3 users as soon as they exist, then
# the LaMP-3 analysis on 600 evaluation users (llama.cpp server for gemma, as in its other cells).
cd "$(dirname "$0")"
PY=../venv/bin/python
B=~/.ollama/models/blobs
SRV=/opt/homebrew/Cellar/ollama/0.30.8/libexec/lib/ollama/llama-server
until grep -q "EXTENDED lamp3 600" ../logs/chain6.log; do sleep 60; done
$SRV -m $B/sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373 \
  --port 18091 --host 127.0.0.1 -c 8192 -np 4 -ngl 999 --jinja --no-webui > ../logs/llamaserver_gemma.log 2>&1 &
until curl -s http://127.0.0.1:18091/health | grep -q ok; do sleep 5; done
N_EVAL_USE=600 LLAMACPP_PORT_gemma2_9b=18091 WORKERS=4 $PY run.py audit lamp3 gemma2:9b evaltop
pkill -f "llama-server -m .*ff1d1fc78170d787.* --port 18091"
N_EVAL_USE=600 $PY analyze.py lamp3 > ../logs/analyze_lamp3.log 2>&1
echo CHAIN7_DONE
