#!/bin/bash
# LaMP-3 frozen at the first 450 evaluation users (laptop throughput; planned 600). Reader outputs
# and utilities are rebuilt from the cache, gemma2:9b labels the top-8 candidates of users
# 300-450 (llama.cpp), then the analysis runs.
cd "$(dirname "$0")"
PY=../venv/bin/python
B=~/.ollama/models/blobs
SRV=/opt/homebrew/Cellar/ollama/0.30.8/libexec/lib/ollama/llama-server
export N_EVAL_USE=450 N_READER=450 WORKERS=3
$PY gate.py lamp3 > ../logs/gate450_lamp3.log
$PY run.py reader lamp3
$PY run.py score lamp3
$SRV -m $B/sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373 \
  --port 18091 --host 127.0.0.1 -c 8192 -np 4 -ngl 999 --jinja --no-webui > ../logs/llamaserver_gemma.log 2>&1 &
until curl -s http://127.0.0.1:18091/health | grep -q ok; do sleep 5; done
LLAMACPP_PORT_gemma2_9b=18091 WORKERS=4 $PY run.py audit lamp3 gemma2:9b evaltop
pkill -f "llama-server -m .*ff1d1fc78170d787.* --port 18091"
$PY analyze.py lamp3 > ../logs/analyze_lamp3.log 2>&1
echo CHAIN8_DONE
