#!/bin/bash
# Continuation of repl_chain2.sh with the LaMP-QA replication reduced to no profile and top-5.
cd "$(dirname "$0")"
PY=../venv/bin/python
B=~/.ollama/models/blobs
SRV=/opt/homebrew/Cellar/ollama/0.30.8/libexec/lib/ollama/llama-server
export LLAMACPP_PORT_llama3_1_8b=18092 WORKERS=2
$PY run.py reader lampqa llama3.1:8b ../results/lampqa_contexts_repl.json
pkill -f "llama-server -m .*667b0c1932bc6ffc.* --port 18092"
$SRV -m $B/sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730 \
  --port 18093 --host 127.0.0.1 -c 16384 -np 4 -ngl 999 --jinja --no-webui > ../logs/llamaserver_qwen.log 2>&1 &
until curl -s http://127.0.0.1:18093/health | grep -q ok; do sleep 5; done
export LLAMACPP_PORT_qwen2_5_7b=18093 WORKERS=4
$PY serving_check.py
$PY run.py judge lampqa llama3.1:8b
echo REPL_DONE
