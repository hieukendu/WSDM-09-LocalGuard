#!/bin/bash
# llama.cpp cells: llama3.1:8b replication reader (LaMP-QA) and gemma2:9b second auditor on the
# top-8 candidates of all evaluation users. Same weights and server as their earlier calls.
cd "$(dirname "$0")"
PY=../venv/bin/python
B=~/.ollama/models/blobs
SRV=/opt/homebrew/Cellar/ollama/0.30.8/libexec/lib/ollama/llama-server
$SRV -m $B/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29 \
  --port 18092 --host 127.0.0.1 -c 8192 -np 2 -ngl 999 --jinja --no-webui > ../logs/llamaserver_llama.log 2>&1 &
until curl -s http://127.0.0.1:18092/health | grep -q ok; do sleep 5; done
LLAMACPP_PORT_llama3_1_8b=18092 WORKERS=2 $PY run.py reader lampqa llama3.1:8b ../results/lampqa_contexts_repl.json
pkill -f "llama-server -m .*667b0c1932bc6ffc.* --port 18092"
echo LLAMA_READER_DONE
$SRV -m $B/sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373 \
  --port 18091 --host 127.0.0.1 -c 8192 -np 4 -ngl 999 --jinja --no-webui > ../logs/llamaserver_gemma.log 2>&1 &
until curl -s http://127.0.0.1:18091/health | grep -q ok; do sleep 5; done
for T in lamp3 lamp7 lampqa; do LLAMACPP_PORT_gemma2_9b=18091 WORKERS=4 $PY run.py audit $T gemma2:9b evaltop; echo GEMMA $T; done
pkill -f "llama-server -m .*ff1d1fc78170d787.* --port 18091"
# judge of the llama outputs uses the main judge backend (Ollama qwen2.5:7b)
WORKERS=2 $PY run.py judge lampqa llama3.1:8b
echo CPP_DONE
