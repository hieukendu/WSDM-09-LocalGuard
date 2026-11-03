#!/bin/bash
# Round-1 revision compute, sequential to limit memory on the shared laptop:
# (1) reader calls for all 300 evaluation users of LaMP-3 and LaMP-7 through a private llama.cpp
#     server with the Ollama qwen2.5:7b weights; (2) second auditor (gemma2:9b) on the top-8
#     candidates of all evaluation users; (3) the same extension for LaMP-QA; (4) re-analysis.
cd "$(dirname "$0")"
PY=../venv/bin/python
B=~/.ollama/models/blobs
SRV=/opt/homebrew/Cellar/ollama/0.30.8/libexec/lib/ollama/llama-server
QW=$B/sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730
GM=$B/sha256-ff1d1fc78170d787ee1201778e2dd65ea211654ca5fb7d69b5a2e7b123a50373
start_qwen() { $SRV -m $QW --port 18093 --host 127.0.0.1 -c 16384 -np 4 -ngl 999 --jinja --no-webui >> ../logs/llamaserver_qwen.log 2>&1 &
               until curl -s http://127.0.0.1:18093/health | grep -q ok; do sleep 5; done; }
extend() { T=$1
  N_READER=300 $PY gate.py $T > ../logs/gate300_$T.log
  N_READER=300 $PY run.py audit $T
  N_READER=300 $PY run.py reader $T
  if [ $T = lampqa ]; then $PY run.py judge $T; else $PY run.py score $T; fi
  echo EXTENDED $T; }
until grep -q REPL_DONE ../logs/repl_llama.log; do sleep 30; done
export LLAMACPP_PORT_qwen2_5_7b=18093 WORKERS=4
curl -s http://127.0.0.1:18093/health | grep -q ok || start_qwen
extend lamp3; extend lamp7
pkill -f "llama-server -m .*2bada8a7450677.* --port 18093"
$SRV -m $GM --port 18091 --host 127.0.0.1 -c 8192 -np 4 -ngl 999 --jinja --no-webui > ../logs/llamaserver_gemma.log 2>&1 &
until curl -s http://127.0.0.1:18091/health | grep -q ok; do sleep 5; done
for T in lamp3 lamp7 lampqa; do LLAMACPP_PORT_gemma2_9b=18091 $PY run.py audit $T gemma2:9b evaltop; echo GEMMA $T; done
pkill -f "llama-server -m .*ff1d1fc78170d787.* --port 18091"
for T in lamp3 lamp7; do $PY analyze.py $T > ../logs/analyze_$T.log 2>&1; done
start_qwen
extend lampqa
pkill -f "llama-server -m .*2bada8a7450677.* --port 18093"
$PY analyze.py lampqa > ../logs/analyze_lampqa.log 2>&1
echo CHAIN3_DONE
