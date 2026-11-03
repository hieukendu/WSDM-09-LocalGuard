#!/bin/bash
# Held-out third auditor: Llama-3.3-70B-Instruct Q4_K_M on the shared endpoint :18170 (llama.cpp
# 6f3a9f3d, see ../../../ENDPOINTS.md), same auditor prompt, top-8 candidates of the first
# 150 (LaMP-3) / 100 (LaMP-QA, LaMP-7) evaluation users. Cached per call, so it resumes.
cd "$(dirname "$0")"
PY=../venv/bin/python
export LLAMACPP_PORT_llama3_3_70b=18170 WORKERS=4
export LLAMACPP_BACKEND="llamacpp-6f3a9f3d-endpoint18170-Llama-3.3-70B-Instruct-Q4_K_M-sha256-32df3bac"
N_EVAL_USE=150 $PY run.py audit lamp3 llama3.3:70b evaltop; echo L70 lamp3
N_EVAL_USE=100 $PY run.py audit lampqa llama3.3:70b evaltop; echo L70 lampqa
N_EVAL_USE=100 $PY run.py audit lamp7 llama3.3:70b evaltop; echo L70 lamp7
echo CHAIN10_DONE
