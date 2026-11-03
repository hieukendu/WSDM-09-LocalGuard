#!/bin/bash
# Re-run of the pure-numpy analyses after the 2026-10-08 additions to analyze.py (special-category
# exposure of the fixed references, MINIM-style fixed-threshold rows). Same seeds, same inputs.
cd "$(dirname "$0")"
PY=../venv/bin/python
N_EVAL_USE=300 $PY analyze.py lampqa > ../logs/analyze_lampqa.log 2>&1; echo A_lampqa
N_EVAL_USE=300 $PY analyze.py lamp7 > ../logs/analyze_lamp7.log 2>&1; echo A_lamp7
N_EVAL_USE=450 $PY analyze.py lamp3 > ../logs/analyze_lamp3.log 2>&1; echo A_lamp3
N_EVAL_USE=450 $PY analyze.py lamp3 qwen25_14b > ../logs/analyze_lamp3_14b.log 2>&1; echo A_lamp3_14b
N_EVAL_USE=450 $PY analyze.py lamp3 llama31_8b > ../logs/analyze_lamp3_llama.log 2>&1; echo A_lamp3_llama
# 7B reader restricted to the first 300 evaluation users, the user set of the 14B table
N_EVAL_USE=300 OUT_SUFFIX=_300users $PY analyze.py lamp3 > ../logs/analyze_lamp3_300.log 2>&1; echo A_lamp3_300
$PY replication.py > ../logs/replication.log 2>&1; echo A_repl
echo REANALYZE_DONE
