# Artifact: Exposure-Controlled Data Minimization for Personalized LLM Prompts

Every number, table and figure in the paper is produced by the scripts in `code/` from raw
model outputs saved in `results/`. There are no hand-entered numbers.

## Layout

- `code/download_data.sh`: fetches LaMP-3, LaMP-7 and LaMP-QA (see `DATA.md` for URLs,
  versions, checksums and licences).
- `code/prep.py`: samples users (seed 20261005), builds candidate pools (MiniLM similarity,
  BM25 scores) and random baseline draws.
- `code/common.py`: model access with an append-only prompt cache, prompts (reader, auditor,
  rubric judge), utility metrics.
- `code/run.py`: LLM stages: `audit` (auditor labels), `detect` (Presidio + lexicon),
  `cf` (counterfactual single-record utilities of training users), `reader`, `judge`, `score`.
- `code/gate.py`: fits the necessity and sensitivity scorers on training users and writes the
  admission sets of every rule for evaluation users, plus training-only threshold grids.
- `code/crc.py`: the conformal risk control rule (and a Hoeffding variant).
- `code/analyze.py`: certificate validity over 1000 calibration/test splits, matched-alpha
  utility, user-bootstrap intervals, minimality gap.
- `code/oracle.py`: greedy counterfactual approximation of the minimal sufficient set.
- `code/sim_crc.py`: synthetic numeric check of Proposition 1 (labelled synthetic).
- `code/validate_audit.py`: agreement between the primary and the second auditor.
- `code/replication.py`: second-reader replication.
- `code/make_assets.py`: writes `../manuscript/generated/` (macros, tables, figure).
- `PROOFS.md`: full proof of Proposition 1 and the assumptions.

## Reproduce

```bash
bash code/download_data.sh
python3.13 -m venv venv && venv/bin/pip install numpy scipy scikit-learn pandas matplotlib \
    requests rank_bm25 rouge-score presidio-analyzer spacy sentence-transformers torch
# plus the spaCy model en_core_web_sm 3.8.0
cd code
../venv/bin/python prep.py lamp3 lamp7 lampqa
for T in lampqa lamp3 lamp7; do
  ../venv/bin/python run.py audit $T; ../venv/bin/python run.py detect $T; ../venv/bin/python run.py cf $T
  ./pipeline_rest.sh $T          # score/judge, gate, reader, oracle (first 150 users)
done
./chain4_qwen.sh                 # reader, auditor, judge for all 300 evaluation users (Ollama)
./chain4_cpp.sh                  # llama3.1:8b replication reader, gemma2:9b second auditor (llama.cpp)
./chain6.sh; ./chain8.sh          # LaMP-7 to 300 users, LaMP-3 to 450 evaluation users (Ollama), gemma labels
../venv/bin/python run.py reader lamp3 llama3.1:8b ../results/lamp3_contexts_repl.json  # (llama.cpp)
../venv/bin/python sim_crc.py
../venv/bin/python run.py audit lampqa gemma2:9b 300   # 300-record agreement sample, also lamp3, lamp7
for T in lampqa lamp7; do N_EVAL_USE=300 ../venv/bin/python analyze.py $T; done
N_EVAL_USE=450 ../venv/bin/python analyze.py lamp3
../venv/bin/python analyze.py lamp3 llama31_8b
./chain9_14b.sh                  # qwen2.5:14b reader: LaMP-3 table and LaMP-QA pilot (Ollama)
N_EVAL_USE=450 ../venv/bin/python analyze.py lamp3 qwen25_14b
for T in lampqa lamp7; do N_EVAL_USE=300 ../venv/bin/python shift_mech.py $T; done; N_EVAL_USE=450 ../venv/bin/python shift_mech.py lamp3
../venv/bin/python qa_pilot.py; ../venv/bin/python validate_audit.py; ../venv/bin/python replication.py; ../venv/bin/python make_assets.py
```

Models: see `DATA.md` ("Model backends per cell"). With the prompt cache in
`results/llm_cache/` every stage replays without model calls.

## Notes

- `results/ablation_prompt_v1/` keeps the first LaMP-QA counterfactual run with an earlier
  reader prompt (records placed before the question). It feeds no table.
- `results/llm_cache/` holds user-authored text from the benchmarks and must not be published.
- `results/*_audit*.json` hold per-record self-disclosure labels (health, finance, sexual
  orientation and others) of real pseudonymous authors that can be joined back to the public
  benchmarks. They are sensitive derived data and are excluded from any public release.
- Hardware: one Apple Silicon laptop (64 GB), shared with other jobs.

## 2026-10-10 revision experiments

`code/mean_rating_baseline.py` evaluates the full-history mean-only control on all
450 main LaMP-3 users with Qwen-7B, the original prompt suffix and decoding.
`results/lamp3_mean_rating_summary.json` is aggregate; the predictions and LLM cache
remain private. The mean itself is personal data despite zero admitted record text.

`code/revision_model_checks.py repeat --model MODEL` makes fresh auditor calls in a
separate cache (32 uniformly sampled top-eight records per task). Use
`N_EVAL_USE=450 WORKERS=3` and the original model backends. Gemma uses the bundled
llama.cpp server and `LLAMACPP_PORT_gemma2_9b=18109` in this revision.
`revision_model_checks.py judge` scores every saved main LaMP-QA reader context with
Gemma and writes a distinct utility file. It never overwrites the Qwen-judge results.
Then run `N_EVAL_USE=300 ../venv/bin/python analyze.py lampqa qwen25_7b_judge_gemma2_9b`.
The model weights and manifests are in `results/auditor_provenance_20261010.json`;
SHA-256 was recomputed over each GGUF. No model weights were downloaded.

`make_assets.py` generates the main 7B table and separate 14B table.
`make_revision_supplement.py` generates the new supplement's evidence and exact prompt
listings from aggregate outputs. Compile `manuscript/main.tex` and `supplement.tex`.
A missing second-judge summary is explicitly rendered as a run in progress.

`human_full_pool.py prepare PRIVATE_DIRECTORY` creates a distinct blind study with
60 LaMP-3 users, 30 calibration and 30 test, and all eight candidates per user.
Each of two humans labels 480 independently shuffled records. Supply only each
person's CSV and INSTRUCTIONS.md; retain keys and cohort assignments privately.
After genuine labels and truthful attestations, use `human_full_pool.py score
PRIVATE_DIRECTORY`, adjudicate disagreements as requested, then score again.
The rule/grid hashes must remain unchanged. The original sparse packet remains
available but cannot identify the same complete-cohort operating point.
None of these packets, keys, individual labels, predictions or cached outputs belongs
in a public or anonymous submission artifact.
