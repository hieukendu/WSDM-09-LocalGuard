# LocalGuard

Reproducibility code and aggregate results for **Exposure-Controlled Data Minimization for Personalized LLM Prompts**.

This repository is prepared for a server checkout. It contains the analysis and model-call code, public-data download script, aggregate results, proofs, and the manuscript snapshot.

## Privacy boundary

The public repository deliberately excludes raw or processed benchmark text, embeddings, per-user utilities and admission sets, prompt/response caches, row-level auditor or human labels, human annotation packets and keys, logs, credentials, and internal review correspondence.

The benchmark files are downloaded by `artifact/code/download_data.sh`. Do not commit `artifact/data/`, `artifact/results/llm_cache/`, `artifact/results/handcheck/`, or newly generated row-level result files.

## Server setup

Use Python 3.13 or a compatible recent Python:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
bash artifact/code/download_data.sh
```

The main experiment uses Qwen2.5-7B through Ollama. The second auditor and replication reader use Gemma2-9B and Llama3.1-8B through the llama.cpp backend described in `artifact/DATA.md`. The stronger reader uses Qwen2.5-14B. Model weights are intentionally not included.

Run the deterministic checks before any model calls:

```bash
python -m unittest discover -s artifact/code -p 'test_*.py'
python artifact/code/sim_crc.py
```

The original pipeline commands and sample sizes are documented in `artifact/README.md` and `artifact/DATA.md`. Existing aggregate tables can be regenerated with:

```bash
python artifact/code/make_assets.py
python artifact/code/make_secondary_tables.py
```

Full reader/auditor reruns require the model services and will create private caches under `artifact/results/llm_cache/`.

## Human audit status

The completed LaMP-3 human audit is summarized in `PUBLIC-STATUS.md` and `artifact/HUMAN-AUDIT-RESULT.md`. Individual human annotations and the private join key are excluded from this repository.

## Manuscript

The manuscript source and PDFs are included as a snapshot. The human-audit aggregate was completed after the current PDF snapshot was generated; update the stale human-audit wording in the manuscript source and recompile before treating the PDF as final.
