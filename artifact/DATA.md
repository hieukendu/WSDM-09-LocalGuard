# Data provenance

The three main datasets are public and were downloaded on 2026-10-05 by `code/download_data.sh`.
Raw files are not redistributed with this artifact; the script fetches them again.

| Dataset | File used | Source URL | Version / date | SHA-256 |
|---|---|---|---|---|
| LaMP-3 (product rating), user-based split, dev | `LaMP_3/dev_questions.json` | https://ciir.cs.umass.edu/downloads/LaMP/LaMP_3/dev/dev_questions.json | as served 2026-10-05 | ed7bb84c8f35bec9b9e7239d576cb476e935fd42c0a8d48320167c40a19da721 |
| | `LaMP_3/dev_outputs.json` | https://ciir.cs.umass.edu/downloads/LaMP/LaMP_3/dev/dev_outputs.json | as served 2026-10-05 | 7b459bd016990556b695118fe79185980a3d80572e70305228f06c43d2568972 |
| LaMP-7 (tweet paraphrasing), user-based split, dev | `LaMP_7/dev_questions.json` | https://ciir.cs.umass.edu/downloads/LaMP/LaMP_7/dev/dev_questions.json | as served 2026-10-05 | 0f53a6ebabdf4a80b5ee4150ad274af45af9a54d1a2825bf2aa302219afd44ea |
| | `LaMP_7/dev_outputs.json` | https://ciir.cs.umass.edu/downloads/LaMP/LaMP_7/dev/dev_outputs.json | as served 2026-10-05 | e59f7336f0a64de62af00ed633015f8b2b7d8b248a8816e3f53e5d3391df9a28 |
| LaMP-QA, Lifestyle and Personal Development, validation | `LaMP_QA/lifestyle_validation.json` | https://huggingface.co/datasets/alireza7/LaMP-QA (file `data/Lifestyle_and_Personal_Development/validation/validation.json`) | commit 1db3d5fcd986c11161797b934c2f34020ac0f0cf | 1739e01a6ca22b8f33f0be34e04c0c4d0ebfdbfa7b282d2bc74add234a35ee20 |

## Licences

- **LaMP** (Salemi et al., ACL 2024): the benchmark code and data-creation methods are
  CC BY-NC-SA 4.0; each task inherits the licence of its source dataset (LaMP-3: Amazon
  review data of Ni et al., EMNLP 2019; LaMP-7: Sentiment140 of Go et al., 2009).
  Use here is non-commercial research. No LaMP text is redistributed.
- **LaMP-QA** (Salemi and Zamani, EMNLP 2025): no licence is declared on the dataset
  card, the code repository or in the paper. The posts derive from StackExchange
  (SE-PQA). We treat the licence as unspecified, use the data for research only and
  redistribute none of it.

## Splits

Only the dev (LaMP) and validation (LaMP-QA) portions are used because their gold outputs
are public. In the LaMP user-based split every dev entry is a different user, and users are
additionally deduplicated by profile content. `code/prep.py` draws users in a fixed random
order (seed 20261005) and assigns disjoint users to the gate-training pool (120 users used)
and the evaluation pool (300 audited users per task, 450 for LaMP-3). `code/prep_extend.py` appended 200 LaMP-3
users by replaying the same random stream; the paper uses the first 450 evaluation users
(`N_EVAL_USE=450`), because the laptop run was stopped there. Reader calls cover 150 LaMP-QA
users, 450 LaMP-3 users and 300 LaMP-7 users. Validity uses 1000 random
calibration/test halvings; utility uses 2-fold cross-fitting over 200 partitions. No test
user is ever used to fit a scorer, a grid or a threshold.

## Model backends per cell

| Cell | Model | Backend |
|---|---|---|
| Reader, auditor, judge, counterfactuals, oracle (all main results) | qwen2.5:7b Q4_K_M | Ollama 0.30.8 `/api/generate`, temperature 0, seed 0 |
| Second auditor (sample of 300 and all top-8 candidates) | gemma2:9b (Ollama GGUF weights) | llama.cpp server bundled with Ollama 0.30.8, `/v1/chat/completions`, GGUF chat template, temperature 0, seed 0 |
| Replication reader | llama3.1:8b (Ollama GGUF weights) | same llama.cpp server |
| Judge of the replication outputs | qwen2.5:7b | Ollama (same as the main judge) |
| Stronger reader, LaMP-3 (first 300 evaluation users) and LaMP-QA pilot (40 users) | qwen2.5:14b Q4_K_M (Ollama library) | Ollama 0.30.8 `/api/generate`, temperature 0, seed 0, laptop |

No cell mixes backends, so no cross-backend fidelity check was needed. Cache entries written
after 2026-10-06 18:30 record their backend; earlier entries are Ollama for qwen2.5:7b and
llama.cpp for gemma2:9b and llama3.1:8b. The main three-task cells above did not use
the GPU servers offered by the protocol. The later LaMP-2 pilot is a separate endpoint
cell, described below.

## LaMP-2 pilot and pending extension

The movie-tagging pilot uses the new user-based dev split. Local raw file modification
times are 2026-10-07 09:36; these are not an independently verified download timestamp.
The earlier acquisition command/URL was not retained in this project, so the exact retrieval
provenance remains incomplete. No text is redistributed. Local SHA-256 checksums are:

- `LaMP_2/dev_questions_new.json`: `8db15c6df50e963f532c2f4f89f0f73fdf2da11443700023691d8c6374696e08`
- `LaMP_2/dev_outputs_new.json`: `651614c4b9a64febb842ee28940f883e018b643a62fbfa958a0aa310d45deccd`

The pilot has 60 evaluation users; the prepared extension has 150 evaluation users and
120 disjoint training users. Its reader is Qwen2.5-14B-Instruct Q4_K_M, endpoint 18140,
backend tag `llamacpp-6f3a9f3d-endpoint18140-Qwen2.5-14B-Instruct-Q4_K_M-official-split`,
temperature 0 and seed 0. Do not merge these outputs with the LaMP-3 Ollama 14B reader.
The two extension auditors use the same respective backends as the main tasks. The full
reader extension is incomplete; only the pilot utility is reported.

The secondary weighted/bin/grid analyses use the already-audited evaluation users:
300 LaMP-QA, 450 LaMP-3, 300 LaMP-7, and 150 LaMP-2. The synthetic cumulative-ranking
stress uses those same real profiles and existing labels, not new observed queries.


## Synthetic data

`code/sim_crc.py` generates synthetic users to check the exposure bound numerically.
Its output (`results/sim_crc.json`) is labelled synthetic and feeds no table of real results.

## Privacy handling

Records are user-authored public text. Prompt/response caches under `results/llm_cache/`
contain record text and are excluded from any public release; per-record auditor labels are
stored by record index only.

## How the data and pages were collected (polite-collection record)

- **LaMP files (ciir.cs.umass.edu).** Downloaded once on 2026-10-05 with curl and its default
  User-Agent. The first single-connection attempt ran at roughly 30 KB/s, so
  `code/download_data.sh` fetched the two large files with **parallel HTTP range requests
  from the same machine and IP: 4 concurrent connections for LaMP-7 `dev_questions.json`
  (3.6 MB) and 16 concurrent connections for LaMP-3 `dev_questions.json` (401 MB)**. The
  server returned no 429, no Retry-After and no block, and no proxy, tunnel or IP change was
  used. This does not meet the later protocol rule of fetching from one connection with a
  descriptive User-Agent, and it is recorded here plainly. robots.txt was not checked before
  the download. The small output files were fetched with single requests. Nothing was
  re-downloaded afterwards. A future re-run should use `curl -A "<project contact>"` with one
  connection and a rate limit (for example `--limit-rate 2M`), accepting the long download.
- **LaMP-QA (huggingface.co).** One file (82 MB) fetched with a single connection at a pinned
  commit. No rate limiting was encountered.
- **Model weights.** Encoder `sentence-transformers/all-MiniLM-L6-v2` and the spaCy model
  `en_core_web_sm-3.8.0` (GitHub release) were fetched once each through the standard
  library tools. Ollama models were already present and none was pulled.
- **Literature checks.** A helper agent queried the arXiv API, Crossref, ACL Anthology,
  PMLR and conference sites one request at a time. Some arXiv API queries were retried
  later after rate limiting, Semantic Scholar answered 429 throughout and was then left
  alone, and DBLP and OpenReview were unreachable or challenged and were not worked around.
  No proxy or connection switching was used.
