# Anonymous reproducibility supplement

The submission artifact is this allowlisted package, not the private development archive.
Read `artifact/SUPPLEMENT.md` for scope, caveats and pending human/server work, and
`MANIFEST.json` for exact files, sizes and SHA-256 checksums.

## Included

- Generated aggregate tables and figure. The manuscript PDF and sources are included
  only when `manuscript_included` in the manifest is true; otherwise they are withheld
  pending a separate content/privacy check.
- Executable analysis/model-call source, prompt templates (not cached prompts), tests.
- Aggregate result JSON and generated secondary tables; proof and data provenance.

## Deliberately excluded

Raw/processed benchmark text, embeddings, per-user data or utility, admission sets,
record/user indices, sensitive per-record labels, outputs/caches, human sheets and keys,
logs, machine notes, credentials, review correspondence and Git history are not distributed.
No individual annotation or model output is needed to render the released aggregate tables.
Reproducing audits/experiments from scratch requires obtaining the public benchmarks under
their terms and running the model calls again; the package cannot replay sensitive caches.

## Verify and render

Use Python 3.13 with numpy, scipy, scikit-learn, matplotlib, requests, pandas,
sentence-transformers, torch, rank_bm25, rouge-score, presidio-analyzer and spaCy
(en_core_web_sm 3.8.0 for the detector). `artifact/DATA.md` describes model/backend cells.

```bash
python -m unittest discover -s artifact/code -p 'test_*.py'
python artifact/code/make_assets.py
python artifact/code/make_secondary_tables.py
cd manuscript
# Only when manuscript_included is true:
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The released aggregates are sufficient for the two table-generation commands. To recompute
secondary analyses, first regenerate private processed data, labels, proxy scores and rule
sets using `prep.py`, `run.py` and `gate.py`, following the sample sizes in `DATA.md`.
Then run `secondary_exposure.py lampqa lamp3 lamp7 lamp2` and
`cumulative_exposure.py lampqa lamp3 lamp7 lamp2`. The latter is explicitly a synthetic
ranking/arrival stress, not observed multi-query evidence. Never release the regenerated
row-level files. Human audit code requires actual independent human annotations; synthetic
unit-test fixtures do not constitute annotation results.

For private human annotations, `human_audit.py coverage --packet PRIVATE_DIRECTORY
--results PRIVATE_RESULTS` reports structural pool coverage without scoring blanks.
After two independent human passes and adjudication, `human_audit.py score --packet
PRIVATE_DIRECTORY --results PRIVATE_RESULTS` adds deterministic missing-label exposure
and operating-point bounds. These bounds are not population estimates or confidence intervals.

Manuscript compliance must be checked on the exact separately submitted PDF. Rendering
uses acmart; the revision uses ragged-bottom columns and disables automatic final-reference
balancing to avoid a TeX Live 2020 balancing overflow. Fonts, margins and content size
are unchanged. An artifact-only export makes no assertion about an unbundled PDF.
