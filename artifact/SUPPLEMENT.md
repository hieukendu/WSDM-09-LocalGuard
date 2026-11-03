# Supplement: scope, sensitivity analyses, and unresolved evidence

This supplement separates measured aggregate results from planned work. All disclosure
results remain relative to the named auditors. They are not human-validated privacy rates.
The generated tables in `SECONDARY-TABLES.md` are produced from the released aggregate JSON.

## Human audit and operating-point uncertainty

The complete-pool LaMP-3 human audit is now finished privately. It contains 480 records
from 60 uniformly selected users: 30 calibration users and 30 disjoint held-out test users,
with all eight candidates per user and two independent annotation passes. Thirty-one
category-set disagreements were adjudicated. Any-disclosure agreement is 0.9708 with
Cohen's kappa 0.883. The public release contains only the aggregate summary; packets,
keys, attestations, adjudication rows and individual labels remain excluded. The workflow
cannot verify an attestation's truth; the study coordinator must obtain genuine independent
human judgments.

Only 96/120 records have Gemma labels: 34/40 LaMP-QA, 27/40 LaMP-3, 35/40 LaMP-7.
Comparisons of Qwen, Gemma and their union use the same covered subset; missing Gemma
labels are never treated as negatives. A separate Qwen-only descriptive row uses all
covered sample items. Because selection was stratified on Qwen's binary decision, these
unweighted metrics describe this audit sample, not population precision or recall. Population
estimates require the original sampling frame/inclusion probabilities, with uncertainty
clustered by user. Rare categories may have zero positive examples; undefined recall is
reported as missing, not perfect agreement.

The completed packet identifies full per-user human labels for this small LaMP-3 cohort.
The scores and threshold grid were frozen before annotation, calibration uses the first
30 users, and held-out measurement uses the disjoint last 30 users. At alpha=.2 and .5,
human recalibration selected the same grid index as the union-auditor calibration. These
are descriptive results for this LaMP-3 sample, not population estimates and not validation
of LaMP-QA or LaMP-7.

The earlier sparse 120-record packet remains a separate descriptive check and does not
identify full per-user losses. The completed full-pool audit is LaMP-3-only and does not
establish human validation for the other tasks.

`human_audit.py score --packet PRIVATE_DIRECTORY --results PRIVATE_RESULTS` now computes
partial-identification diagnostics after complete two-human annotation/adjudication of
the packet. For each user and rule it bounds loss between known human-positive admitted
records and that count plus all unknown admitted records. It takes the running-maximum
envelope separately for both bounds. The CRC index under the upper loss is a lower bound
on the fully human-labelled CRC index; the index under the lower loss is an upper bound.
These are deterministic missing-label bounds, not confidence intervals. The diagnostic
uses fixed seeded user folds, freezes the union-calibrated policy before human measurement,
and reports held-out loss bounds and calibration-only operating-point index bounds.
No candidate identity or individual label is exported. Wide intervals are reported as
non-identification; exact human results are never fabricated from a partially labelled pool.

## Query exclusion and minimality

Every evaluated rule sends the user's query, including the empty-profile baseline. Thus
100% of evaluated prompts contain an excluded query, but its disclosure prevalence has
not been measured. Treating the query as one record gives total exposure
`E_total = E_profile + a(query)`. Since `0 <= a(query) <= 1`, a profile-only expected
budget alpha implies at most alpha + 1 for this total-record measure, not alpha.
For alpha = 0.2 this worst-case bound is 1.2 records. It is not an estimate of actual
query leakage. A full-prompt policy needs query labels, a bound of K+1, and a genuine
zero-loss fallback such as local processing or abstention; sending only the query is
not such a fallback. Semantic inference and category severity are also not captured
by the binary record count. “Minimality” means only auditor-flagged records within the
candidate pool, not all personal data or all information inferable by the provider.

## Weighted exposure and fixed-bin calibration

The secondary analysis fixes a normative severity sensitivity before recalibration:
each flagged record costs 2 if either auditor assigns HEALTH, MENTAL_HEALTH,
SEXUAL_ORIENTATION, RACE_NATIONALITY or FINANCE, otherwise 1; unflagged records cost 0.
Overlapping categories use their maximum weight, not a sum. The bounded loss has B=10
for K=5. These weights are illustrative policy choices, not a validated harm scale or
legal categorization. Comparing at the same numerical alpha deliberately makes the
weighted budget stricter; the units differ from the uniform record-count budget.
No utility claim is inferred from record counts.

Profile-length bin cutpoints are medians of the 120 training users, not medians estimated
from evaluation users. Across 300 matched random user-level calibration/test halves,
we compare pooled and separate within-bin calibration, retain the actual calibration
size in each bin, and report average held-out loss. Conditional expected-risk control
requires exchangeability within each fixed bin. This does not assert simultaneous
high-probability control of every bin or a bound for each individual user. For an empty
bin the deployment fallback must be empty admission. Even with zero empirical loss,
nonempty CRC admission requires n >= ceil(B/alpha)-1: for alpha=.2, at least 24
calibration users under count loss and 49 under severity2. Counts and infeasible draws
are included in the aggregate JSON. The earlier evaluation-median split in the original
analysis is retained as exploratory shift evidence, not used to support the revised
fixed-bin guarantee claim.

## Finer cleared-rule grid

The original eight thresholds are retained and augmented with 41 training-score quantiles
(duplicates removed). The sensitivity proxy is refitted on the same 120 training users;
the script aborts unless its evaluation scores reproduce the saved scores within 1e-7.
This changes only the discretization, not scores or calibration/test labels. The running
maximum is recomputed over the entire expanded grid. A finer grid can increase this
envelope as well as reduce threshold quantization, so it need not uniformly improve utility.
The exposure/admission analysis is complete. Additional reader contexts are missing;
the report counts them by backend and does not report utility on a selected complete-case
subset. Completing reader calls and confirming utility remains open.

## Cumulative unique disclosures: synthetic ranking stress only

The raw files contain no exact full-profile group with multiple distinct query strings
under the diagnostic implemented in `cumulative_exposure.py`. Different profiles might
still belong to the same person; we do not infer real identity or claim that the diagnostic
establishes distinct humans. The evaluated data were deliberately deduplicated to one query.

We therefore report a limited, explicitly synthetic arrival/ranking sensitivity on real
profiles: 1, 2, 5 or 10 independently shuffled rankings of the same already-audited top-8
pool, with the original query-specific sensitivity scores fixed. Each arrival admits at
most five cleared records. Unique flagged record IDs and unique category names are
deduplicated within user across arrivals, rather than summed as independent disclosures.
The comparison is single-prompt calibration versus direct user-level cumulative-loss
calibration, using the same calibration/test halves. The cumulative loss bound is
min(8,5T), not 5T after deduplication within this fixed pool. Categories are descriptive;
they are not the calibrated loss. The user, not an arrival, is the exchangeable unit.

This is not an empirical multi-query benchmark: there are no new query texts, retrieval
never reaches outside the top-8 pool, and no utility is measured. The simulation cannot
establish safety for real changing queries, adaptive interaction, or new records. For
general sequences, unique disclosures are at most the sum of per-prompt disclosures;
T*alpha applies only if each prompt's marginal assumptions and calibration remain valid.
Observed multi-query validation with full-profile retrieval and newly audited records remains open.

## Larger personalization gain

The declared LaMP-2 pilot uses 60 users with no-profile and top-5 contexts, Qwen2.5-14B
on its original endpoint backend, and a paired user bootstrap. The aggregate gain and
interval are reported in the main text and generated table. The pilot passed its declared
go criterion (gain at least .05 and positive 95% lower bound). It is not independent
confirmation: complete the 150-user run, explicitly separate the 60 pilot users from the
90 additional users, and report the added-user contrast before claiming replication.
No unfinished reader or held-out auditor result is represented as complete.

## Reproduction and distribution

`MANIFEST.json` is the exact allowlist with file sizes and SHA-256 hashes. The archive
contains code, proof/provenance documentation, aggregate results, generated tables/figure,
and, only after a separate content/privacy check, anonymous manuscript sources/PDF.
The manifest explicitly states whether the manuscript is included. It excludes all raw/processed benchmark data,
embeddings, per-user utilities and admission sets, record indices, model outputs/caches,
auditor labels, annotation packets/keys, logs, operational plans, review notes, credentials,
and Git metadata. Code identifiers describing schemas are not actual joinable record IDs.
Aggregate JSON is rejected if it contains row-level identity/text/category fields.
The privacy-preserving export is a distinct submission artifact, not a claim that a private
development repository or its historical commits are anonymous.

## 2026-10-10 additions

The compiled companion `../manuscript/supplement.pdf` now contains the 14B table,
LaMP-2 pilot, weighted exposure, exact auditor and judge prompt templates, model
manifest/GGUF digests and fresh auditor agreement. Both auditors reproduced all
96 sampled binary flags and category sets (32 records per task). This is repeatability,
not human validation.

The 450-user Qwen-7B mean-past-rating-only control has utility 0.893 versus 0.883
without profile and 0.922 with top-5. It retains 26.8% [1.3,48.4] of the top-5 gain
(90% paired user-bootstrap interval), significantly below top-5. It sends a personal
statistic and therefore must not be described as information-free.

A complete-pool human audit contains 480 records from 60 uniformly selected LaMP-3 users
(30 calibration / 30 held-out test). Both annotation passes and the adjudication are now
complete. Its code freezes rule/grid hashes and reports precision/recall plus held-out human
exposure and CRC-index changes. It does not establish validation on the other two tasks.

The Gemma second-judge run is now complete on all 3,184 contexts for 150 users.
Top-5 minus no profile is -0.006 [-0.031,+0.022], versus Qwen +0.016
[-0.005,+0.040] on shared paired user-bootstrap draws (95%). The compiled supplement
includes the full four-rule comparison and parsing diagnostics: Gemma has 3,184 valid
complete vectors; Qwen 3,043, with 101 wrong lengths and 42 coercion/clipping cases
(overlap possible). Original parser behavior is retained. No human validation is inferred.
