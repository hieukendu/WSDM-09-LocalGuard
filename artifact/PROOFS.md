# Proofs

## Setting

A user is a triple `Z = (q, P, a)`: a query `q`, a profile `P` of personal records, and
auditor labels `a(r) in {0,1}` for the records (1 = the auditor flags the record as
disclosing). An admission rule is a family of maps `A_lambda(q, P) ⊆ P`, indexed by a
threshold `lambda` in a finite grid `Lambda = {lambda_0 < lambda_1 < ... < lambda_J}`, with

- (cap) `|A_lambda(q, P)| <= K` for all `lambda`;
- (empty start) `A_{lambda_0}(q, P) = ∅`.

The scorers inside `A_lambda` are fixed functions: they are fitted on training users that are
independent of the calibration and test users. The rule may use any on-device feature of
`(q, P)` but never the auditor labels.

Exposure of user `Z` at `lambda` is `E(Z, lambda) = sum_{r in A_lambda(q,P)} a(r)`, an integer
in `[0, K]`. The monotone envelope is `L(Z, lambda_j) = max_{j' <= j} E(Z, lambda_{j'})`.
By construction `L` is nondecreasing in `lambda`, `L(Z, lambda_0) = 0`, `0 <= L <= B := K`
and `E <= L` pointwise. If the family is nested (`A_lambda ⊆ A_lambda'` for
`lambda <= lambda'`), then `E` is already nondecreasing and `L = E`.

Calibration users `Z_1, ..., Z_n` come with auditor labels. Write
`R_n(lambda) = (1/n) sum_i L(Z_i, lambda)` and define

    lambda_hat = max( {lambda_0} ∪ { lambda in Lambda : (n R_n(lambda) + B) / (n + 1) <= alpha } ).

When no threshold satisfies the inequality (which happens when `alpha < B/(n+1)`), the rule
falls back to `lambda_0` and admits nothing.

## Proposition 1 (exposure certificate)

Assume `Z_1, ..., Z_n, Z_{n+1}` are exchangeable. Then for every `alpha >= 0`

    E[ E(Z_{n+1}, lambda_hat) ]  <=  E[ L(Z_{n+1}, lambda_hat) ]  <=  alpha,

where the expectation is over the calibration users and the test user jointly.

### Proof

The first inequality is `E <= L`. For the second, write `L_i(lambda) = L(Z_i, lambda)` and let

    R_{n+1}(lambda) = (1/(n+1)) sum_{i=1}^{n+1} L_i(lambda),
    lambda' = max { lambda in Lambda : R_{n+1}(lambda) <= alpha }.

`lambda'` is well defined since `R_{n+1}(lambda_0) = 0 <= alpha`.

Step 1 (`lambda_hat <= lambda'`). If `lambda_hat = lambda_0` there is nothing to show.
Otherwise, since `L_{n+1}(lambda_hat) <= B`,

    R_{n+1}(lambda_hat) = (n R_n(lambda_hat) + L_{n+1}(lambda_hat)) / (n+1)
                        <= (n R_n(lambda_hat) + B) / (n+1) <= alpha,

so `lambda_hat` belongs to the set whose maximum is `lambda'`.

Step 2 (monotonicity). `L_{n+1}` is nondecreasing, hence
`L_{n+1}(lambda_hat) <= L_{n+1}(lambda')`.

Step 3 (exchangeability). `lambda'` is a symmetric function of `(Z_1, ..., Z_{n+1})`.
Exchangeability therefore gives, for every `i`,
`E[L_{n+1}(lambda')] = E[L_i(lambda')]`, and averaging over `i`

    E[L_{n+1}(lambda')] = E[ (1/(n+1)) sum_i L_i(lambda') ] = E[ R_{n+1}(lambda') ] <= alpha.

Combining the three steps proves the claim. ∎

This is the argument of Angelopoulos et al. (Conformal Risk Control, ICLR 2024, Theorem 1)
mirrored to a nondecreasing loss, on a finite grid, with the monotone envelope handling
admission rules that backfill and are therefore not nested.

### Remarks

1. **User level.** The exchangeable unit is the user. If a user issues several queries, take
   `E(Z, lambda)` to be the mean (or the maximum) exposure over that user's queries. The
   loss stays in `[0, K]` and monotone after the envelope, and the proposition bounds the
   expected per-query (or worst-query) exposure of a new user. Pooling queries of one user
   as if they were independent units would break exchangeability.
2. **What is certified.** The bound is on exposure as measured by the auditor. If the
   auditor misses a fraction of truly disclosing records, the true exposure can be larger.
   The bound is marginal over the draw of calibration users, so individual calibration draws
   can exceed `alpha`; the experiments report the fraction that do.
3. **Score-agnostic.** Nothing is assumed about the quality of the necessity or sensitivity
   scorers. Poor scorers cost utility (a smaller `lambda_hat`), never validity.
4. **Feasibility.** A non-trivial `lambda_hat` requires `alpha >= B/(n+1)`.
5. **Second loss.** The same statement holds for any other bounded per-user loss that is
   made monotone by the envelope, with `B` replaced by its bound. The paper uses the
   indicator `Z(S) = 1[E(S) >= 1]` (`B = 1`, so the correction is `1/(n+1)`) and exposure
   restricted to special categories (`B = K`).
6. **Fixed in advance.** The grid, the family and its scores must not depend on the
   calibration or test users (they are fitted on separate training users in the artifact),
   otherwise `lambda'` is no longer a symmetric function of exchangeable losses.
7. **High-probability variant.** Learn-then-Test (Angelopoulos et al., Ann. Appl. Stat. 2025)
   with Hoeffding-Bentkus p-values (Bates et al., JACM 2021) for `H_j: R(lambda_j) > alpha`,
   losses scaled to `[0, 1]` by `B`, tested in increasing order of `lambda` and stopping at the
   first non-rejection (fixed-sequence testing controls the family-wise error at `delta`
   without a union bound). Then `P(R(lambda_hat) <= alpha) >= 1 - delta` over the calibration
   draw, where `R` is the population risk. Implemented in `code/crc.py::ltt_index`.

## Numeric check

`code/sim_crc.py` draws synthetic users (labelled synthetic), with user-level heterogeneity
in base rates, noisy and even anti-correlated sensitivity scores, and a backfilling rule
whose raw exposure is not monotone. Over 4000 calibration draws for each of 48
configurations (`n` in {50, 100, 200, 400}, three targets, four score regimes), the mean
monotone loss of the fresh user never exceeds `alpha`; see `results/sim_crc.json`.
