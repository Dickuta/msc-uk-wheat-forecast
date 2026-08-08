# `data/expected/` — provenance manifest

This directory holds the values that stage `05` verifies against (the `verify()`
cell / `ALL CHECKS PASSED`).

## Origin — read this first

**Regenerated 2026-08-08 from the corrected pipeline** (`stages/05_Model.py`,
run of the same date). These files are the output of this repository's
corrected code, not the thesis's original numbers.

* The only substantive change vs the thesis is **ARIMAX**, caused by a
  deliberate bug fix: covariate selection now reads p-values **by name**
  instead of the thesis's positional slice `pvalues[-n_exog:]` (which treated
  `ar.L1` / `sigma2` as covariates). ARIMAX RMSE/MAE move by ≈0.006/0.009 t/ha.
* Rounding-level library drift (4th decimal) also moves the hybrid h=3 RMSE
  and a few DM statistics.
* The original thesis numbers are archived in `../thesis_reference/` with their
  own manifest.

This 2026-08-08 regeneration additionally extends the verification schema — all
**additive** (no change to any RMSE/MAE point: max abs diff = 0.0 vs the 2026-08-06
checkpoint):

* DM tests now report **both MSE and MAE** losses, with a per-horizon
  **Bonferroni** significance flag (`significant_bonferroni`, `bonferroni_alpha`
  = 0.05 / n_pairs).
* Prediction-interval experiment now covers **SARIMA and ARIMAX** (in addition
  to ARIMA and Prophet) at 95%, reporting `interval_score` (Gneiting & Raftery,
  2007) alongside coverage and width.
* ADF diagnostics (constant and constant+trend) were added to stage 03.
* A Jupyter kernel had been racing this run (rewriting `decision_guide.md` and
  `arima_residual_ljungbox.csv` with stale content); it was killed before this
  checkpoint was written, so every file below was produced by a single clean
  stage-05 execution.

Consequence: `ALL CHECKS PASSED` certifies that **this pipeline reproduces its
corrected, verifiable results**. If you regenerate these files again, record the
reason and update this manifest — otherwise the check silently becomes
meaningless.

## Files

| File | SHA-256 |
|---|---|
| `baseline_results.csv` | `787369c7d539fb8666835c823fec2c4c5debd0cf61797dc6a842d256607c83fe` |
| `dm_test_results.csv` | `77faf6e699bb2ac08e409cfb34df2e2b90aab35b086b226e2b9dd13498a2c981` |
| `model_comparison_results_corrected.csv` | `96a1b657ea6777bfafacc4ad2b9badc5c4774b004e5b7caa5a1d5069a8fe94fe` |
| `model_details_results_corrected.csv` | `41f66e1cc3c64c1fc58385bb4f32863c7b0b6b6e3dc47d37947da3c1af2f3bb8` |
| `oracle_exogenous_results.csv` | `7e1f8ef5717414fd7930941f3983b14c96103b934aea5a79834d9bc1497aef29` |
| `pi_coverage_results_corrected.csv` | `745dfa43ab47a7784cb6336645b4a049463ebc3dbe34edd85df0fb6e55de496e` |
| `pi_detailed_results_corrected.csv` | `c89fc3ace408c5eccdf06425dc75696c2d84e417de2004b593589cd691015572` |
| `decision_guide.md` | `ab6320cd7c03a2c8854e183425b1413636046c34c68832928ff54008ecb6fc8e` |
| `arima_residual_ljungbox.csv` | `47101ef1a76f1f6734fcf03db099aab3c57bdc877f0846ea71c855ae0407cce7` |

Verified by full `05` runs on 2026-08-06 and 2026-08-08 (latest, with oracle
h=1 fix). Two caveats on the hashes above:

* `model_comparison_results_corrected.csv` also carries `avg_train_time_s` /
  `avg_peak_memory_mb` — measured wall-clock metrics that legitimately vary
  run-to-run and are **excluded** from the verify gate. If a later run's hash
  differs here and only here, that is expected, not a regression.
* `decision_guide.md` and `arima_residual_ljungbox.csv` are regenerated
  deliverables (not gated by `verify()`), but are still hashed here as
  change-detection fingerprints.
* The `verify()` gate compares every column under absolute tolerances
  (1e-4 to 1e-6), with PI coverage metrics (`pi_coverage_95`, `avg_pi_width`,
  `avg_interval_score`, `n_test`) matched as exact strings. The hashes are a
  stronger fingerprint of the checkpoint itself, not of what a fresh run must
  reproduce exactly.

Baseline and PI files are byte-identical to the thesis. The oracle file gains one
row (ARIMA h=1 context, 0% improvement) due to the off-by-one fix documented in
`changes_vs_thesis.csv`.
