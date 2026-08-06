# `data/expected/` — provenance manifest

This directory holds the values that stage `05` verifies against (the `verify()`
cell / `ALL CHECKS PASSED`).

## Origin — read this first

**Regenerated 2026-08-06 from the corrected pipeline** (`scripts/05_Model.py`,
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

Consequence: `ALL CHECKS PASSED` certifies that **this pipeline reproduces its
corrected, verifiable results**. If you regenerate these files again, record the
reason and update this manifest — otherwise the check silently becomes
meaningless.

## Files

| File | SHA-256 |
|---|---|
| `baseline_results.csv` | `787369c7d539fb8666835c823fec2c4c5debd0cf61797dc6a842d256607c83fe` |
| `dm_test_results.csv` | `90f17b7953c6777719a00a3e0ab79b3bf24dcf8ffe46a3f6cdac6f1f907abe1f` |
| `model_comparison_results_corrected.csv` | `cff12d731bc48920ffb0bcd8abefa0cb3b18088276aff06c5238bc86865320d1` |
| `oracle_exogenous_results.csv` | `7e1f8ef5717414fd7930941f3983b14c96103b934aea5a79834d9bc1497aef29` |
| `pi_coverage_results_corrected.csv` | `bbb6f2978d00b89a8422206e019c727c20ac5ebd25b4da19b330514fa40e90d5` |
| `pi_detailed_results_corrected.csv` | `af9b12fbef06391cfafa8b9a172c37b780471ed45858cd83d624f9708c55bdd1` |

Verified by full `05` runs on 2026-08-06 and 2026-08-06 (latest, with oracle h=1
fix). Two caveats on the hashes above:

* `model_comparison_results_corrected.csv` also carries `avg_train_time_s` /
  `avg_peak_memory_mb` — measured wall-clock metrics that legitimately vary
  run-to-run and are **excluded** from the verify gate. If a later run's hash
  differs here and only here, that is expected, not a regression.
* The `verify()` gate compares the remaining columns under absolute tolerances
  (1e-4 to 1e-6); the hashes are a stronger fingerprint of the checkpoint
  itself, not of what a fresh run must reproduce exactly.

Baseline and PI files are byte-identical to the thesis. The oracle file gains one
row (ARIMA h=1 context, 0% improvement) due to the off-by-one fix documented in
`changes_vs_thesis.csv`.
