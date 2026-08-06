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
| `oracle_exogenous_results.csv` | `dd8223c54a5a001d3feb2fc7379386ea45f4fe9d12470fb5ada0e0083e4e2a12` |
| `pi_coverage_results_corrected.csv` | `5c2ff1ba237e31d6c98e5fd14a35e2f285dbf1803c4e5bf4b98522d4decff6e4` |
| `pi_detailed_results_corrected.csv` | `519c9e24852e05501d091264a078922f14be598381e99c863975d4def684e7f8` |

Verified by a full `05` run on 2026-08-06 (`ALL CHECKS PASSED`). Two caveats on
the hashes above:

* `model_comparison_results_corrected.csv` also carries `avg_train_time_s` /
  `avg_peak_memory_mb` — measured wall-clock metrics that legitimately vary
  run-to-run and are **excluded** from the verify gate. If a later run's hash
  differs here and only here, that is expected, not a regression.
* The `verify()` gate compares the remaining columns under absolute tolerances
  (1e-4 to 1e-6); the hashes are a stronger fingerprint of the checkpoint
  itself, not of what a fresh run must reproduce exactly.

Baseline and PI files are byte-identical to the thesis.
