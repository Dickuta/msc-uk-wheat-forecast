# `data/thesis_reference/` — original thesis numbers (archived)

These six CSVs are the **original canonical thesis outputs**, extracted from the
corrected Colab pipeline and the thesis Results chapter. They are the numbers
`ALL CHECKS PASSED` previously certified against.

## Why this directory exists

On 2026-08-06 `data/expected/` was regenerated from the corrected pipeline (see
`../expected/manifest.md`). The only substantive difference is **ARIMAX**:

* the thesis generated ARIMAX covariate selection from a **positional p-value
  slice** (`fitted.pvalues[-n_exog:]`), which under the statsmodels parameter
  layout `[const, x1..xk, ar.L1, sigma2]` reads `ar.L1` and `sigma2` as if they
  were covariates and never inspects the first exogenous parameter;
* the corrected pipeline reads p-values **by name**, so ARIMAX selects
  different covariates and its RMSE/MAE move by ≈0.006/0.009 t/ha across the
  four horizons.

Rounding-level library drift (4th decimal) also moves the hybrid
(ARIMA+XGBoost) h=3 RMSE and a handful of DM statistics. Everything else is
byte-identical.

These files are preserved so the thesis viva trail stays intact: anyone can
still see the exact numbers the thesis published, and the exact corrected
numbers, side by side.

## Files

Identical to `data/expected/` before regeneration (2026-08-06). Recompute
hashes with `sha256sum data/thesis_reference/*.csv`.
