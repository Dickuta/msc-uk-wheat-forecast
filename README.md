# UK Wheat Yield Forecasting — Reproducible Pipeline

A modular, documented pipeline for the UK wheat yield forecasting study.
It reimplements the *corrected* Colab pipeline as five clean stages, all of
which run end-to-end from a **single consolidated notebook**
(`uk_wheat_pipeline_full.ipynb`) — in Colab or locally. The working copies of
the notebooks, stage scripts, tests and notes live outside the repo (in
`dont_track/`) and are intentionally not tracked by git.

```
uk_wheat_pipeline/
├── uk_wheat_pipeline_full.ipynb  # single Colab notebook: setup + stages 01–05
├── config.py                     # every path, URL, seed and hyperparameter
├── requirements.txt              # pinned runtime dependencies (see §dependencies)
├── pyproject.toml                # project metadata
├── src/                          # shared, documented code
│   ├── metrics.py                # RMSE, MAE, Diebold–Mariano (HLN-corrected)
│   ├── cv.py                     # expanding-window CV + baselines
│   ├── models.py                 # 8 model factories + ML tuning + predict_interval/oracle hooks
│   ├── features.py               # exogenous forecasting (ARIMA 1,0,0)
│   ├── weather.py                # seasonal phenological-window aggregation
│   └── plotting.py               # chart style (all charts render inline)
└── data/                         # committed (≈360K) so the repo runs offline on Colab
    ├── raw/                      # frozen Met Office series + manifest (stage 01 reuses them)
    ├── processed/                # modelling table (the single downstream input)
    ├── expected/                 # canonical outputs the pipeline verifies against
    ├── thesis_reference/         # archived original thesis numbers (viva trail)
    └── outputs/                  # result CSVs + decision guide (charts inline)
```

## How to run

Open `uk_wheat_pipeline_full.ipynb` and run it end-to-end (in Colab or with
Jupyter/VSCode). The notebook clones this repository, installs the pinned
`requirements.txt` dependencies, and executes stages 01 → 05 in order:
data acquisition, modelling table, EDA, feature engineering, and the model
comparison (CV, DM tests, prediction intervals, oracle experiment, and final
verification). The pipeline resolves its own root directory, so it works from
any location.

Stage 05 ends with an automatic **verification** that every produced number
matches the canonical thesis values in `data/expected/`.

## Run on Google Colab

The whole pipeline runs in Google Colab with a single notebook:

> [Open `uk_wheat_pipeline_full.ipynb` in Colab](https://colab.research.google.com/github/Dickuta/msc-uk-wheat-forecast/blob/master/uk_wheat_pipeline_full.ipynb)

From the Colab menu choose **Runtime → Run all**. It clones this repository,
installs the pinned `requirements.txt` dependencies (auto-restarting the
runtime once so numpy loads cleanly), then executes the five stages in order.
Everything the pipeline needs — including the frozen raw Met Office files in
`data/raw/` — is committed to the repo, so **no network downloads are required**
once cloned; stage 01 detects the frozen copies and skips the download.

Expect roughly 10–20 minutes on a standard Colab CPU runtime (stage 05 runs the
full expanding-window CV across 8 models).

## Dependencies

`requirements.txt` pins the exact runtime versions used to produce the thesis
numbers (including `numpy==2.3.5`, the last release that avoids the Colab
`_center`/`_blas_supports_fpe` import crash). The same pins are mirrored in
`pyproject.toml` (`pip install -e .` installs them).

## Data provenance (nothing is hard-coded)

| Stage | Reads from | Writes to |
|---|---|---|
| 01 | public sources (Met Office UK Climate Series), or frozen copies in `data/raw/` | `data/raw/` + `manifest.csv` |
| 02 | `data/raw/` + canonical building blocks | the modelling table |
| 03 | modelling table | inline charts only |
| 04 | modelling table | nothing (demonstrates feature construction) |
| 05 | modelling table | result CSVs (charts inline) |

* **Weather** — Met Office UK Climate Series (area-weighted UK means derived
  from HadUK-Grid 1 km), Open Government Licence. Downloaded by stage 01.
* **Yield / policy dummies** — canonical internal files (see thesis
  `Data/data_sources.md`): DEFRA/USDA national series and constructed policy
  indicators.
* **Canonical vs UK-mean weather** — the thesis numbers use a warmer regional
  (England-focused) weather extraction that cannot be rebuilt from the public
  UK-mean series. Stage 02 keeps the canonical table as ground truth and
  *quantifies* the difference so the provenance is fully transparent.

## Reproducing the thesis numbers

Run stages 01 → 05. The final cell of `05_Model` diffs all outputs against
`data/expected/` and prints `ALL CHECKS PASSED` when every thesis number is
reproduced.

## Known limitations & disclosures

* **ML hyperparameter tuning uses the full series.** RandomForest, XGBoost and
  ARIMA+XGBoost are tuned once over all 45 years with `TimeSeriesSplit(5)`
  (`src/models.py`), so the selected hyperparameters see data from 2000–2023 —
  years that are out-of-sample *test* years for the expanding-window evaluation.
  Their reported errors are therefore mildly optimistic relative to the strict
  no-leakage rule, and it is asymmetric with the statistical models, which pick
  structure per-origin from training data only. This reproduces the corrected
  Colab pipeline exactly, so it is **disclosed, not silently "fixed"** — changing
  it would move every ML number. If the results are ever regenerated, re-tuning
  on pre-2001 data only is the fix.
* **Canonical vs UK-mean weather.** The thesis numbers use a warmer
  England-focused weather extraction that cannot be rebuilt from the public
  UK-mean series. Stage 02 keeps the canonical table as ground truth and
  *quantifies* the difference so the provenance stays transparent.
* **`data/expected/` was regenerated from the corrected pipeline.** On
  2026-08-06 the verification target was rebuilt from the corrected stage 05
  because the refactor fixed the thesis's ARIMAX p-value-selection bug (the
  thesis read a positional `pvalues[-n_exog:]` slice; the corrected code reads
  p-values by name). ARIMAX RMSE/MAE move by ≈0.006/0.009 t/ha. The original
  thesis numbers are preserved in `data/thesis_reference/`, with a
  `changes_vs_thesis.csv` documenting every difference. `ALL CHECKS PASSED`
  therefore certifies that the corrected pipeline reproduces its own verified
  results; it does not re-certify the thesis numbers themselves.
