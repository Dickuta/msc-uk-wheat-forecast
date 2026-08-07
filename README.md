# UK Wheat Yield Forecasting — Reproducible Pipeline

A modular, documented pipeline for the UK wheat yield forecasting study.
It reimplements the *corrected* Colab pipeline as five clean stages, each
available **both** as a Python file (VSCode-native, `# %%` cells) **and** as
an executed Jupyter notebook (`.ipynb`) with figures and tables rendered
inline.

```
uk_wheat_pipeline/
├── config.py                 # every path, URL, seed and hyperparameter
├── requirements.txt          # pinned runtime dependencies (see §dependencies)
├── requirements-dev.txt      # test + notebook tooling
├── pyproject.toml            # project metadata + pytest config
├── PLAYBOOK.md               # live maintenance checklist
├── src/                      # shared, documented code
│   ├── metrics.py            # RMSE, MAE, Diebold–Mariano (HLN-corrected)
│   ├── cv.py                 # expanding-window CV + baselines
│   ├── models.py             # 8 model factories + ML tuning + predict_interval/oracle hooks
│   ├── features.py           # exogenous forecasting (ARIMA 1,0,0)
│   ├── weather.py            # seasonal phenological-window aggregation
│   └── plotting.py           # chart style (all charts render inline)
├── stages/                  # the five stages as runnable .py files
│   ├── 01_Data_Acquisition.py       # acquire raw data from public sources
│   ├── 02_Modelling_Table.py        # assemble the modelling table
│   ├── 03_EDA.py                    # exploratory data analysis
│   ├── 04_Feature_Engineering.py    # model-ready features
│   └── 05_Model.py                  # comparison, DM, PIs, oracle, verify
├── notebooks/                # the same five stages as executed .ipynb
│   ├── 01_Data_Acquisition.ipynb
│   ├── 02_Modelling_Table.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   └── 05_Model.ipynb
├── tests/                    # fast unit tests (pytest)
└── data/
    ├── raw/                  # downloaded by 01 (Met Office series + manifest)
    ├── processed/            # modelling table (the single downstream input)
    ├── expected/             # canonical outputs the pipeline verifies against
    ├── thesis_reference/     # archived original thesis numbers (viva trail)
    └── outputs/              # result CSVs + decision guide (charts inline)
```

## How to run

Run the notebooks in order (or their `.py` twins inside VSCode). The scripts
resolve the pipeline root themselves, so they work from any directory:

```bash
python stages/01_Data_Acquisition.py   # or open notebooks/01_Data_Acquisition.ipynb
python stages/02_Modelling_Table.py
python stages/03_EDA.py
python stages/04_Feature_Engineering.py
python stages/05_Model.py              # heavy: ~30–70 min on CPU
```

Stage 05 ends with an automatic **verification** that every produced number
matches the canonical thesis values in `data/expected/`.

Each notebook is executed end-to-end, so every chart renders inline in the
`.ipynb` (charts are never saved as image files).

## Dependencies

`requirements.txt` pins the exact runtime versions used to produce the thesis
numbers; `requirements-dev.txt` adds `pytest` and the notebook-tooling
(`jupytext`, `nbconvert`, `nbformat`) used to keep the `.py`/`.ipynb` twins in
sync. The same pins are mirrored in `pyproject.toml` (`pip install -e ".[dev]"`
installs both).

## Tests

```bash
python -m pytest tests/ -q
```


## Data provenance (nothing is hard-coded)

| Stage | Reads from | Writes to |
|---|---|---|
| 01 | public sources (Met Office UK Climate Series) | `data/raw/` + `manifest.csv` |
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
  2026-08-06 the verification target was rebuilt from `stages/05_Model.py`
  because the refactor fixed the thesis's ARIMAX p-value-selection bug (the
  thesis read a positional `pvalues[-n_exog:]` slice; the corrected code reads
  p-values by name). ARIMAX RMSE/MAE move by ≈0.006/0.009 t/ha. The original
  thesis numbers are preserved in `data/thesis_reference/`, with a
  `changes_vs_thesis.csv` documenting every difference. `ALL CHECKS PASSED`
  therefore certifies that the corrected pipeline reproduces its own verified
  results; it does not re-certify the thesis numbers themselves.
