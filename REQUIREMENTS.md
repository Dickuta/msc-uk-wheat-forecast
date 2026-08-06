# UK Wheat Pipeline — Requirements & Guarantees

Living statement of what this system is **supposed to guarantee**, and where in
the code each guarantee is enforced. Update it when the code changes; a task is
only "enforced" when the enforcement point exists and is verified.

Status legend: **enforced** = a code path enforces it · **disclosed** = true but
intentional, documented · **partial** = partly enforced · **open** = not yet.

## The essence

This is not a production service. It is a reproducible scientific instrument: it
runs once, on a laptop, over ~45 annual observations, to produce numbers a
thesis stakes its credibility on. Latency, throughput and availability are
irrelevant. Correctness, reproducibility and leakage-safety dominate, because
the entire contribution is *an honest, standardised evaluation framework*.

## Functional requirements

| ID | Requirement | Enforcement | Status |
|---|---|---|---|
| FR-1 | Acquire raw public data reproducibly, with provenance (URL, checksum, date) | `scripts/01` download + `manifest.csv` (`01_Data_Acquisition.py:163-183`) | enforced |
| FR-2 | One canonical modelling table; sole input to every downstream stage | `03` writes it; 02/04/05 read only `uk_wheat_modelling_table_1980_2024.csv` | enforced |
| FR-3 | Weather→phenological-window alignment in one shared function | `aggregate_seasonal` (`src/weather.py`); runtime spot-check `assert_alignment` (`01_Data_Acquisition.py:242-268`) | enforced |
| FR-4 | One evaluation protocol: expanding window, horizons 1–4, paired RMSE/MAE | `ExpandingWindowCV` (`src/cv.py`), config-driven | enforced |
| FR-5 | Statistical comparison (Diebold–Mariano, multi-step correction) | `diebold_mariano` (`src/metrics.py:78-83`) | enforced |
| FR-6 | Uncertainty: PI coverage vs nominal | §5.7 of `scripts/05_Model.py`; verified in §5.10 | enforced |
| FR-7 | Bound the ceiling: oracle (perfect weather foresight) | §5.9 of `scripts/05_Model.py`, `oracle_fc` factories | enforced |
| FR-8 | Self-verify against canonical expected values, fail loudly | `verify()` (`05_Model.py:613-689`) | enforced |
| FR-9 | Practitioner decision guide, traceable to the numbers | §5.11 of `scripts/05_Model.py` → `data/outputs/decision_guide.md` | enforced |

## Non-functional requirements

| ID | Requirement | Enforcement | Status |
|---|---|---|---|
| NFR-1 | Leakage-safety: no test-period information in any shaping decision | Runtime guards `assert_training_precedes_origin`, `assert_horizon_consistent` (`src/guards.py:18-42`) at every CV fold; ML full-series tuning still disclosed | **enforced (structural + runtime)** |
| NFR-2 | Protocol symmetry across model families | `warn_on_swallowed_fits` per-model (`cv.py:172`); `assert_balanced_test_sets` in verify (`05_Model.py:656`); ML tuning asymmetry still disclosed | **partial (runtime guards + disclosure)** |
| NFR-3 | Small-sample honesty in inference | HLN DM correction, HAC autocovariances, Student-t T−1 df | enforced |
| NFR-4 | Bit-level reproducibility given fixed inputs + seed | single `config.SEED`, `random_state=42`, per-origin model cache | enforced |
| NFR-5 | No execution-order dependence | stages read fresh CSVs; mild smell: `02_EDA.py:155` mutates a local `decade` column | partial |
| NFR-6 | Traceability: every published number maps to an output CSV + code | `verify()` + `data/expected/manifest.md` | enforced |
| NFR-7 | Single source of truth for shared logic | `src/` imported by all stages; no copy-paste | enforced |
| NFR-8 | Configuration centralised | `config.py` (paths, seeds, horizons, grids) | enforced |
| NFR-9 | Address data by identity, not position | `select_exog_pvalues` guard (`src/guards.py:63-86`) used in `arimax_stepwise_selection` (`src/models.py:122`); `verify()` keys on named tuples (`05_Model.py:616`); cv uses column names | enforced |
| NFR-10 | Prose–data consistency (no markdown claim not re-derived) | `verify()` gates numbers; README figures reconciled | partial |
| NFR-11 | Data-source longevity: frozen raw + loud, specific failure | frozen-raw reuse + friendly error in `01_Data_Acquisition.py:77-108` | enforced |
| NFR-12 | Dependency rot is the #1 long-term risk | pinned `requirements.txt`; env recorded in `PLAYBOOK.md` §8 | enforced |
| NFR-13 | Extensibility is the real scaling axis | new model = new factory in `src/models.py`; new covariate = config entry | enforced (convention) |

### NFR-1 / NFR-2 — the tuning disclosure (read this)

`tune_rf` / `tune_xgb` select hyperparameters once over **all 45 years**
(`TimeSeriesSplit(5)`, `src/models.py:350-443`, called at `05_Model.py:198-207`),
including 2000–2023 — the out-of-sample test years of the evaluation. The ML
errors are therefore mildly optimistic and asymmetric vs the statistical models.

This reproduces the corrected Colab pipeline exactly, so the numbers are
thesis-faithful. It is **disclosed, not silently fixed**: re-tuning on pre-2001
data only would change every ML result and require regenerating
`data/expected/`. Do that only if the results are deliberately regenerated.

**Runtime leakage guards now active:** `assert_training_precedes_origin` and
`assert_horizon_consistent` are called at every CV fold (`src/cv.py:118-119`);
`warn_on_swallowed_fits` warns when folds are dropped (`src/cv.py:172`);
`assert_balanced_test_sets` raises if any model sees a different test set
(`scripts/05_Model.py:656`). These catch structural leakage but do not undo
the disclosed ML tuning asymmetry.

## Failure modes we've seen (and their mitigations)

| ID | Failure mode | Mitigation | Status |
|---|---|---|---|
| F-1 | Silent numerical misalignment (off-by-one, positional slices) | shared alignment fn; identity-keyed verify; `assert_alignment` spot-check in 01; `select_exog_pvalues` refuses positional slices | enforced |
| F-2 | Silent leakage (tuning/selection/scaling touching the test period) | `assert_training_precedes_origin` + `assert_horizon_consistent` at every fold; `warn_on_swallowed_fits`; ML full-series tuning still disclosed | **enforced (structural)** + disclosed (ML tuning) |
| F-3 | The verify harness's own blind spot (expected == reproduced-itself) | `data/expected/manifest.md` pins origin + hashes; original thesis numbers archived in `data/thesis_reference/` | enforced |
| F-4 | Silent `except: continue` → unequal `n_test`, invisible apples-to-oranges | cv records `skipped_folds` + warns per horizon; `warn_on_swallowed_fits`; §5.10 `assert_balanced_test_sets` raises on asymmetry | enforced |
| F-5 | Dependency drift moves numbers without a code edit | pinned deps + recorded env | enforced |
| F-6 | Data-source drift (revised/rotten sources) | frozen raw + SHA-256 manifest | enforced |

## Cross-cutting principles

* **Epistemic integrity** — every methodological choice that could flatter a
  result is disclosed (ML tuning, canonical-vs-public weather). A benchmark that
  cannot be defended at a viva has failed its one job.
* **Auditability over time** — provenance manifests + versioned `expected/` +
  `PLAYBOOK.md` changelog turn "trust me" into "here's the trail."
