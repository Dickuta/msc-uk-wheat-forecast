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
| FR-3 | Weather→phenological-window alignment in one shared function | `aggregate_seasonal` in `src/weather.py` | enforced |
| FR-4 | One evaluation protocol: expanding window, horizons 1–4, paired RMSE/MAE | `ExpandingWindowCV` (`src/cv.py`), config-driven | enforced |
| FR-5 | Statistical comparison (Diebold–Mariano, multi-step correction) | `diebold_mariano` (`src/metrics.py:78-83`) | enforced |
| FR-6 | Uncertainty: PI coverage vs nominal | §5.7 of `scripts/05_Model.py`; verified in §5.10 | enforced |
| FR-7 | Bound the ceiling: oracle (perfect weather foresight) | §5.9 of `scripts/05_Model.py`, `oracle_fc` factories | enforced |
| FR-8 | Self-verify against canonical expected values, fail loudly | `verify()` (`05_Model.py:613-689`) | enforced |
| FR-9 | Practitioner decision guide, traceable to the numbers | §5.11 of `scripts/05_Model.py` → `data/outputs/decision_guide.md` | enforced |

## Non-functional requirements

| ID | Requirement | Enforcement | Status |
|---|---|---|---|
| NFR-1 | Leakage-safety: no test-period information in any shaping decision | ML tuning runs `TimeSeriesSplit` over the full series (`src/models.py:350-443`) | **disclosed** (see below) |
| NFR-2 | Protocol symmetry across model families | Statistical models: per-origin training-only selection; ML: global full-series tuning | **disclosed** (asymmetric) |
| NFR-3 | Small-sample honesty in inference | HLN DM correction, HAC autocovariances, Student-t T−1 df | enforced |
| NFR-4 | Bit-level reproducibility given fixed inputs + seed | single `config.SEED`, `random_state=42`, per-origin model cache | enforced |
| NFR-5 | No execution-order dependence | stages read fresh CSVs; mild smell: `02_EDA.py:155` mutates a local `decade` column | partial |
| NFR-6 | Traceability: every published number maps to an output CSV + code | `verify()` + `data/expected/manifest.md` | enforced |
| NFR-7 | Single source of truth for shared logic | `src/` imported by all stages; no copy-paste | enforced |
| NFR-8 | Configuration centralised | `config.py` (paths, seeds, horizons, grids) | enforced |
| NFR-9 | Address data by identity, not position | `verify()` keys on named tuples (`05_Model.py:616`); cv uses column names | enforced |
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

## Failure modes we've seen (and their mitigations)

| ID | Failure mode | Mitigation | Status |
|---|---|---|---|
| F-1 | Silent numerical misalignment (off-by-one, positional slices) | shared alignment fn; identity-keyed verify; `match` checks | enforced |
| F-2 | Silent leakage (tuning/selection/scaling touching the test period) | NFR-1 disclosure; per-origin selection for statistical models | disclosed |
| F-3 | The verify harness's own blind spot (expected == reproduced-itself) | `data/expected/manifest.md` pins origin + hashes; original thesis numbers archived in `data/thesis_reference/` | enforced |
| F-4 | Silent `except: continue` → unequal `n_test`, invisible apples-to-oranges | cv records `skipped_folds` + warns per horizon; §5.10 asserts per-horizon test-year symmetry across models and checks `n_test` against expected | enforced |
| F-5 | Dependency drift moves numbers without a code edit | pinned deps + recorded env | enforced |
| F-6 | Data-source drift (revised/rotten sources) | frozen raw + SHA-256 manifest | enforced |

## Cross-cutting principles

* **Epistemic integrity** — every methodological choice that could flatter a
  result is disclosed (ML tuning, canonical-vs-public weather). A benchmark that
  cannot be defended at a viva has failed its one job.
* **Auditability over time** — provenance manifests + versioned `expected/` +
  `PLAYBOOK.md` changelog turn "trust me" into "here's the trail."
