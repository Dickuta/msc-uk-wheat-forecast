# UK Wheat Pipeline — Maintenance Playbook

Living checklist for the maintenance & hardening work on the pipeline. Update the
status of each item as work proceeds. A task is only "Done" when its verification
step passes.

Legend: `[ ]` pending · `[~]` in progress · `[x]` done

---

## 0. Ground rules

- **Reproducibility is sacred.** Stage 05 ends with a `verify()` step against
  `data/expected/`; any refactor that changes produced numbers (beyond the
  documented tolerances) is a regression and must be reverted or re-verified.
- **No code comments unless they explain *why*.** The stage scripts are prose
  notebooks; keep them readable.
- **One source of truth for logic.** Anything reused (model specs, CV, exog
  forecasting, seasonal alignment) lives in `src/`, not in `stages/`.
- **Entry-point discipline.** Each stage script exposes `def main()` and an
  `if __name__ == "__main__": main()` guard. `main.py` imports and calls
  `main()` via `importlib` (no subprocess), so stages are composable units.

## 1. File organisation (DONE)

- [x] Stage `.py` files moved to `stages/` (`stages/01_Data_Acquisition.py` … `05_Model.py`)
- [x] Stage notebooks moved to `notebooks/` (`notebooks/01_Data_Acquisition.ipynb` … `05_Model.ipynb`)
- [x] Stage names disambiguated: `01_Data_Processing` → `01_Data_Acquisition`, `03_Data_Processing` → `03_Modelling_Table` (scripts + notebooks + docs)
- [x] `__pycache__/` removed (root + `src/`)
- [x] `README.md` structure diagram + run commands updated
- [x] Re-run `05_Model` from the moved layout (executed `notebooks/05_Model.ipynb`,
      2026-08-06) → `ALL CHECKS PASSED`; outputs repopulated, decision guide
      written

## 2. Reproducibility & environment hardening (DONE except commit decision)

- [x] `requirements.txt` pinned to the exact installed versions (see §8)
- [x] `requirements-dev.txt` added (test/notebook tooling)
- [x] `pyproject.toml` added (project metadata + pytest config)
- [x] `.gitignore` added (pycache, notebook checkpoints, outputs, data)
- [x] `git init` in the pipeline root
- [x] Decide: commit the executed notebooks? — **Committed everything on
      2026-08-06** (`bf53691`), notebooks included (user decision). Repo-local
      git identity `dickson <dickson@localhost>` set.

## 3. Code — remove duplication (`src/` owns the logic) (DONE except full verify)

- [x] `src/models.py`: `fit_best_arima(y, p_range, d_range, q_range)` extracted and
      used by `arima_factory` + hybrid, preserving the `p → d → q` loop order and
      AICc tie-breaking
- [x] `src/models.py`: Prophet refactored into helpers
      (`_select_prophet_changepoint_scale`, `_fill_future_covariates`) shared by
      the CV factory and the PI path; `interval_width` now explicit (0.95)
- [x] `src/models.py`: `predict_interval(h)` added to ARIMA + Prophet predictors
- [x] `src/models.py`: optional `oracle_fc` argument added to ARIMAX, Prophet, RF,
      XGBoost and hybrid factories (replaces `features.forecast_exogenous`)
- [x] `stages/05_Model.py`: deleted the duplicated oracle predictors
      (`RFPredictorOracle`, `XGBPredictorOracle`, `HybridPredictorOracle`),
      `evaluate_oracle`, `patch_forecast`, `arima_predictor`, `fit_prophet_pi`,
      `prophet_fcst_with_pi`
- [x] `stages/05_Model.py`: PI + oracle sections rebuilt on `src.models` factories
      (oracle run is now just `run_cv(name, partial(factory, oracle_fc=...))`)
- [x] `src/models.py`: stepwise-selection docstring corrected to describe the
      constant-column short-circuit (statsmodels `ValueError` → keep full set;
      this IS the verified behaviour, do not "fix")
- [x] Verify: `data/outputs/*` match `data/expected/*` on a full `05` run
      (executed notebook, 2026-08-06) → **`ALL CHECKS PASSED`**. **Note:** on
      2026-08-06 `data/expected/` was regenerated from the corrected pipeline
      (ARIMAX p-value bug fix — see §6); the original thesis numbers are
      archived in `data/thesis_reference/`. The gate now certifies the corrected
      pipeline reproduces its own verified results, and `changes_vs_thesis.csv`
      documents every difference from the thesis numbers.

## 4. Robustness fixes (DONE)

- [x] Scripts location-independent: `sys.path` root resolves from `__file__` with a
      `Path.cwd()` fallback for notebooks, in all 5 scripts (+ the 4 patched notebooks)
- [x] Headless-safe charting: stages 02/04/05 import `src.plotting` so `python stages/*.py`
      works without a display (no more `TkAgg` crash); notebooks still render inline
- [x] `src/cv.py`: warns when predictions fail on a fold instead of silently skipping
- [x] `src/cv.py`: `expanding_windows`/`evaluate_baseline` take `initial_train_end`
      (passed `config.INITIAL_TRAIN_END` from `05`)
- [x] `src/cv.py`: `ExpandingWindowCV.model_name` added — explicit name with a
      fallback for `functools.partial` factories (the oracle run); this fixed the
      `AttributeError: 'partial' object has no attribute '__name__'` that crashed
      the first full `05` run in §5.9
- [x] `src/cv.py`: `ExpandingWindowCV.skipped_folds` records `(horizon,
      test_year, error)` for every prediction that failed and was skipped (F-4);
      `run_cv` prints per-horizon skip warnings; `05` §5.10 asserts test-year
      symmetry across models per horizon
- [x] `stages/01_Data_Acquisition.py`: frozen-raw fallback (NFR-11) — if
      `data/raw/` already holds the file, reuse it (no network); a failed
      download raises a friendly `RuntimeError`; manifest timestamp from file stat
- [x] `stages/05_Model.py`: §5.11 "Practitioner decision guide" (FR-9) writes
      `data/outputs/decision_guide.md` from the verified result CSVs (best model,
      per-horizon best, DM significances, PI coverage, oracle ceiling); summary
      renumbered §5.12. `config.OUTPUT_FILES["decision_guide"]` added
- [x] `src/guards.py` added — six runtime invariant guards (LeakageError,
      AlignmentError, `assert_training_precedes_origin`, `assert_horizon_consistent`,
      `select_exog_pvalues`, `assert_alignment`, `assert_balanced_test_sets`,
      `warn_on_swallowed_fits`); each maps to a specific silent-failure mode
- [x] `src/models.py`: `arimax_stepwise_selection` now uses
      `select_exog_pvalues` (NFR-9 / F-1) — raises if a covariate is not in
      fitted `param_names`, eliminating the positional-slice bug class
- [x] `src/cv.py`: fold-level `assert_training_precedes_origin` +
      `assert_horizon_consistent` (NFR-1); `warn_on_swallowed_fits` replaces
      the inline warning with a quantified message (NFR-2 / F-4)
- [x] `stages/05_Model.py`: §5.10 verify uses `assert_balanced_test_sets`
      (NFR-2 / F-4) — raises instead of soft print
- [x] `stages/01_Data_Acquisition.py`: §1.4 spot-checks seasonal alignment
      with `assert_alignment` (FR-3 / F-1) on a reference harvest year
- [x] Notebook twins: `notebooks/01` and `03` re-synced from the renamed scripts
      via `jupytext --update` (outputs preserved); `05_Model.ipynb` regenerated
      from the refactored script — re-run 05 to repopulate outputs
- [x] `src/weather.py` now listed in README + `config.py` docstring layout
- [x] `stages/02_Modelling_Table.py`: §2.4 validation now **raises** on failure
      (was print-only); `to_csv` deferred to after the `assert all(checks.values())`
      so a broken table never reaches downstream stages
- [x] `stages/05_Model.py`: PI loops (`compute_arima_pi`, `compute_prophet_pi`)
      now track `skipped` per model and emit `warn_on_swallowed_fits` (was silent
      `except: continue`); return `(DataFrame, skipped_count)` so callers can report
      the gap; `assert_balanced_test_sets` added on `pi_details` for PI protocol
      symmetry (NFR-2 / F-4)
- [x] `stages/05_Model.py`: Oracle ARIMA loop horizon range fixed from `[2,3,4]`
      to `[1,2,3,4]` — h=1 context rows were being dropped, making the oracle
      comparison table incomplete for ARIMA across all 4 horizons
- [x] `stages/01_Data_Acquisition.py`: `payload` `UnboundLocalError` on fresh
      downloads fixed (was only bound inside the retry loop; now initialised
      before); winter seasonal alignment guard (`assert_alignment_spanning_year`)
      added spot-check; duplicate seasonal spot-check print removed
- [x] `main.py`: converted from `subprocess.call` to `importlib`-based
      import-and-call; each stage script now exposes `main()` (Entry-point
      discipline — see §0), stages are composable units, error handling
      distinguishes AssertionError from other exceptions

## 5. Tests (DONE)

- [x] `tests/` with fast, deterministic unit tests: `metrics`, `weather.aggregate_seasonal`
      (incl. year-boundary windows), `features.forecast_exogenous` (short/constant series),
      `cv.expanding_windows`/`evaluate_baseline`/`ExpandingWindowCV`,
      `models.fit_best_arima`/`predict_interval`/oracle plumbing/stepwise behaviour
- [x] `tests/test_guards.py` — 19 tests for the six invariant guards (test count grew as edge
      cases were added: boundary-span, missing-covariate, multi-model imbalance)
- [x] `conftest.py` at root so `pytest` resolves `src` without installation
- [x] `pytest` installed (9.1.1) + 43 tests passing from the pipeline root
- [ ] Optional: add a GitHub Actions workflow to run `pytest` on push

## 6. Docs (DONE)

- [x] `config.py` docstring directory layout updated (stages/, notebooks/, `src/weather.py`)
- [x] `README.md`: tree shows `src/weather.py`, `tests/`, `PLAYBOOK.md`,
      `requirements-dev.txt`, `pyproject.toml`, `REQUIREMENTS.md`; "Dependencies" +
      "Tests" sections added; "invoke from pipeline root" note removed
      (location-independent now)
- [x] `REQUIREMENTS.md` created — living FR/NFR/failure-mode tables, each mapped
      to the code that satisfies it; ML tuning leak documented under NFR-1/2
- [x] `data/expected/manifest.md` — provenance rewritten: target regenerated
      2026-08-06 from the corrected pipeline; hashes for unchanged baseline/PI
      files retained, `TODO` for the changed dm/comparison/oracle files (fill
      after the next full run with `sha256sum`)
- [x] `data/thesis_reference/` — original thesis CSVs archived before
      regeneration, with `manifest.md` (why it differs) and `changes_vs_thesis.csv`
      (per-file diff ledger; 39 changed rows)
- [x] `README.md` "Known limitations & disclosures": ML tuning uses full series
      (disclosed, not silently fixed); canonical weather caveat; `expected/`
      regeneration note + pointer to `thesis_reference/`

## 7. Deferred (deliberately not done)

- **Parallelise the CV** — the oracle/PI paths reseed numpy per origin and
  Prophet draws from the global RNG; thread/process parallelism would change
  draws → break byte-identical reproduction. Revisit only if run-time becomes a
  blocker, and re-verify `ALL CHECKS PASSED` if you do.
- **Cache Prophet changepoint selection** — each origin fits on a different
  window, so cross-origin caching doesn't apply. Would need a persisted artifact
  store; out of scope for a thesis pipeline.
- **Full packaging / `pip install -e .`** — the root-path bootstrap makes installs
  unnecessary; revisit if the pipeline becomes a library.

## 8. Environment (recorded 2026-08-06)

Python 3.13.11. Pinned in `requirements.txt` / `pyproject.toml`:

| Package | Version |
|---|---|
| numpy | 2.5.1 |
| pandas | 3.0.5 |
| scipy | 1.18.0 |
| scikit-learn | 1.9.0 |
| statsmodels | 0.14.6 |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |
| requests | 2.32.5 |
| prophet | 1.3.0 |
| xgboost | 3.4.0 |

Dev-only (`requirements-dev.txt` / `[project.optional-dependencies] dev`):

| Package | Version |
|---|---|
| pytest | 9.1.1 |
| jupytext | 1.19.3 |
| nbconvert | 7.16.6 |
| nbformat | 5.10.4 |

## 9. Verification commands

```bash
python -m pytest tests/ -q            # unit tests (43 pass)
python main.py --stage 04             # run stages 01–04 via import (no subprocess)
python stages/05_Model.py            # stage 05: 30–70 min, ends with ALL CHECKS PASSED
```
