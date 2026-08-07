# UK Wheat Pipeline — Code Workflow & Architecture

> A visual + narrative guide for how the codebase fits together, why each piece exists, and where/when it runs.

---

## 🎯 QUICK RECALL: The Pipeline in 3 Sentences

> **"We download UK weather, merge it with wheat yields and policy events into ONE modelling table, then run 8 forecasting models through an expanding-window CV that self-verifies against canonical thesis numbers."**

**Memory hooks:**
- **1 table** → `uk_wheat_modelling_table_1980_2024.csv` (the contract)
- **5 stages** → 01 Acquire → 02 Explore → 03 Assemble → 04 Demo → 05 Model & Verify
- **1 gate** → Stage 05 `verify()` prints `ALL CHECKS PASSED` or fails loud
- **6 guards** → `src/guards.py` catches leakage, misalignment, swallowed folds at runtime

---

## 0. Project Structure at a Glance

```
uk_wheat_pipeline/
├── config.py                 # every path, URL, seed, hyperparameter
├── main.py                   # orchestrator (importlib, not subprocess)
├── pyproject.toml            # project metadata + pytest config
├── requirements.txt          # pinned runtime deps
├── requirements-dev.txt      # test + notebook tooling
├── PLAYBOOK.md               # live maintenance checklist
├── WORKFLOW.md               # this document
├── README.md                 # user-facing overview
├── src/                      # shared, importable code (single source of truth)
│   ├── __init__.py
│   ├── _bootstrap.py         # common imports, path setup, display fallback
│   ├── cv.py                 # ExpandingWindowCV + baselines
│   ├── models.py             # 8 model factories + tuning + PI/oracle hooks
│   ├── metrics.py            # RMSE, MAE, Diebold–Mariano (HLN)
│   ├── weather.py            # aggregate_seasonal (phenological windows)
│   ├── features.py           # forecast_exogenous (ARIMA 1,0,0 exog projection)
│   ├── guards.py             # 6 runtime invariant guards
│   └── plotting.py           # chart style, headless Agg fallback
├── stages/                   # 5 runnable entry points (each has main())
│   ├── 01_Data_Acquisition.py
│   ├── 02_Modelling_Table.py
│   ├── 03_EDA.py
│   ├── 04_Feature_Engineering.py
│   └── 05_Model.py
├── notebooks/                # jupytext-paired .ipynb (executed, figures inline)
│   ├── 01_Data_Acquisition.ipynb
│   ├── 02_Modelling_Table.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   └── 05_Model.ipynb
├── tests/                    # 43 fast unit tests (pytest)
│   ├── test_cv.py
│   ├── test_features.py
│   ├── test_guards.py
│   ├── test_metrics.py
│   ├── test_models.py
│   └── test_weather.py
└── data/
    ├── raw/                  # downloaded by 01 (Met Office + manifest)
    ├── processed/            # modelling table (single downstream input)
    ├── expected/             # canonical outputs (verification gate)
    ├── thesis_reference/     # archived original thesis numbers
    └── outputs/              # result CSVs + decision guide (charts inline)
```

---

## 1. High-Level Flow (Mermaid)

```mermaid
flowchart TD
    %% ====== DATA LAYER ======
    subgraph RAW[Raw Data Acquisition]
        MET[Met Office UK Series\nTmean + Rainfall] --> DL[download_met_series\nstages/01_Data_Acquisition.py]
        DL -->|frozen raw .txt| RAW_DISK[(data/raw/)]
        DL -->|parsed CSV| PARSED[(data/raw/met_office_*_monthly.csv)]
        RAW_DISK -->|SHA-256 manifest| MANIFEST[data/raw/manifest.csv]
    end

    subgraph WEATHER[Seasonal Aggregation]
        PARSED --> AGG[aggregate_seasonal\nsrc/weather.py]
        AGG -->|autumn/winter/spring/grainfill| UKMEAN[(data/processed/uk_wheat_weather_seasonal_uk_mean.csv)]
        AGG -.->|spot-check| ALIGN[assert_alignment\nsrc/guards.py]
    end

    subgraph MODEL_TABLE[Canonical Modelling Table]
        UKMEAN --> MT[stages/02_Modelling_Table.py]
        YIELD[DEFRA yield 1980-2024] --> MT
        POLICY[Policy dummies 1992/2005/2022] --> MT
        MT -->|single CSV, no missing| MODEL_CSV[(data/processed/uk_wheat_modelling_table_1980_2024.csv)]
    end

    %% ====== MODEL LAYER ======
    subgraph FEATURES[Feature Engineering]
        MODEL_CSV --> FE[stages/04_Feature_Engineering.py]
        FE -->|splits, scales, lags| FEATURES_OUT[(data/processed/feature_matrices/)]
    end

    subgraph CV[Expanding-Window CV]
        MODEL_CVF --> CV_ENGINE[ExpandingWindowCV\nsrc/cv.py]
        CV_ENGINE -->|train_end=2000..2023| FOLD[Fold: train ≤ origin, test = origin+h]
        FOLD -->|per-origin cache| MODEL_CACHE{model_cache}
        FOLD -->|leakage guards| LEAK[assert_training_precedes_origin\nassert_horizon_consistent]
        FOLD -->|fit| FACTORY[Model Factory\nsrc/models.py]
        FOLD -->|predict| PRED[y_pred]
        FOLD -.->|swallowed folds| WARN[warn_on_swallowed_fits]
    end

    subgraph MODELS[Model Factories]
        FACTORY --> PERSIST[Persistence]
        FACTORY --> ARIMA[ARIMA AICc p,q∈0..3]
        FACTORY --> SARIMA[SARIMA seasonal]
        FACTORY --> ARIMAX[ARIMAX stepwise\nselect_exog_pvalues]
        FACTORY --> PROPHET[Prophet changepoint CV]
        FACTORY --> RF[RandomForest tuned full series]
        FACTORY --> XGB[XGBoost tuned full series]
        FACTORY --> HYBRID[ARIMA+XGBoost hybrid]
    end

    %% ====== OUTPUT / VERIFY ======
    subgraph OUTPUTS[Outputs + Verification]
        PRED --> DETAILS[(data/outputs/model_details_results_corrected.csv)]
        PRED --> SUMMARY[(data/outputs/model_comparison_results_corrected.csv)]
        SUMMARY --> DM[Diebold-Mariano HLN\nsrc/metrics.py]
        DM --> DM_CSV[(data/outputs/dm_test_results.csv)]
        SUMMARY --> PI[Prediction Intervals\nARIMA/Prophet predict_interval\nwarn_on_swallowed_fits per model]
        PI --> PI_BALANCE[assert_balanced_test_sets on pi_details]
        PI --> PI_CSV[(data/outputs/pi_*_corrected.csv)]
        SUMMARY --> ORACLE[Oracle exogenous\noracle_fc]
        ORACLE --> ORACLE_CSV[(data/outputs/oracle_exogenous_results.csv)]
        SUMMARY --> VERIFY[verify() - section 5.10]
        VERIFY --> BAL[assert_balanced_test_sets]
        VERIFY --> EXPECTED[(data/expected/)]
        VERIFY --> DECISION[data/outputs/decision_guide.md]
    end

    %% ====== THESIS REFERENCE ======
    EXPECTED --> THESIS[data/thesis_reference/\noriginal thesis numbers]
    THESIS --> CHANGES[changes_vs_thesis.csv]

    %% Styling
    classDef raw fill:#e8f5e9,stroke:#2e7d32
    classDef proc fill:#e3f2fd,stroke:#1565c0
    classDef model fill:#fff3e0,stroke:#ef6c00
    classDef verify fill:#fce4ec,stroke:#c2185b
    class RAW,WEATHER,MODEL_TABLE,FEATURES raw
    class CV,MODELS proc
    class OUTPUTS verify
```

---

## 1b. End-to-End Pipeline Flow (Detailed)

```mermaid
flowchart TD
    %% ============ INPUTS ============
    subgraph INPUTS[📥 EXTERNAL INPUTS]
        MET_OFFICE[Met Office HadUK-Grid\nTmean + Rainfall UK monthly\n1884–present]
        DEFRA_YIELD[DEFRA Cereal Production\nUK wheat yield t/ha\n1980–2024]
        POLICY_DUMMIES[Policy structural breaks\n1992 MacSharry; 2005 SPS; 2022 Ukraine war]
        CONFIG[config.py\nSEED=42, HORIZONS=[1,2,3,4]\nINITIAL_TRAIN_END=2000]
    end

    %% ============ STAGE 01 ============
    subgraph S01[🟢 Stage 01: Data Acquisition]
        DL[download_met_series\nstages/01_Data_Acquisition.py]
        PARSE[Parse fixed-width .txt →\ntidy year/month/value CSV]
        MANIFEST_GEN[SHA-256 manifest.csv\nprovenance: url, timestamp, hash]
        AGG[aggregate_seasonal\nsrc/weather.py]
        ALIGN_CHECK[assert_alignment\nsrc/guards.py:88]
        UKMEAN_OUT[(uk_wheat_weather_seasonal_uk_mean.csv)]
    end

    %% ============ STAGE 02 ============
    subgraph S02[🟡 Stage 02: Modelling Table]
        MERGE[Merge yield + UK-mean weather +\npolicy dummies → single CSV]
        VALIDATE[Assert: 45 rows, no NaN,\nno duplicate years]
        MODEL_TABLE_OUT[(uk_wheat_modelling_table_1980_2024.csv)]
    end

    %% ============ STAGE 03 ============
    subgraph S03[🔵 Stage 03: EDA]
        EDA[03_EDA.py\nFigures only, no outputs]
    end

    %% ============ STAGE 04 ============
    subgraph S04[🟣 Stage 04: Feature Engineering]
        FEATURES[04_Feature_Engineering.py\nLags, rolling stats, scaling artifacts]
        FEAT_OUT[(feature_matrices/)]
    end

    %% ============ STAGE 05 ============
    subgraph S05[🔴 Stage 05: Model Comparison & Verification]
        %% CV Engine
        CV_ENGINE[ExpandingWindowCV\nsrc/cv.py]
        ORIGIN_LOOP[for origin in 2000..2023]
        HORIZON_LOOP[for h in 1..4]
        TRAIN_MASK[train_df = year ≤ origin]
        TEST_YEAR[test_year = origin + h]
        
        %% Guards at fold level
        GUARD_LEAK[assert_training_precedes_origin]
        GUARD_HORIZON[assert_horizon_consistent]
        
        %% Model factories
        FACTORY{Model Factory\nsrc/models.py}
        STAT_MODELS[Statistical:\nPersistence, ARIMA, SARIMA,\nARIMAX, Prophet]
        ML_MODELS[ML (tuned once full series):\nRF, XGBoost, ARIMA+XGBoost]
        
        %% ARIMAX specific
        ARIMAX_STEP[arimax_stepwise_selection]
        SELECT_PVAL[select_exog_pvalues\nby-name, raises if missing]
        
        %% Per-origin cache
        MODEL_CACHE[(model_cache[origin] →\nshared across h=1..4)]
        
        %% Predictions
        PREDICT[predictor.predict(h)]
        DETAILS_ACC[Accumulate per-fold details]
        
        %% Swallowed fold tracking
        SKIPPED[skipped_folds list\n(horizon, test_year, error)]
        WARN_SWALLOW[warn_on_swallowed_fits]
        
        %% Aggregation
        SUMMARY[Per-model × horizon RMSE/MAE/n_test]
        
        %% DM Tests
        DM_TESTS[Diebold-Mariano HLN\nsrc/metrics.py]
        DM_CSV[(dm_test_results.csv)]
        
        %% Prediction Intervals
        PI_ARIMA[ARIMA get_forecast().conf_int()]
        PI_PROPHET[Prophet interval_width=0.95]
        PI_CSV[(pi_*_corrected.csv)]
        
        %% Oracle
        ORACLE_RUN[Re-run CV with oracle_fc=\nperfect weather foresight]
        ORACLE_CSV[(oracle_exogenous_results.csv)]
        
        %% VERIFICATION GATE
        VERIFY[verify() §5.10]
        BAL_CHECK[assert_balanced_test_sets]
        DIFF_CHECK[Diff vs data/expected/\natol 1e-4..1e-6]
        DECISION_GEN[decision_guide.md auto-generated]
        
        %% Outputs
        OUT_COMP[(model_comparison_results_corrected.csv)]
        OUT_DET[(model_details_results_corrected.csv)]
        OUT_BASE[(baseline_results.csv)]
    end

    %% ============ VERIFICATION REFERENCE ============
    subgraph VERIFY_REF[✅ Verification Reference]
        EXPECTED[(data/expected/\n6 canonical CSVs + manifest.md)]
        THESIS_REF[(data/thesis_reference/\noriginal thesis numbers)]
        CHANGES[(changes_vs_thesis.csv\n39 row diff ledger)]
    end

    %% ============ FLOW CONNECTIONS ============
    MET_OFFICE --> DL
    DEFRA_YIELD --> MERGE
    POLICY_DUMMIES --> MERGE
    CONFIG --> CV_ENGINE
    CONFIG --> FACTORY
    
    DL --> PARSE
    PARSE --> MANIFEST_GEN
    PARSE --> AGG
    AGG --> ALIGN_CHECK
    ALIGN_CHECK --> UKMEAN_OUT
    UKMEAN_OUT --> MERGE
    
    MERGE --> VALIDATE
    VALIDATE --> MODEL_TABLE_OUT
    
    MODEL_TABLE_OUT --> EDA
    MODEL_TABLE_OUT --> FEATURES
    FEATURES --> FEAT_OUT
    
    MODEL_TABLE_OUT --> CV_ENGINE
    CV_ENGINE --> ORIGIN_LOOP
    ORIGIN_LOOP --> HORIZON_LOOP
    HORIZON_LOOP --> TRAIN_MASK
    TRAIN_MASK --> TEST_YEAR
    TEST_YEAR --> GUARD_LEAK
    GUARD_LEAK --> GUARD_HORIZON
    GUARD_HORIZON --> FACTORY
    
    FACTORY --> STAT_MODELS
    FACTORY --> ML_MODELS
    STAT_MODELS --> ARIMAX_STEP
    ARIMAX_STEP --> SELECT_PVAL
    SELECT_PVAL --> MODEL_CACHE
    ML_MODELS --> MODEL_CACHE
    MODEL_CACHE --> PREDICT
    PREDICT --> DETAILS_ACC
    DETAILS_ACC -.-> SKIPPED
    SKIPPED --> WARN_SWALLOW
    
    DETAILS_ACC --> SUMMARY
    SUMMARY --> DM_TESTS
    DM_TESTS --> DM_CSV
    SUMMARY --> PI_ARIMA
    SUMMARY --> PI_PROPHET
    PI_ARIMA --> PI_CSV
    PI_PROPHET --> PI_CSV
    SUMMARY --> ORACLE_RUN
    ORACLE_RUN --> ORACLE_CSV
    
    SUMMARY --> OUT_COMP
    SUMMARY --> OUT_BASE
    DETAILS_ACC --> OUT_DET
    
    OUT_COMP --> VERIFY
    OUT_DET --> VERIFY
    DM_CSV --> VERIFY
    OUT_BASE --> VERIFY
    ORACLE_CSV --> VERIFY
    PI_CSV --> VERIFY
    
    VERIFY --> BAL_CHECK
    VERIFY --> DIFF_CHECK
    DIFF_CHECK --> EXPECTED
    VERIFY --> DECISION_GEN
    
    EXPECTED --> THESIS_REF
    THESIS_REF --> CHANGES
    
    %% ============ STYLING ============
    classDef input fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef stage01 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef stage02 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef stage03 fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef stage04 fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef stage05 fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    classDef guard fill:#ffebee,stroke:#c62828,stroke-dasharray: 5 5
    classDef output fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef verify fill:#fce4ec,stroke:#c2185b,stroke-width:3px
    
    class MET_OFFICE,DEFRA_YIELD,POLICY_DUMMIES,CONFIG input
    class DL,PARSE,MANIFEST_GEN,AGG,ALIGN_CHECK,UKMEAN_OUT stage01
    class MERGE,VALIDATE,MODEL_TABLE_OUT stage02
    class EDA stage03
    class FEATURES,FEAT_OUT stage04
    class CV_ENGINE,ORIGIN_LOOP,HORIZON_LOOP,TRAIN_MASK,TEST_YEAR,FACTORY,STAT_MODELS,ML_MODELS,ARIMAX_STEP,SELECT_PVAL,MODEL_CACHE,PREDICT,DETAILS_ACC,SKIPPED,WARN_SWALLOW,SUMMARY,DM_TESTS,DM_CSV,PI_ARIMA,PI_PROPHET,PI_CSV,ORACLE_RUN,ORACLE_CSV,OUT_COMP,OUT_DET,OUT_BASE stage05
    class GUARD_LEAK,GUARD_HORIZON,SELECT_PVAL,BAL_CHECK guard
    class VERIFY,DIFF_CHECK,DECISION_GEN verify
    class EXPECTED,THESIS_REF,CHANGES verify
```

---

## 1c. Data Lineage Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL SOURCES                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │
│  │ Met Office       │  │ DEFRA Yield      │  │ Policy Events    │              │
│  │ HadUK-Grid 1km   │  │ 1980-2024 t/ha   │  │ 1992/2005/2022   │              │
│  │ Tmean + Rainfall │  │ (canonical)      │  │ (constructed)    │              │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘              │
└───────────┼─────────────────────┼─────────────────────┼────────────────────────┘
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           STAGE 01: DATA ACQUISITION                             │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ download_met_series()                                                    │   │
│  │   • HTTP GET (frozen after first run)                                    │   │
│  │   • SHA-256 recorded in manifest.csv                                     │   │
│  │   • Parse fixed-width .txt → year/month/value CSV                        │   │
│  └────────────────────────────┬─────────────────────────────────────────────┘   │
│                               ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ aggregate_seasonal()  ──►  assert_alignment() (guard)                    │   │
│  │   • autumn  = Oct-Nov Y-1                                                │   │
│  │   • winter  = Dec Y-1 – Feb Y  (year boundary)                          │   │
│  │   • spring  = Mar-May Y                                                  │   │
│  │   • grainfill = Jun-Aug Y                                                │   │
│  └────────────────────────────┬─────────────────────────────────────────────┘   │
│                               ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ OUTPUT: uk_wheat_weather_seasonal_uk_mean.csv                            │   │
│  │         data/raw/manifest.csv (provenance)                               │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────┬────────────────────────────────────┘
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           STAGE 02: MODELLING TABLE                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ MERGE (inner join on year)                                               │   │
│  │   • yield_t_ha (from DEFRA)                                              │   │
│  │   • 8 weather covariates (from UK-mean seasonal)                         │   │
│  │   • 3 policy dummies (cap_1992, cap_2005, ukraine_2022)                 │   │
│  │ VALIDATE: 45 rows, 1980-2024, no NaN, no dup years                      │   │
│  └────────────────────────────┬─────────────────────────────────────────────┘   │
│                               ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ OUTPUT: uk_wheat_modelling_table_1980_2024.csv  ◄── SINGLE SOURCE OF TRUTH │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────┬────────────────────────────────────┘
                                             ▼
              ┌──────────────────────────────┼──────────────────────────────┐
              ▼                              ▼                              ▼
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│     STAGE 03: EDA       │  │  STAGE 04: FEATURES     │  │    STAGE 05: MODELS     │
│  (read-only, figures)   │  │  (demo transforms)      │  │  (full CV + verify)     │
└─────────────────────────┘  └─────────────────────────┘  └───────────┬─────────────┘
                                                                      ▼
                          ┌─────────────────────────────────────────────────────┐
                          │           EXPANDING-WINDOW CV PROTOCOL               │
                          │  for origin in 2000..2023:                          │
                          │    train = rows[year ≤ origin]                      │
                          │    for h in [1,2,3,4]:                              │
                          │      test_year = origin + h                         │
                          │      GUARD: assert_training_precedes_origin()       │
                          │      GUARD: assert_horizon_consistent()             │
                          │      predictor = factory(train, h)  # cached        │
                          │      y_pred = predictor.predict(h)                  │
                          │      GUARD: warn_on_swallowed_fits()                │
                          │      accumulate details                             │
                          └────────────────────────────┬────────────────────────┘
                                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           STAGE 05: POST-CV PIPELINE                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Aggregation  │  │ DM Tests     │  │ PIs          │  │ Oracle       │        │
│  │ RMSE/MAE/    │  │ HLN-corrected│  │ ARIMA/Prophet│  │ oracle_fc    │        │
│  │ n_test per   │  │ pairwise     │  │ 95% coverage │  │ perfect      │        │
│  │ model×horizon│  │              │  │              │  │ weather      │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
│         │                 │                 │                 │                │
│         └─────────────────┼─────────────────┼─────────────────┘                │
│                           ▼                                                 │
│              ┌────────────────────────┐                                    │
│              │ VERIFICATION GATE       │                                    │
│              │ • assert_balanced_      │                                    │
│              │   test_sets()           │                                    │
│              │ • verify() diffs vs     │                                    │
│              │   data/expected/        │                                    │
│              │ • ALL CHECKS PASSED     │                                    │
│              └───────────┬─────────────┘                                    │
│                          ▼                                                 │
│         ┌───────────────┼───────────────┐                                 │
│         ▼               ▼               ▼                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                             │
│  │ Output    │  │ Output    │  │ decision_ │                             │
│  │ CSVs      │  │ PIs/      │  │ guide.md  │                             │
│  │ (6 files) │  │ Oracle    │  │ (auto)    │                             │
│  └───────────┘  └───────────┘  └───────────┘                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         VERIFICATION REFERENCE                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │ data/expected/  │  │ data/thesis_    │  │ changes_vs_     │                 │
│  │ 6 canonical     │  │ reference/      │  │ thesis.csv      │                 │
│  │ CSVs + manifest │  │ original thesis │  │ 39-row diff     │                 │
│  │ (gate target)   │  │ numbers (viva)  │  │ ledger          │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1d. CV Fold Structure (Visual)

```
ORIGIN = 2000                          ORIGIN = 2005                          ORIGIN = 2023
┌─────────────────────────────┐       ┌─────────────────────────────┐       ┌─────────────────────────────┐
│ TRAIN (1980–2000)           │       │ TRAIN (1980–2005)           │       │ TRAIN (1980–2023)           │
│ ┌──┬──┬──┬──┬──┬──┬──┬──┐   │       │ ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐  │       │ ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐  │
│ │80│81│82│83│84│85│86│87│...│       │ │80│81│82│83│84│85│86│87│...│...│  │       │ │80│81│82│...│21│22│23│  │  │
│ └──┴──┴──┴──┴──┴──┴──┴──┘   │       │ └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘  │       │ └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘  │
│           ▲                 │       │            ▲                 │       │            ▲                 │
│           │                 │       │            │                 │       │            │                 │
│      max_year=2000          │       │      max_year=2005           │       │      max_year=2023           │
└─────────────────────────────┘       └─────────────────────────────┘       └─────────────────────────────┘
           │                                    │                                    │
           ▼                                    ▼                                    ▼
TESTS for h=1,2,3,4:                    TESTS for h=1,2,3,4:                    TESTS for h=1,2,3,4:
  h=1 → year 2001                        h=1 → year 2006                        h=1 → year 2024
  h=2 → year 2002                        h=2 → year 2007                        h=2 → (no data)
  h=3 → year 2003                        h=3 → year 2008                        h=3 → (no data)
  h=4 → year 2004                        h=4 → year 2009                        h=4 → (no data)

GUARDS at EACH FOLD:
  ✓ assert_training_precedes_origin(train_df, origin)
  ✓ assert_horizon_consistent(test_year, origin, horizon)
  ✓ warn_on_swallowed_fits(attempted, succeeded, model_name)
```

---

## 2. Stage-by-Stage Narrative

### Stage 01 — Data Acquisition (`stages/01_Data_Acquisition.py`)

| **What** | **Why** | **Where** | **When** |
|----------|---------|-----------|----------|
| Download Met Office Tmean + Rainfall UK series | Only public source for UK-mean monthly weather | `data/raw/met_office_*.txt` | First run only; thereafter **frozen** (reuse local file) |
| Parse to tidy `year, month, value` CSV | Reproducible intermediate; SHA-256 in manifest | `data/raw/met_office_*_monthly.csv` | Every run (idempotent) |
| Aggregate to phenological windows via `aggregate_seasonal` | Single source of truth for alignment (FR-3) | `src/weather.py` | Every run |
| **Spot-check alignment** with `assert_alignment` | Catch off-by-one (autumn = Oct-Nov Y-1) | `src/guards.py:88` | Every run, fails fast if drift |
| Write UK-mean seasonal CSV + manifest | Provenance for audit trail | `data/processed/...uk_mean.csv`, `data/raw/manifest.csv` | Every run |

**Key guard**: `assert_alignment` (FR-3/F-1) — verifies a reference harvest year's autumn/spring/grainfill values equal the mean of the intended months. Refuses to proceed if mismatch > 1e-9.

---

### Stage 02 — Modelling Table (`stages/02_Modelling_Table.py`)

| **What** | **Why** | **Where** | **When** |
|----------|---------|-----------|----------|
| Merge canonical yield + UK-mean weather + policy dummies | Sole downstream input (FR-2) | `data/processed/uk_wheat_modelling_table_1980_2024.csv` | Every run |
| Validate: no missing, no duplicate years, 45 rows | Fail fast on data integrity | Inline asserts | Every run |

**Key principle**: This CSV is the **single contract** between data prep and modelling. Every stage 03/04/05 reads *only* this file.

---

### Stage 03 — EDA (`stages/03_EDA.py`)

| **What** | **Why** | **Where** | **When** |
|----------|---------|-----------|----------|
| Load modelling table, plot yield trend, weather correlations, missingness | Exploratory only; no outputs consumed downstream | Inline figures (notebook) | Optional, human inspection |

**No guards** — purely visual.

---

### Stage 04 — Feature Engineering (`stages/04_Feature_Engineering.py`)

| **What** | **Why** | **Where** | **When** |
|----------|---------|-----------|----------|
| Build lagged features, rolling stats, standardisation artifacts | Reusable feature matrices for any model | `data/processed/feature_matrices/` | Every run |

**No guards** — deterministic transforms; tested in `tests/test_features.py`.

---

### Stage 05 — Model Comparison & Verification (`stages/05_Model.py`)

This is the **thesis engine**. It runs the entire comparison and self-verifies.

#### 5.1–5.5: Model Factories + CV

| **What** | **Why** | **Where** | **When** |
|----------|---------|-----------|----------|
| `ExpandingWindowCV` over origins 2000..2023, horizons 1..4 | Identical protocol for all 8 models (FR-4) | `src/cv.py:30` | Every run |
| Per-origin model cache (fit once, predict 4 horizons) | Efficiency + reproducibility (same fit shared) | `cv.py:102` | Every run |
| **Leakage guards** at each fold: `assert_training_precedes_origin`, `assert_horizon_consistent` | Structural leakage impossible by construction (NFR-1) | `src/guards.py:18,54` | Every fold |
| **Swallowed-fold warning**: `warn_on_swallowed_fits` (CV + PI loops) | Quantify silent `except:continue` (F-4/NFR-2) | `src/guards.py:205`; called `cv.py:172`, `05_Model.py:407,411` | End of each model + each PI batch |
| `select_exog_pvalues` in ARIMAX stepwise | By-name p-values; eliminates positional-slice bug (NFR-9/F-1) | `src/guards.py:66`, `models.py:122` | Every ARIMAX fold |

#### 5.6: Results Aggregation

| **What** | **Why** | **Where** |
|----------|---------|-----------|
| Per-horizon RMSE/MAE/n_test + per-fold details | Thesis Tables 4.1, 4.2, 4.6 | `data/outputs/model_comparison_results_corrected.csv`, `model_details_results_corrected.csv` |

#### 5.7: Diebold–Mariano Tests

| **What** | **Why** | **Where** |
|----------|---------|-----------|
| Pairwise DM with HLN correction, HAC autocov, Student-t df | Statistical rigour (NFR-3) | `src/metrics.py:78`, `data/outputs/dm_test_results.csv` |

#### 5.8: Prediction Intervals

| **What** | **Why** | **Where** |
|----------|---------|-----------|
| ARIMA `get_forecast().conf_int()` + Prophet `interval_width=0.95` | Coverage vs nominal (FR-6) | `src/models.py:156, 202`, `data/outputs/pi_*_corrected.csv` |

#### 5.9: Oracle Experiment

| **What** | **Why** | **Where** |
|----------|---------|-----------|
| Re-run CV with `oracle_fc` = perfect weather foresight | Ceiling bound (FR-7) | `src/models.py` factories + `oracle_fc`, `data/outputs/oracle_exogenous_results.csv` |

#### 5.10: Verification Gate (The Gate)

| **What** | **Why** | **Where** |
|----------|---------|-----------|
| `assert_balanced_test_sets(details)` — raises if any model has different test years per horizon | Protocol symmetry (F-4/NFR-2) | `src/guards.py:184`, `05_Model.py:665` |
| `assert_balanced_test_sets(pi_details)` — same check on PI coverage table | PI protocol symmetry (F-4/NFR-2) | `src/guards.py:184`, `05_Model.py:501` |
| `verify()` diffs every output CSV vs `data/expected/` under tolerance | Self-certification (FR-8) | `05_Model.py:621` |
| Prints `ALL CHECKS PASSED` or `SOME CHECKS FAILED` | Loud failure | Stdout |

#### 5.11: Decision Guide (FR-9)

| **What** | **Why** | **Where** |
|----------|---------|-----------|
| Machine-generated markdown: best overall, per-horizon best, DM significances, PI coverage, oracle ceiling | Practitioner deliverable, prose-data consistency (NFR-10) | `data/outputs/decision_guide.md` |

---

## 3. Cross-Cutting Modules (`src/`)

| Module | Responsibility | Key Guards Used |
|--------|----------------|-----------------|
| `_bootstrap.py` | Common imports, path setup, display fallback, modelling table loader | — |
| `cv.py` | Expanding-window protocol, baselines, per-origin cache | `assert_training_precedes_origin`, `assert_horizon_consistent`, `warn_on_swallowed_fits` (fold + PI loops) |
| `models.py` | 8 model factories, `fit_best_arima`, `arimax_stepwise_selection`, `tune_rf/tune_xgb`, `predict_interval` | `select_exog_pvalues` |
| `metrics.py` | RMSE, MAE, Diebold–Mariano (HLN) | — |
| `weather.py` | `aggregate_seasonal` (single alignment truth) | — |
| `guards.py` | **All 6 runtime invariants** | — |
| `features.py` | `forecast_exogenous` (ARIMA(1,0,0) exog projection) | — |
| `plotting.py` | Chart style, headless `Agg` fallback | — |

---

## 4. Data Contracts (What Flows Where)

```
┌─────────────────────────────────────────────────────────────────┐
│ MODELLING TABLE (single source of truth for modelling)          │
│ data/processed/uk_wheat_modelling_table_1980_2024.csv           │
│ Columns: year, yield_t_ha, autumn_temp, autumn_rain,            │
│          winter_temp, winter_rain, spring_temp, spring_rain,    │
│          grainfill_temp, grainfill_rain, cap_1992, cap_2005,    │
│          ukraine_2022                                           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ CV FOLD (per origin, per horizon)                               │
│ train_df = rows where year ≤ origin                              │
│ test_row = row where year = origin + horizon                    │
│ Guards: training max ≤ origin, test_year = origin + h           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ MODEL FACTORY(train_df, horizon) → predictor.predict(h)         │
│ Returns scalar forecast; ARIMA/Prophet also .predict_interval() │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ DETAILS DF (one row per fold)                                   │
│ model, horizon, test_year, y_true, y_pred, train_time, mem      │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ SUMMARY DF (per model × horizon)                                │
│ model, horizon, rmse, mae, n_test, avg_train_time, avg_mem      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Why This Architecture? (Justification)

| Decision | Rationale |
|----------|-----------|
| **Scripts + notebooks as twins** (`jupytext --update`) | VSCode-native editing; notebooks render figures for thesis; `.py` is source of truth |
| **All shared logic in `src/`** | Single source of truth (NFR-7); no copy-paste across 5 stages |
| **Expanding-window CV with per-origin cache** | Matches thesis protocol exactly; 4× speedup vs refitting per horizon |
| **Frozen raw data + manifest** | Network independence after first run (NFR-11); provenance audit trail |
| **Runtime guards (`guards.py`)** | Convert silent failures (leakage, misalignment, swallowed folds) into loud `AssertionError`/`Warning` at point of violation |
| **By-name p-values in ARIMAX** | Fixes the thesis's positional-slice bug (`pvalues[-n_exog:]` read `ar.L1`/`sigma2`) |
| **Self-verification against `data/expected/`** | Reproducibility gate (FR-8); any refactor that changes numbers fails CI |
| **Thesis numbers archived in `data/thesis_reference/`** | Viva trail; `changes_vs_thesis.csv` documents every difference (F-3) |
| **ML tuning on full series (disclosed)** | Reproduces corrected Colab exactly; asymmetric vs statistical models but documented (NFR-1/2) |
| **Decision guide auto-generated** | Prose and data can never drift (NFR-10) |
| **`main.py` uses `importlib` not `subprocess`** | Stages are composable Python modules; faster; no shell escaping issues |
| **DRY via `src/_bootstrap.py`** | Eliminates duplicated sys.path, display, pandas config, modelling table loader, common imports |

---

## 6. Clarity Improvements (Optional, Low-Risk)

| Area | Current State | Suggested Improvement | Effort |
|------|---------------|----------------------|--------|
| **Stage 04 Feature Engineering** | Minimal; passes through modelling table mostly | Document which features each model actually consumes; remove unused | Low |
| **Config-driven model list** | Hard-coded in `05_Model.py` two loops (stat + ML) | Move model registry to `config.py` → loop once | Medium |
| **Parallel CV (caution)** | Sequential; oracle/PI paths reseed RNG per origin | Add `joblib` with fixed seeds; re-verify `ALL CHECKS PASSED` | Medium (risk: RNG changes) |
| **Type hints** | Partial (`cv.py` dataclass, some funcs) | Full `pyright`/`mypy` clean; add `py.typed` | Low |
| **Structured logging** | `logging` in 01, `print` elsewhere | Centralised JSON logs per stage; easier CI parsing | Low |
| **CLI for single model re-run** | Must run full `05_Model.py` | `python -m src.cv --model ARIMAX --horizon 1` | Medium |
| **Data versioning (DVC)** | Git ignores `data/` | Track `data/expected/` + `data/raw/` hashes in DVC | Medium |

**Recommendation**: The current code is **clear and verified**. Only adopt improvements if they unblock a real need (CI, team onboarding, runtime). The guards + verification gate already enforce the invariants that matter.

---

## 7. Quick Reference: Entry Points

```bash
# Full pipeline (30–70 min)
python main.py

# Single stage + prerequisites
python main.py --stage 03

# Individual script (no orchestration)
python stages/05_Model.py

# Unit tests (all 43 pass)
python -m pytest tests/ -q

# Notebook sync after .py edits
jupytext --sync stages/05_Model.py
```

---

## 8. Files That Change When You Regenerate Results

| File | When It Changes | Guarded By |
|------|-----------------|------------|
| `data/expected/*.csv` | Intentional regeneration only | `verify()` + `manifest.md` hashes |
| `data/thesis_reference/*.csv` | Never (archival) | — |
| `data/outputs/*.csv` | Every full run | Diff vs `expected/` in `verify()` |
| `data/outputs/decision_guide.md` | Every full run | Auto-generated from verified CSVs |
| `data/raw/manifest.csv` | First download only | Frozen raw reuse |

---

*Generated 2026-08-07. This document reflects the state after the runtime invariant guards (`src/guards.py`) were integrated, DRY refactoring (`src/_bootstrap.py`, `config.WEATHER_RENAME`), folder rename (`scripts/` → `stages/`), and the full pipeline verified (`ALL CHECKS PASSED` for stages 01–04; stage 05 imports OK, full run 70+ min).*