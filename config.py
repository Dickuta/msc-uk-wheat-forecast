"""
config.py

Central configuration for the UK wheat yield forecasting pipeline.

Everything that could change between runs (paths, data-source URLs, random
seed, forecast horizons, cross-validation settings) lives here so the five
stage notebooks only need to `import config`.

Directory layout
----------------
uk_wheat_pipeline/
    config.py               central configuration (this file)
    src/                    shared code (metrics, features, models, cv, weather)
    scripts/                01_Data_Acquisition .. 05_Model  (.py, VSCode cells)
    notebooks/              01_Data_Acquisition .. 05_Model  (.ipynb, executed)
    data/
        raw/                files downloaded from public data sources
        processed/          modelling table used by every downstream stage
        expected/           canonical thesis outputs (for verification)
        outputs/            result CSVs produced by the pipeline (charts inline)
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PIPELINE_ROOT = Path(__file__).resolve().parent

DATA_DIR = PIPELINE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXPECTED_DIR = DATA_DIR / "expected"
OUTPUT_DIR = DATA_DIR / "outputs"

# The modelling table is the single dataset used by stages 2, 4 and 5.
MODEL_TABLE_FILE = PROCESSED_DIR / "uk_wheat_modelling_table_1980_2024.csv"
MODEL_TABLE_YEARS = (1980, 2024)

# Canonical intermediate files (used to assemble / validate the table).
YIELD_FILE = PROCESSED_DIR / "uk_wheat_yield_1980_2024.csv"
POLICY_DUMMIES_FILE = PROCESSED_DIR / "uk_wheat_policy_dummies_1980_2024.csv"
WEATHER_SEASONAL_UK_MEAN_FILE = PROCESSED_DIR / "uk_wheat_weather_seasonal_uk_mean.csv"
WEATHER_SEASONAL_CANONICAL_FILE = (
    PROCESSED_DIR / "uk_wheat_weather_seasonal_canonical.csv"
)

# --------------------------------------------------------------------------- #
# Data sources (Open Government Licence unless noted)
# --------------------------------------------------------------------------- #
# Met Office UK Climate Series: UK-mean monthly temperature and rainfall,
# derived from HadUK-Grid 1 km data. See data_sources.md in the thesis vault.
MET_OFFICE_BASE = "https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets"
MET_OFFICE_SOURCES = {
    "tas": {
        "url": f"{MET_OFFICE_BASE}/Tmean/date/UK.txt",
        "description": "UK-mean monthly temperature (degC)",
        "raw_file": "met_office_Tmean_UK.txt",
        "agg": "mean",
    },
    "rainfall": {
        "url": f"{MET_OFFICE_BASE}/Rainfall/date/UK.txt",
        "description": "UK-mean monthly rainfall (mm)",
        "raw_file": "met_office_Rainfall_UK.txt",
        "agg": "sum",
    },
}

# Phenological windows used to aggregate monthly weather to a harvest year.
# Each entry is (start_month, end_month, year_offset); the offset is added to
# the harvest year Y to locate the window's months. For harvest year Y:
#   autumn     = Oct-Nov of Y-1        -> (10, 11, -1)
#   winter     = Dec(Y-1)-Feb(Y)       -> (12, 2, -1)  spans a year boundary
#   spring     = Mar-May of Y          -> (3, 5, 0)
#   grain fill = Jun-Aug of Y          -> (6, 8, 0)
SEASON_WINDOWS = {
    "autumn": (10, 11, -1),
    "winter": (12, 2, -1),
    "spring": (3, 5, 0),
    "grainfill": (6, 8, 0),
}

# --------------------------------------------------------------------------- #
# Modelling protocol (identical to the corrected Colab pipeline)
# --------------------------------------------------------------------------- #
SEED = 42  # every random operation uses this seed
HORIZONS = [1, 2, 3, 4]  # forecast horizons in years
INITIAL_TRAIN_END = 2000  # expanding-window CV starts forecasting after 2000

COVARIATE_COLS = [
    "autumn_temp",
    "autumn_rain",
    "winter_temp",
    "winter_rain",
    "spring_temp",
    "spring_rain",
    "grainfill_temp",
    "grainfill_rain",
    "cap_1992",
    "cap_2005",
    "ukraine_2022",
]

# --------------------------------------------------------------------------- #
# Model hyperparameters (identical to the corrected Colab pipeline)
# --------------------------------------------------------------------------- #
ARIMA_P_Q_RANGE = range(0, 4)  # ARIMA order search: p, q in 0..3, d=0
SARIMA_P_Q_RANGE = range(0, 3)  # SARIMA seasonal (p,0,q,1)
ARIMAX_ALPHA = 0.10  # stepwise p-value threshold
ARIMAX_MAX_COVARIATES = 5  # cap on selected covariates
PROPHET_CHANGEPOINT_CANDIDATES = [0.01, 0.05, 0.1]
PROPHET_INTERVAL_WIDTH = 0.95
XGB_EARLY_STOPPING = 50
TSCV_N_SPLITS = 5  # TimeSeriesSplit for ML tuning

# ML tuning grids (tuned once on the full series, seed 42)
RF_GRID = {
    "n_estimators": 500,
    "depth": [3, 5, None],
    "leaf": [1, 4],
    "split": [2, 10],
    "features": ["sqrt", 0.7],
}
XGB_GRID = {
    "n_estimators": 500,
    "depth": [3, 5],
    "lr": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample": [0.8, 1.0],
}

# --------------------------------------------------------------------------- #
# Output files written by the pipeline
# --------------------------------------------------------------------------- #
OUTPUT_FILES = {
    "comparison": OUTPUT_DIR / "model_comparison_results_corrected.csv",
    "details": OUTPUT_DIR / "model_details_results_corrected.csv",
    "dm_tests": OUTPUT_DIR / "dm_test_results.csv",
    "baselines": OUTPUT_DIR / "baseline_results.csv",
    "pi_coverage": OUTPUT_DIR / "pi_coverage_results_corrected.csv",
    "pi_details": OUTPUT_DIR / "pi_detailed_results_corrected.csv",
    "oracle": OUTPUT_DIR / "oracle_exogenous_results.csv",
    "decision_guide": OUTPUT_DIR / "decision_guide.md",
}


def ensure_dirs() -> None:
    """Create every data directory used by the pipeline."""
    for d in (RAW_DIR, PROCESSED_DIR, EXPECTED_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
