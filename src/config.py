"""Configuration module for dataset schema, paths, and model hyperparameters.

Provides:
- Raw dataset column definitions (target, subtarget, and features).
- Feature grouping rules (leakage columns, columns to drop, scaling and encoding lists).
- Pipeline parameters (random state, test split ratio, CV splits, scoring metrics).
- File and directory paths resolved dynamically from PROJECT_ROOT.
"""

from pathlib import Path

# --- DATASET SCHEMA (SINGLE SOURCE OF TRUTH) ---

SEX_COL: str = "SexuponOutcome"
AGE_COL: str = "AgeuponOutcome"
NAME_COL: str = "Name"
SPECIES_COL: str = "AnimalType"
DATETIME_COL: str = "DateTime"
BREED_COL: str = "Breed"
COLOR_COL: str = "Color"
TARGET_COL: str = "OutcomeType"
SUBTARGET_COL: str = "OutcomeSubtype"
ID_COL: str = "AnimalID"

# --- FEATURE GROUPS & SCHEMA UTILITIES ---
LEAKAGE_COLS: tuple[str, ...] = (TARGET_COL, SUBTARGET_COL)
CATEGORICAL_COLS: tuple[str, ...] = (BREED_COL, COLOR_COL)
FILL_TARGETS: tuple[str, ...] = (NAME_COL, BREED_COL, COLOR_COL)
COLUMNS_TO_REMOVE: tuple[str, ...] = (ID_COL,)
CRITICAL_COLS: tuple[str, ...] = (DATETIME_COL, SPECIES_COL)
ESSENTIAL_COLS: tuple[str, ...] = (DATETIME_COL, SEX_COL, AGE_COL, BREED_COL, COLOR_COL, NAME_COL,)


NUM_SCALE_COLS: tuple[str, ...] = (
    "Hour_sin",
    "Hour_cos",
    "Wday_sin",
    "Wday_cos",
    "DoY_sin",
    "DoY_cos",
    "log_age_in_days",
)
CAT_ENCODE_COLS: tuple[str, ...] = (BREED_COL, COLOR_COL, "Reproductive_Status")

# --- GLOBAL EXECUTION & MODEL SETTINGS ---
RANDOM_STATE: int = 42
DEFAULT_TEST_SIZE: float = 0.2
MAX_ITER=1000
N_SPLITS: int = 5
HOLDOUT_FRACTION: float = 0.2
SPECIES: tuple[str, ...] = ("Dog", "Cat")
MAX_OTHER_RATIO: float = 0.15

SCORING: dict[str, str] = {
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
}

# --- PATHS ---

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw_data" / "train.csv"

SPLIT_DATA_DIR: Path = PROJECT_ROOT / "data" / "split_data"

MODELS_DIR: Path = PROJECT_ROOT / "models"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"



