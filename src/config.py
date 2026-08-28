"""Configuration module for the dataset schema, the paths, and the YAML loader.

Provides:
- Raw dataset column definitions.
- Feature grouping rules .
- The two constants that are fixed once for the whole project rather than
  chosen per run: the random state and the set of species.
- File and directory paths resolved dynamically from PROJECT_ROOT.
- "load_params", which reads the run parameters from "config.yaml".

Notes
-----
It was decided to create two configuration files because this module describes the data, therefore
changing any of it is a code change. "config.yaml", instead, contains what describes a run:
the split proportions, the number of cross-validation folds, the scoring metrics etc. Changing
any of these is an analysis decision. "RANDOM_STATE" deliberately sist on this since
it's just a guarantee of replicability, so it's fixed.
Importing this module has no side effect: the YAML file is read only when
"load_params" is called.
-------------------
This module is the only place the dataset schema is written down. Every
column name below is read from here, never spelled out in the modules, and
every transformer takes its column names as an __init__ parameter defaulting
to one of these constants. Renaming a column in the source data is therefore
a one-line change here, and adding a species is a one-line change to SPECIES:
the Snakefile derives its wildcards from it, train validates --species
against it, and evaluate loops over it, so a third model would be trained and
evaluated without touching anything else.

The boundary: the names of the columns the transformers write are fixed
(config.LOG_AGE_COL, config.IS_MIX_COL and the cyclical ones), because
NUM_SCALE_COLS and CAT_ENCODE_COLS have to name them for the ColumnTransformer
downstream. Adding a derived feature means declaring it in one of those two
tuples, not just producing it.
"""

from pathlib import Path
from typing import Any

import yaml


# --- DATASET SCHEMA ---

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

# Dropped by prepare_data before anything else: the outcome and its subtype are
# the answer, so leaving either in the feature matrix would result in data leakage.
NON_FEATURE_COLS: tuple[str, ...] = (TARGET_COL, SUBTARGET_COL)

# Default columns of CategoricalFeaturesEngineer, charachterized by high-cardinality:
# rare entries are collapsed into "Other" before one-hot encoding
CATEGORICAL_COLS: tuple[str, ...] = (BREED_COL, COLOR_COL)

# Default fill targets of DataCleaner: columns whose missing values are filled
# with the literal "Unknown"
FILL_TARGETS: tuple[str, ...] = (NAME_COL, BREED_COL, COLOR_COL)

# Default columns to remove of DataCleaner: identifiers that carry no signal
COLUMNS_TO_REMOVE: tuple[str, ...] = (ID_COL,)

# Passed to drop_rows_missing_critical, outside the pipeline: rows missing any of
# these cannot be used at all, since neither can be successfully imputed
ROW_REQUIRED_COLS: tuple[str, ...] = (DATETIME_COL, SPECIES_COL)


# Derived columns
HOUR_SIN_COL: str = "Hour_sin"
HOUR_COS_COL: str = "Hour_cos"
WDAY_SIN_COL: str = "Wday_sin"
WDAY_COS_COL: str = "Wday_cos"
DOY_SIN_COL: str = "DoY_sin"
DOY_COS_COL: str = "DoY_cos"
IS_WEEKEND_COL: str = "IsWeekend"
LOG_AGE_COL: str = "log_age_in_days"
HAS_NAME_COL: str = "has_name"
IS_MIX_COL: str = "is_mix"
REPRODUCTIVE_STATUS_COL: str = "Reproductive_Status"

# Numerical columns that are min-max scaled
NUM_SCALE_COLS: tuple[str, ...] = (
    HOUR_SIN_COL,
    HOUR_COS_COL,
    WDAY_SIN_COL,
    WDAY_COS_COL,
    DOY_SIN_COL,
    DOY_COS_COL,
    LOG_AGE_COL,
)

# Binary columns that are one-hot encoded
CAT_ENCODE_COLS: tuple[str, ...] = (BREED_COL, COLOR_COL, REPRODUCTIVE_STATUS_COL)

# --- GLOBAL EXECUTION & MODEL SETTINGS ---

# Guarantee of replicability for all stochastic components
RANDOM_STATE: int = 42
# The values AnimalType takes. One independent tournament runs per species,
# since the drivers of an adoption differ between dogs and cats.
SPECIES: tuple[str, ...] = ("Dog", "Cat")

# --- PATHS ---

# The directory above the package, so every path below is absolute and holds
# regardless of the working directory a script is launched from.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw_data" / "train.csv"

SPLIT_DATA_DIR: Path = PROJECT_ROOT / "data" / "split_data"

MODELS_DIR: Path = PROJECT_ROOT / "models"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"

FIGURES_DIR: Path = REPORTS_DIR / "figures"

CONFIG_FILE_PATH: Path = PROJECT_ROOT / "config.yaml"

# --- FILE NAMING CONVENTIONS ---

# prepare_data splits features and target for both train and test data
TRAIN_FEATURES_FILE: str = "train_features.csv"
TRAIN_TARGET_FILE: str = "train_target.csv"
TEST_FEATURES_FILE: str = "test_features.csv"
TEST_TARGET_FILE: str = "test_target.csv"

METRICS_FILE: str = "metrics.json"

# Written by train, loaded back by evaluate. {species} is filled in lowercase.
MODEL_FILE_TEMPLATE: str = "best_shelter_model_{species}.pkl"


# --- RUN PARAMETERS (config.yaml) ---

def load_params(path: Path = CONFIG_FILE_PATH) -> dict[str, Any]:
    """Load the run parameters from the YAML configuration file.

    The file is read on every call rather than at import time, so that
    importing this module has no side effect and the tests can point at a
    temporary file.

    Parameters
    ----------
    path : Path, default=CONFIG_FILE_PATH
        Path of the YAML file to read.

    Returns
    -------
    dict[str, Any]
        Mapping from parameter name to value.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    ValueError
        If the YAML file is empty or just commented.
    """
    with open(path, encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    if params is None:
        raise ValueError(
            f"{path} holds no YAML document: an empty parameter file would "
            "only fail later, on the first key the pipeline looks up."
        )

    return params
