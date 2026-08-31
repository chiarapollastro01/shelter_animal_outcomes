"""Data preparation module for the Shelter Animal Outcomes dataset.

This module is the first step of the pipeline: it reads the raw Kaggle file,
separates the features from the target, and splits both into a training and a
test set that every later step reads back from disk.

Exported Functions
------------------
prepare_and_split_data(raw_csv_path, test_size, random_state) -> tuple[...]
    Function that reads the raw CSV, withholds the outcome columns from the
    features, and performs the stratified split.

main(raw_csv_path, output_dir, config_path, random_state) -> None
    Function that runs the preparation and writes the four resulting files,
    reading the split proportion from the YAML parameter file.

parse_args(args_list) -> argparse.Namespace
    Function that parses the command-line arguments, kept apart from main so
    that it can be tested without touching sys.argv.

Note on the stratification
--------------------------
The train/test split is stratified on both outcome and species simultaneously.
Each species gets its own model trained, so any drift in the test set's species
distribution would leave one model with insufficient evaluation data.
The price is more strata, hence a higher chance that one of them is too small to be split at all.

CLI Usage
---------
Using default options (run as a module from the project root):
    python -m src.prepare_data data/raw_data/train.csv

Or specifying custom output directory and run parameters:
    python -m src.prepare_data data/raw_data/train.csv \
        --output-dir data/split_data \
        --config config.yaml \
        --random-state 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

logger = logging.getLogger(__name__)


def prepare_and_split_data(
    raw_csv_path: Path,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Read raw CSV, extract target, drop target leakage columns,
       and perform stratified train/test split.

    Parameters
    ----------
    raw_csv_path : Path
        Path to the raw train.csv file.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Controls the shuffling applied to the data before applying the split.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        Tuple of (X_train, X_test, y_train, y_test).

    Raises
    ------
    KeyError
        If config.TARGET_COL is missing from the input CSV DataFrame.

    ValueError
        If the target column holds missing values, or if a stratum is too
        small to be split, raised by the stratified split itself.
    """
    logger.info("Reading raw dataset from %s", raw_csv_path)
    df = pd.read_csv(raw_csv_path)

    if config.TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{config.TARGET_COL}' missing from input file.")

    y = df[config.TARGET_COL]

    if y.isna().any():
        raise ValueError(
            f"Target column '{config.TARGET_COL}' has {int(y.isna().sum())} "
            "missing values: a row with no outcome cannot be used."
        )

    cols_to_drop = [col for col in config.NON_FEATURE_COLS if col in df.columns]
    X = df.drop(columns=cols_to_drop)

    if config.SPECIES_COL in X.columns:
        strata = y.astype(str) + "_" + X[config.SPECIES_COL].astype(str)
    else:
        strata = y

    logger.info("Performing stratified train/test split (test_size=%.2f)...", test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strata,
    )

    logger.info(
        "Split completed. Train shape: %s, Test shape: %s",
        X_train.shape,
        X_test.shape,
    )
    return X_train, X_test, y_train, y_test


def main(
    raw_csv_path: Path,
    output_dir: Path,
    config_path: Path,
    random_state: int,
) -> None:
    """Orchestrate the data preparation pipeline and write split datasets to disk.

    Parameters
    ----------
    raw_csv_path : Path
        Path to the input raw CSV file.
    output_dir : Path
        Destination directory for the split CSV files.
    config_path : Path
        Path to the YAML configuration file.
    random_state : int
        Random seed for reproducibility.
    """
    run_params = config.load_params(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = prepare_and_split_data(
        raw_csv_path=raw_csv_path,
        test_size=run_params["test_size"],
        random_state=random_state,
    )

    X_train.to_csv(output_dir / config.TRAIN_FEATURES_FILE, index=False)
    X_test.to_csv(output_dir / config.TEST_FEATURES_FILE, index=False)
    y_train.to_frame().to_csv(output_dir / config.TRAIN_TARGET_FILE, index=False)
    y_test.to_frame().to_csv(output_dir / config.TEST_TARGET_FILE, index=False)

    logger.info("Successfully saved train and test datasets to %s", output_dir)


def parse_args(args_list: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the data preparation module.

    Isolates argument parsing logic from the main execution flow to enable
    direct and clean unit testing without mutating sys.argv.

    Parameters
    ----------
    args_list : list[str] | None, default=None
        List of command-line argument strings to parse. If None, arguments
        are read automatically from sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing:
        - raw_csv_path (Path): Path to the input raw CSV file.
        - output_dir (Path): Destination directory for the split CSV files.
        - config_path (Path): Path to the YAML configuration file.
        - random_state (int): Random seed for reproducible splitting.

    Examples
    --------
    >>> args = parse_args(["data/raw_data/train.csv", "--random-state", "7"])
    >>> args.raw_csv_path.name
    'train.csv'
    >>> args.random_state
    7
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_csv_path",
        type=Path,
        nargs="?",
        default=config.RAW_DATA_PATH,
        help="Path to raw train.csv",
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.SPLIT_DATA_DIR,
        help="Directory where split files will be saved.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        dest="config_path",
        default=config.CONFIG_FILE_PATH,
        help="YAML file holding the run parameters",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=config.RANDOM_STATE,
        help="Random state for reproducibility",
    )

    return parser.parse_args(args_list)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    main(raw_csv_path=args.raw_csv_path,
        output_dir=args.output_dir,
        config_path=args.config_path,
        random_state=args.random_state,
         )
