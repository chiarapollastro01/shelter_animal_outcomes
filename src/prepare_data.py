"""
Data Preparation Module for Shelter Animal Outcomes.

Separates raw training data into feature matrix (X) and target series (y),
removing data leakage columns (OutcomeType, OutcomeSubtype) before pipeline entry.

CLI Usage
---------
Using default options:
    python -m src.prepare_data data/raw_data/train.csv

Or specifying custom output directory and split parameters:
    python -m src.prepare_data data/raw_data/train.csv \
        --output-dir data/split_data \
        --test-size 0.2 \
        --random-state 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL = "OutcomeType"
LEAKAGE_COLS = ("OutcomeType", "OutcomeSubtype")


def prepare_and_split_data(
    raw_csv_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Read raw CSV, extract target, drop target leakage columns, and perform stratified train/test split.

    Parameters
    ----------
    raw_csv_path : Path
        Path to the raw train.csv file.
        test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Controls the shuffling applied to the data before applying the split.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Tuple of (X_features, X_test, y_target, y_test).
    """
    logger.info("Reading raw dataset from %s", raw_csv_path)
    df = pd.read_csv(raw_csv_path)

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' missing from input file.")

    y = df[TARGET_COL].rename("target")

    cols_to_drop = [col for col in LEAKAGE_COLS if col in df.columns]
    X = df.drop(columns=cols_to_drop)

    logger.info("Performing stratified train/test split (test_size=%.2f)...", test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Split completed. Train shape: %s, Test shape: %s",
        X_train.shape,
        X_test.shape,
    )
    return X_train, X_test, y_train, y_test


def main(
    raw_csv_path: Path, output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Orchestrate the data preparation pipeline and write split datasets to disk.

    Parameters
    ----------
    raw_csv_path : Path
        Path to the input raw CSV file (e.g., data/raw_data/train.csv).
    output_dir : Path
        Destination directory for the split CSV files.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = prepare_and_split_data(
        raw_csv_path=raw_csv_path,
        test_size=test_size,
        random_state=random_state,
    )

    X_train.to_csv(output_dir / "train_features.csv", index=False)
    X_test.to_csv(output_dir / "test_features.csv", index=False)
    y_train.to_frame().to_csv(output_dir / "train_target.csv", index=False)
    y_test.to_frame().to_csv(output_dir / "test_target.csv", index=False)

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
        - output_features (Path): Destination path for extracted features.
        - output_target (Path): Destination path for extracted target.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_csv_path", type=Path, help="Path to raw train.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/split_data"),
        help="Directory where split files will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of test set (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
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
        test_size=args.test_size,
        random_state=args.random_state,
         )