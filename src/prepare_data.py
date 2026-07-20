"""
Data Preparation Module for Shelter Animal Outcomes.

Separates raw training data into feature matrix (X) and target series (y),
removing data leakage columns (OutcomeType, OutcomeSubtype) before pipeline entry.

CLI Usage
---------
python src/prepare_data.py data/raw/train.csv \
    --output-features data/split_data/train_features.csv \
    --output-target data/split_data/train_target.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL = "OutcomeType"
LEAKAGE_COLS = ("OutcomeType", "OutcomeSubtype")


def prepare_and_split_data(
    raw_csv_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Read raw CSV, extract target, and drop leakage columns.

    Parameters
    ----------
    raw_csv_path : Path
        Path to the raw train.csv file.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Tuple of (X_features, y_target).
    """
    logger.info("Reading raw dataset from %s", raw_csv_path)
    df = pd.read_csv(raw_csv_path)

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' missing from input file.")

    y = df[TARGET_COL].rename("target")

    cols_to_drop = [col for col in LEAKAGE_COLS if col in df.columns]
    X = df.drop(columns=cols_to_drop)

    logger.info(
        "Successfully prepared %d rows. Features shape: %s", len(X), X.shape
    )
    return X, y


def main(
    raw_csv_path: Path, output_features_path: Path, output_target_path: Path
) -> None:
    """Orchestrate the data preparation pipeline and write split datasets to disk.

    Parameters
    ----------
    raw_csv_path : Path
        Path to the input raw CSV file (e.g., data/raw_data/train.csv).
    output_features_path : Path
        Destination path for the feature matrix CSV.
    output_target_path : Path
        Destination path for the target series CSV.
    """
    output_features_path.parent.mkdir(parents=True, exist_ok=True)
    output_target_path.parent.mkdir(parents=True, exist_ok=True)

    X, y = prepare_and_split_data(raw_csv_path)

    X.to_csv(output_features_path, index=False)
    y.to_frame().to_csv(output_target_path, index=False)

    logger.info("Saved features to %s", output_features_path)
    logger.info("Saved target to %s", output_target_path)


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
        "--output-features",
        type=Path,
        default=Path("data/split_data/train_features.csv"),
    )
    parser.add_argument(
        "--output-target",
        type=Path,
        default=Path("data/split_data/train_target.csv"),
    )
    
    return parser.parse_args(args_list)


if __name__ == "__main__":  
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()  
    main(args.raw_csv_path, args.output_features, args.output_target)