"""Training script for the Shelter Animal Outcomes classifiers.

Runs one model tournament per species (dog / cat): grid search over the
registered classifier families with stratified cross-validation, unbiased
final evaluation of each winner on a held-out split that never entered model
selection, then persistence of the fitted pipeline plus a metadata sidecar.

CLI Usage
---------
# Using default paths:
python -m src.train

# Or specifying custom paths:
python -m src.train data/split_data/train_features.csv data/split_data/train_target.csv --models-dir models
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.pipeline import get_model_pipeline

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SPLITS = 5
HOLDOUT_FRACTION = 0.2
CRITICAL_COLS = ("DateTime", "AnimalType")
SPECIES = ("Dog", "Cat")

SCORING = {
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
}


def build_search_spaces() -> dict[str, dict[str, list]]:
    """Build hyperparameter grids for each classifier family.

    The parameters shared by every family (categorical-collapse ratio, SMOTE
    neighbours) sono definiti una sola volta e uniti a ciascuna griglia.

    Returns
    -------
    dict[str, dict[str, list]]
        A dictionary mapping each classifier family name to its respective
        hyperparameter grid dictionary.
    """
    common = {
        "categorical_eng__max_other_ratio": [0.10, 0.15, 0.20],
        "smote__k_neighbors": [3, 5],
    }
    return {
        "knn": {
            **common,
            "clf__n_neighbors": [3, 5, 11],
            "clf__weights": ["uniform", "distance"],
        },
        "logistic_regression": {
            **common,
            "clf__C": [0.1, 1.0, 10.0],
        },
        "random_forest": {
            **common,
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 15, 30],
        },
    }


def load_training_data(
    features_path: Path, target_path: Path
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate features and target datasets from CSV files.

    Fails fast if the row counts of features and target do not match.

    Parameters
    ----------
    features_path : Path
        Path to the CSV file containing the input features.
    target_path : Path
        Path to the CSV file containing the target values.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        A tuple containing the loaded features DataFrame and target Series.

    Raises
    ------
    ValueError
        If the number of rows in features and target differs.
    """
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).squeeze("columns")

    if len(X) != len(y):
        raise ValueError(
            f"Features ({len(X)} rows) and target ({len(y)} rows) are misaligned."
        )

    logger.info("Loaded %d rows from %s", len(X), features_path)
    return X, y


def drop_rows_missing_critical(
    X: pd.DataFrame,
    y: pd.Series,
    critical_cols: tuple[str, ...] = CRITICAL_COLS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Remove rows whose critical columns contain missing values.

    A single vectorized boolean mask applied to both X and y keeps them
    aligned without requiring any temporary merge of the target into the features.

    Parameters
    ----------
    X : pd.DataFrame
        The feature dataset.
    y : pd.Series
        The target series.
    critical_cols : tuple[str, ...], default=CRITICAL_COLS
        Tuple of column names considered critical for data integrity.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]:
        The cleaned and index-reset features DataFrame and target Series.
    """
    mask = X[list(critical_cols)].notna().all(axis=1)
    n_dropped = int((~mask).sum())
    if n_dropped:
        logger.info(
            "Dropped %d rows with missing critical columns %s",
            n_dropped, critical_cols,
        )
    # reset_index prevents aligning problems with SMOTE/Scikit
    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)


def run_tournament(
    X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold
) -> tuple[str, GridSearchCV]:
    """Execute a grid-search tournament across all classifier families.

    The winner model is selected based on the highest mean cross-validated 
    F1-macro score.

    Parameters
    ----------
    X : pd.DataFrame
        The training features.
    y : pd.Series
        The training target values.
    cv : StratifiedKFold
        Cross-validation splitting strategy.

    Returns
    -------
    tuple[str, GridSearchCV]
        A tuple containing the name of the winning model family and the 
        fitted GridSearchCV object.
    """
    best_name, best_search = "", None

    for model_name, grid in build_search_spaces().items():
        logger.info("Running GridSearchCV for %s...", model_name)

        search = GridSearchCV(
            estimator=get_model_pipeline(model_name),
            param_grid=grid,
            scoring=SCORING,
            refit="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X, y)

        logger.info(
            "Best CV F1-macro for %s: %.4f (params: %s)",
            model_name, search.best_score_, search.best_params_,
        )

        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name, best_search = model_name, search

    return best_name, best_search


def train_one_species(
    X: pd.DataFrame, y: pd.Series, models_dir: Path, species: str
) -> None:
    """Run a full model tournament for a specific species and persist the winner.

    Performs a hold-out split (untouched by grid search to ensure unbiased 
    evaluation), runs the tournament, logs performance reports, and saves 
    the best estimator pipeline along with a JSON metadata sidecar.

    Parameters
    ----------
    X : pd.DataFrame
        Features dataset filtered for the specific species.
    y : pd.Series
        Target dataset filtered for the specific species.
    models_dir : Path
        Directory where trained models and metadata will be saved.
    species : str
        The animal species being processed (e.g., "dog", "cat").

    Returns
    -------
    None
    """
    logger.info(
        "===== Tournament for %s (%d samples) =====", species.upper(), len(X)
    )

    # Hold-out split: never touched by grid search, so the final score is an
    # unbiased estimate (selecting the best of many models on CV score alone
    # is optimistically biased).
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y,
        test_size=HOLDOUT_FRACTION,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    best_name, best_search = run_tournament(X_train, y_train, cv)

    y_pred = best_search.best_estimator_.predict(X_hold)
    holdout_f1 = f1_score(y_hold, y_pred, average="macro")

    logger.info(
        "[%s] winner: %s | CV F1-macro=%.4f | hold-out F1-macro=%.4f",
        species.upper(), best_name, best_search.best_score_, holdout_f1,
    )
    logger.info(
        "[%s] hold-out classification report:\n%s",
        species.upper(), classification_report(y_hold, y_pred),
    )

    stem = f"best_shelter_model_{species.lower()}"
    model_path = models_dir / f"{stem}.pkl"
    joblib.dump(best_search.best_estimator_, model_path)

    metadata = {
        "species": species,
        "model": best_name,
        "cv_f1_macro": best_search.best_score_,
        "holdout_f1_macro": holdout_f1,
        "best_params": best_search.best_params_,
        "n_samples": len(X),
    }
    (models_dir / f"{stem}.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    logger.info("Saved %s (+ metadata sidecar)", model_path)


def main(features_path: Path, target_path: Path, models_dir: Path) -> None:
    """Execute the complete training pipeline.

    Loads data, cleans critical missing rows, splits the dataset by species,
    runs an independent model tournament for each species, and stores the artifacts.

    Parameters
    ----------
    features_path : Path | None, default=None
        Path to extracted features CSV. If None, relies on CLI arguments.
    target_path : Path | None, default=None
        Path to extracted target CSV. If None, relies on CLI arguments.
    models_dir : Path | None, default=None
        Directory to persist trained models. If None, relies on CLI arguments.

    Returns
    -------
    None
    """
    X, y = load_training_data(features_path, target_path)
    X, y = drop_rows_missing_critical(X, y)

    models_dir.mkdir(parents=True, exist_ok=True)

    for species in SPECIES:
        mask = X["AnimalType"] == species
        X_species = X.loc[mask].drop(columns=["AnimalType"]).reset_index(drop=True)
        y_species = y.loc[mask].reset_index(drop=True)
        train_one_species(
            X_species,
            y_species,
            models_dir,
            species,
        )

    logger.info("All tournaments finished successfully. Training phase complete.")


def parse_args(args_list: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for model training.

    Parameters
    ----------
    args_list : list[str] | None, default=None
        List of command-line argument strings to parse. If None, sys.argv is used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing features_path, target_path, and models_dir.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "features_path",
        type=Path,
        nargs="?",
        default=Path("data/split_data/train_features.csv"),
        help="Path to extracted features CSV",
    )
    parser.add_argument(
        "target_path",
        type=Path,
        nargs="?",
        default=Path("data/split_data/train_target.csv"),
        help="Path to extracted target CSV",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to persist trained models and metadata",
    )
    return parser.parse_args(args_list)


# Testing note:
# This following block is excluded from coverage. Running the actual entry point 
# would trigger a full GridSearchCV inside the test suite, which would be extremely slow.
if __name__ == "__main__": 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    main(args.features_path, args.target_path, args.models_dir)