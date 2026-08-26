"""Training module for the Shelter Animal Outcomes classifiers.

Runs the model tournament for one species: a grid search over the registered
classifier families with stratified cross-validation, an unbiased evaluation
of the winner on a hold-out split that never entered model selection, and
the persistence of the fitted pipeline next to a metadata sidecar.

Exported Functions
------------------
merge_search_grids(search_spaces) -> dict[str, dict[str, list]]
    Function that merges the shared grid into the grid of each classifier
    family, and refuses a family the pipeline cannot build.

load_training_data(features_path, target_path) -> tuple[pd.DataFrame, pd.Series]
    Function that reads the two files prepare_data wrote and fails fast if
    they have drifted out of alignment.

run_tournament(X, y, cv, search_grids, run_params) -> tuple[str, GridSearchCV]
    Function that grid-searches every family and returns the winning one.

train_one_species(X, y, models_dir, species, search_grids, run_params) -> None
    Function that runs the tournament for one species and writes the winner
    to disk.

main(features_path, target_path, models_dir, config_path, species) -> None
    Function that reads the split files and runs the tournament for one
    species, writing the winner to disk.

parse_args(args_list) -> argparse.Namespace
    Function that parses the command-line arguments, kept apart from main so
    that it can be tested without touching sys.argv.

Note on the two scores
----------------------
The cross-validated score selects the winner, and a model selected as the best
of many on that score alone is optimistically biased. The hold-out split exists
to give an honest number: it takes no part in the search, and it is scored on
the same metric so that the two are directly comparable.

Note on the seed
----------------
Every other run parameter is read from config.yaml, the seed is not. It is
fixed once for the whole project rather than chosen per run, which is what
makes two runs of the same configuration comparable at all.

CLI Usage
---------
# One species per invocation, using the default paths:
python -m src.train --species Dog
python -m src.train --species Cat

# Or specifying custom paths:
python -m src.train data/split_data/train_features.csv data/split_data/train_target.csv --models-dir models --species Dog

# Or running a different search space:
python -m src.train --config experiments/wide_grid.yaml --species Dog
"""
from __future__ import annotations

import argparse
import json
import joblib
import pandas as pd
import logging

from typing import Any
from collections.abc import Mapping
from sklearn.base import clone
from sklearn.metrics import classification_report, get_scorer
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from src import config
from src.preprocessing import drop_rows_missing_required
from src.pipeline import available_models, get_model_pipeline

logger = logging.getLogger(__name__)

# Key of the search_spaces mapping holding the parameters shared by every
# classifier family, as opposed to the per-family entries.
COMMON_GRID_KEY = "common"

def merge_search_grids(
    search_spaces: Mapping[str, Mapping[str, list[Any]]],
) -> dict[str, dict[str, list[Any]]]:
    """Merge the shared grid into the grid of each classifier family.

    The entry named by ``COMMON_GRID_KEY`` holds the parameters that apply to
    every family (categorical-collapse ratio, SMOTE neighbours); every other
    entry must name a registered classifier family. A family that repeats a
    shared key overrides it.

    Parameters
    ----------
    search_spaces : Mapping[str, Mapping[str, list]]
        Search grids as read from the configuration file.

    Returns
    -------
    dict[str, dict[str, list]]
        One complete grid per classifier family, ready for GridSearchCV.

    Raises
    ------
    KeyError
        If the shared entry is missing.
    ValueError
        If a key does not name a registered classifier family.

    Examples
    --------
    >>> spaces = {"common": {"smote__k_neighbors": [3]}, "knn": {"clf__n_neighbors": [5]}}
    >>> merge_search_grids(spaces)
    {'knn': {'smote__k_neighbors': [3], 'clf__n_neighbors': [5]}}
    """
    common = dict(search_spaces[COMMON_GRID_KEY])
    grids = {
        name: grid for name, grid in search_spaces.items() if name != COMMON_GRID_KEY
    }

    unknown = set(grids) - set(available_models())
    if unknown:
        raise ValueError(
            f"Unknown classifier families in the search spaces: {sorted(unknown)}. "
            f"Available: {available_models()}"
        )

    return {name: {**common, **dict(grid)} for name, grid in grids.items()}


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
    y = pd.read_csv(target_path)[config.TARGET_COL]

    if len(X) != len(y):
        raise ValueError(
            f"Features ({len(X)} rows) and target ({len(y)} rows) are misaligned."
        )

    logger.info("Loaded %d rows from %s", len(X), features_path)
    return X, y


def run_tournament(
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    search_grids: Mapping[str, Mapping[str, list[Any]]],
    run_params: Mapping[str, Any],
) -> tuple[str, GridSearchCV]:
    """Execute a grid-search tournament across all classifier families.

    The winner is the family with the highest mean cross-validated score on
    the metric named by ``run_params["refit"]``.

    Parameters
    ----------
    X : pd.DataFrame
        The training features.
    y : pd.Series
        The training target values.
    cv : StratifiedKFold
        Cross-validation splitting strategy.
    search_grids : Mapping[str, Mapping[str, list]]
        One complete grid per classifier family, as returned by
        `merge_search_grids`.
    run_params : Mapping[str, Any]
        Run parameters read from the YAML file: holdout_size, cv_n_splits,
        scoring and refit.

    Returns
    -------
    tuple[str, GridSearchCV]
        A tuple containing the name of the winning model family and the
        fitted GridSearchCV object.
    
    Raises
    ------
    ValueError
        If the search_grids mapping is empty, indicating that no classifier families were provided for the tournament.
    """
    if not search_grids:
        raise ValueError("No classifier family to search: the grids are empty.")
    
    best_name = ""
    best_search: GridSearchCV | None = None
    scoring = run_params["scoring"]
    refit = run_params["refit"]

    for model_name, grid in search_grids.items():
        logger.info("Running GridSearchCV for %s...", model_name)

        search = GridSearchCV(
            estimator=get_model_pipeline(model_name),
            param_grid=dict(grid),
            scoring=scoring,
            refit=refit,
            cv=cv,
            n_jobs=-1, # the outer loop: fits are independent, so they run in parallel
            verbose=1,
        )
        search.fit(X, y)

        logger.info(
            "Best CV %s for %s: %.4f (params: %s)",
            refit, model_name, search.best_score_, search.best_params_,
        )

        # Comparable across families because every GridSearchCV refits on
        # the same metric: best_score_ is the mean CV score of that metric, in this case f1_macro.
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name, best_search = model_name, search

    assert best_search is not None
    return best_name, best_search


def train_one_species(
    X: pd.DataFrame,
    y: pd.Series,
    models_dir: Path,
    species: str,
    search_grids: Mapping[str, Mapping[str, list[Any]]],
    run_params: Mapping[str, Any],
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
    search_grids : Mapping[str, Mapping[str, list]]
        One complete grid per classifier family.
    run_params : Mapping[str, Any]
        Run parameters read from the YAML file: holdout_size, cv_n_splits,
        scoring and refit. The seed is not among them: it comes from config,
        being fixed once for the whole project rather than per run.

    Returns
    -------
    None
    """
    holdout_size = run_params["holdout_size"]
    n_splits = run_params["cv_n_splits"]

    logger.info(
        "===== Tournament for %s (%d samples) =====", species.upper(), len(X)
    )

    # Hold-out split: never touched by grid search, so the final score is an
    # unbiased estimate (selecting the best of many models on CV score alone
    # is optimistically biased).
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y,
        test_size=holdout_size,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )

    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE
    )
    best_name, best_search = run_tournament(X_train, y_train, cv, search_grids, run_params)

    # Scored with the same metric used to pick the winner, so that the CV
    # score and the hold-out score are directly comparable.
    y_pred = best_search.best_estimator_.predict(X_hold)
    holdout_score = get_scorer(run_params["refit"])(
        best_search.best_estimator_, X_hold, y_hold
    )
    
    logger.info(
        "[%s] winner: %s | CV %s=%.4f | hold-out %s=%.4f",
        species.upper(), best_name,
        run_params["refit"], best_search.best_score_,
        run_params["refit"], holdout_score,
    )
    logger.info(
        "[%s] hold-out classification report:\n%s",
        species.upper(), classification_report(y_hold, y_pred),
    )

    # The model name lives in config because evaluate has to find this file
    # again. The sidecar name does not: nothing outside this module reads it
    model_path = models_dir / config.MODEL_FILE_TEMPLATE.format(species=species.lower())
    metadata_path = model_path.with_suffix(".json")

    final_model = clone(best_search.best_estimator_).fit(X, y)
    joblib.dump(final_model, model_path)

    metadata = {
        "species": species,
        "model": best_name,
        "metric": run_params["refit"],
        "cv_score": best_search.best_score_,
        "holdout_score": holdout_score,
        "best_params": best_search.best_params_,
        "n_samples": len(X),
        "n_samples_scored": len(X_train),
        "refit_on_full_species_data": True,
        # The resolved run parameters are recorded here because config.yaml can
        # change after training: without them the scores above would not say
        # which metric, which split, or which grid produced them.
        "run_params": dict(run_params),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    logger.info("Saved %s (+ metadata sidecar)", model_path)


def main(
    features_path: Path,
    target_path: Path,
    models_dir: Path,
    config_path: Path,
    species: str,
) -> None:
    """Train and persist the winning model for one species.

    Reads the run parameters and the two split files, drops the rows that miss
    a required column, keeps the rows of the requested species and hands them
    to the tournament, which writes the winner and its metadata to disk.

    One species per invocation: the two tournaments are independent, 
    so the Snakefile can declare one model file per species and rebuild 
    only the one that is missing.

    Parameters
    ----------
    features_path : Path
        Path to extracted features CSV.
    target_path : Path
        Path to extracted target CSV.
    models_dir : Path
        Directory to persist trained models.
    config_path : Path
        Path to the YAML file holding the search spaces and run parameters.
    species : str
        The species to train on, one of config.SPECIES.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If no row of that species survived the filtering.
    """
    run_params = config.load_params(config_path)
    search_grids = merge_search_grids(run_params["search_spaces"])
    logger.info("Loaded search grids for %s from %s", sorted(search_grids), config_path)

    X, y = load_training_data(features_path, target_path)
    X, y = drop_rows_missing_required(X, y)

    models_dir.mkdir(parents=True, exist_ok=True)

    mask = X[config.SPECIES_COL] == species
    if not mask.any():
        raise ValueError(f"No {species} rows in the training set.")
    X_species = X.loc[mask].drop(columns=[config.SPECIES_COL]).reset_index(drop=True)
    y_species = y.loc[mask].reset_index(drop=True)
    train_one_species(X_species, y_species, models_dir, species, search_grids, run_params)


def parse_args(args_list: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for model training.

    Parameters
    ----------
    args_list : list[str] | None, default=None
        List of command-line argument strings to parse. If None, sys.argv is used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing features_path, target_path, models_dir
        and config_path.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "features_path",
        type=Path,
        nargs="?",
        default=config.SPLIT_DATA_DIR / config.TRAIN_FEATURES_FILE,
        help="Path to extracted features CSV",
    )
    parser.add_argument(
        "target_path",
        type=Path,
        nargs="?",
        default=config.SPLIT_DATA_DIR / config.TRAIN_TARGET_FILE,
        help="Path to extracted target CSV",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=config.MODELS_DIR,
        help="Directory to persist trained models and metadata",
    )
    parser.add_argument(
        "--config",
        type=Path,
        dest="config_path",
        default=config.CONFIG_FILE_PATH,
        help="YAML file holding the hyperparameter search spaces and run parameters",
    )

    parser.add_argument(
        "--species",
        required=True,
        choices=config.SPECIES,
        help="Species to train on: one tournament per invocation",
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
    main(
        args.features_path,
        args.target_path,
        args.models_dir,
        args.config_path,
        args.species,
    )