"""Evaluation module for the Shelter Animal Outcomes classifiers.

Reads the models the training step wrote, scores each of them on the held-out
test split of its own species, and writes a single JSON report.

Exported Functions
------------------
evaluate_model(model, X_test, y_test) -> dict[str, float]
    Function that computes the metrics of one fitted model on one test split.

per_class_report(model, X_test, y_test) -> dict[str, dict[str, float]]
    Function that breaks the same predictions down class by class.
    
load_test_data(test_features_path, test_target_path) -> tuple[...]
    Function that reads the two test files prepare_data wrote and fails fast
    if they have drifted out of alignment.

main(test_features_path, test_target_path, models_dir, output_metrics_path) -> None
    Function that evaluates one model per species and writes the report.

parse_args(args_list) -> argparse.Namespace
    Function that parses the command-line arguments, kept apart from main so
    that it can be tested without touching sys.argv.

Note on the metrics
-------------------
The set computed here is deliberately wider than the scoring list in
config.yaml. Those metrics drive model selection and have to be optimisable by
GridSearchCV; these describe the model that was already chosen, and are free to
include log_loss, which scores the predicted probabilities rather than the
predicted labels and so says whether the model believed its own answer.

The per-class breakdown is here because the target is heavily imbalanced:
on a split where one outcome covers half the rows, an aggregate score can look
respectable while a rare class is never predicted at all.

CLI Usage
---------
Using default options:
    python -m src.evaluate

Or specifying custom paths:
    python -m src.evaluate \
        --test-features data/split_data/test_features.csv \
        --test-target data/split_data/test_target.csv \
        --models-dir models/ \
        --output-metrics metrics.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TypedDict, Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)

from src import config
from src.preprocessing import drop_rows_missing_required

logger = logging.getLogger(__name__)

class ProbabilisticClassifier(Protocol):
    """Structural interface for estimators providing label 
       and probability predictions. Using a Protocol enables 
       structural subtyping without inheritance, cleanly accepting
       both imblearn pipelines and standard scikit-learn models while 
       excluding incompatible transformers.

    Attributes
    ----------
    classes_ : np.ndarray
        The labels seen during fit, in the column order of predict_proba.
    """

    classes_: np.ndarray

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return one predicted label per row of X."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return one probability per class per row of X."""

class SpeciesReport(TypedDict):
    """The evaluation block written for one species.

    The two halves have different shapes on purpose: 'overall' summarises the
    model in a flat mapping of floats, 'per_class' holds one such mapping per
    outcome. Declaring them here is what lets main stay typed.

    Attributes
    ----------
    overall : dict[str, float]
        What evaluate_model returns.
    per_class : dict[str, dict[str, float]]
        What per_class_report returns.
    """

    overall: dict[str, float]
    per_class: dict[str, dict[str, float]]

def evaluate_model(
    model: ProbabilisticClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Compute key evaluation metrics on the test dataset.

    Parameters
    ----------
    model : ProbabilisticClassifier
        Trained best model or pipeline implementing predict and predict_proba.
    X_test : pd.DataFrame
        Test features matrix.
    y_test : pd.Series
        True target labels.

    Returns
    -------
    dict[str, float]
        Dictionary of calculated evaluation metrics:
        - 'log_loss': Multi-class logarithmic loss.
        - 'accuracy': Overall classification accuracy.
        - 'balanced_accuracy': Mean of the per-class recalls. Unlike 
          'accuracy' it cannot be inflated by getting the majority class right.
        - 'f1_macro': Unweighted mean of the per-class F1 scores, so every
          class counts the same regardless of how rare it is. This is the
          metric the training tournament selects on.
        - 'f1_weighted': Average F1-score across all classes, weighted by support.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Deliberately wider than the scoring list in config.yaml: those metrics
    # drive selection, these describe the chosen model.
    metrics = {
        # labels=model.classes_ keeps proba columns and label set aligned
        # even when a rare class is absent from this test split.
        "log_loss": float(log_loss(y_test, y_proba, labels=model.classes_)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
    }

    logger.info(
        "Evaluation Results -> Log Loss: %.4f | Accuracy: %.4f | "
        "F1 (Macro): %.4f | F1 (Weighted): %.4f",
        metrics["log_loss"],
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["f1_macro"],
        metrics["f1_weighted"],
    )
    return metrics

def per_class_report(
    model: ProbabilisticClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1 and support for every class separately.

     On an imbalanced problem the aggregate scores hide the classes that matter most, 
    and a rare outcome can be missed entirely without any of them moving much.

    Parameters
    ----------
    model : ProbabilisticClassifier
        Trained model or pipeline implementing predict.
    X_test : pd.DataFrame
        Test features matrix.
    y_test : pd.Series
        True target labels.

    Returns
    -------
    dict[str, dict[str, float]]
        One entry per class the model was trained on, each mapping
        'precision', 'recall', 'f1' and 'support' to their values. The
        aggregate rows scikit-learn adds are dropped, evaluate_model already
        covering them.
    """
    y_pred = model.predict(X_test)

    # labels=model.classes_ keeps a class the model knows in the report even
    # when the test split happens to hold none of it, which for the rarest
    # outcomes is a real possibility. zero_division=0 is what such a class
    # scores, rather than a warning and a nan.
    report = classification_report(
        y_test,
        y_pred,
        labels=model.classes_,
        output_dict=True,
        zero_division=0,
    )

    # classification_report mixes per-class dicts with aggregate rows
    # ('accuracy', 'macro avg', 'weighted avg'); only the former are wanted.
    return {
        str(label): {
            "precision": float(report[str(label)]["precision"]),
            "recall": float(report[str(label)]["recall"]),
            "f1": float(report[str(label)]["f1-score"]),
            "support": float(report[str(label)]["support"]),
        }
        for label in model.classes_
    }

def load_test_data(
    test_features_path: Path, test_target_path: Path
) -> tuple[pd.DataFrame, pd.Series]:
    """Load test features and target datasets, failing fast on row misalignment.

    Parameters
    ----------
    test_features_path : Path
        Path to the test features CSV file.
    test_target_path : Path
        Path to the test target CSV file.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        A tuple containing:
        - X_test (pd.DataFrame): DataFrame with test features.
        - y_test (pd.Series): Series with test target values.

    Raises
    ------
    ValueError
        If the number of rows in the test features and target datasets do not match.
    """
    X_test = pd.read_csv(test_features_path)
    y_test = pd.read_csv(test_target_path)[config.TARGET_COL]

    if len(X_test) != len(y_test):
        raise ValueError(
            f"Test features ({len(X_test)} rows) and target ({len(y_test)} rows) "
            "are misaligned."
        )

    logger.info("Loaded %d test rows from %s", len(X_test), test_features_path)
    return X_test, y_test


def main(
    test_features_path: Path,
    test_target_path: Path,
    models_dir: Path,
    output_metrics_path: Path
) -> None:
    """Orchestrate model evaluation per species and write performance metrics to JSON.

    Parameters
    ----------
    test_features_path : Path
        Path to the test features CSV file.
    test_target_path : Path
        Path to the test target CSV file.
    models_dir : Path
        Directory containing the trained species-specific model files.
    output_metrics_path : Path
        Destination path for saving the generated JSON metrics report.

    Raises
    ------
    FileNotFoundError
        If a species has rows in the test set but no model on disk: the
        artefacts read here are the ones train writes.
    ValueError
        If no species had any row, nothing having been evaluated.
    """
    X_test, y_test = load_test_data(test_features_path, test_target_path)
    X_test, y_test = drop_rows_missing_required(X_test, y_test)

    all_metrics: dict[str, SpeciesReport] = {}
    for species in config.SPECIES:
        mask = X_test[config.SPECIES_COL] == species
        if not mask.any():
            logger.warning("No %s rows in the test set: skipping.", species)
            continue

        # Mirror the training preparation: species filter + drop AnimalType
        X_species = X_test.loc[mask].drop(columns=[config.SPECIES_COL]).reset_index(drop=True)
        y_species = y_test.loc[mask].reset_index(drop=True)

        species_model_path = models_dir / config.MODEL_FILE_TEMPLATE.format(
            species=species.lower()
        )
        logger.info("Evaluating %s model (%d samples) from %s",
            species, len(X_species), species_model_path)
        model = joblib.load(species_model_path)

        all_metrics[species.lower()] = {
            "overall": evaluate_model(model, X_species, y_species),
            "per_class": per_class_report(model, X_species, y_species),
        }

    if not all_metrics:
        raise ValueError(
            "No species had any row in the test set: nothing was evaluated. "
            f"Check that {test_features_path} carries the {config.SPECIES_COL} column."
        )

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.write_text(json.dumps(all_metrics, indent=4))
    logger.info("Saved metrics report to %s", output_metrics_path)


def parse_args(args_list: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the model evaluation module.

    Isolates argument parsing logic from execution to enable clean unit testing
    without modifying sys.argv.

    Parameters
    ----------
    args_list : list[str] | None, default=None
        List of command-line argument strings to parse. If None, arguments
        are read directly from sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing:
        - test_features (Path): Path to test features CSV.
        - test_target (Path): Path to test target CSV.
        - models_dir (Path): Directory containing trained model files per species.
        - output_metrics (Path): Path for output metrics JSON file.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-features",
        type=Path,
        default=config.SPLIT_DATA_DIR / config.TEST_FEATURES_FILE,
        help="Path to test features CSV",
    )
    parser.add_argument(
        "--test-target",
        type=Path,
        default=config.SPLIT_DATA_DIR / config.TEST_TARGET_FILE,
        help="Path to test target CSV",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=config.MODELS_DIR,
        help="Directory containing trained model files per species",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=config.REPORTS_DIR / config.METRICS_FILE,
        help="Path for saving output metrics JSON",
    )
    return parser.parse_args(args_list)

# Testing note: the entry-point guard is excluded from coverage
# (see pyproject.toml), executing it would load real models inside the test suite
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    main(
        test_features_path=args.test_features,
        test_target_path=args.test_target,
        models_dir=args.models_dir,
        output_metrics_path=args.output_metrics,
    )
