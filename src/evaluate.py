"""Model evaluation module.

Evaluates the best model of the train module on the test dataset using key classification
metrics (Log Loss, Accuracy, F1-Score) and exports the results to JSON.

CLI Usage
---------
Using default options:
    python -m src.evaluate

Or specifying custom paths:
    python -m src.evaluate \
        --test-features data/split_data/test_features.csv \
        --test-target data/split_data/test_target.csv \
        --model-path models/best_model.joblib \
        --output-metrics metrics.json
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute key evaluation metrics on the test dataset.

    Parameters
    ----------
    model : Any
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
        - 'f1_weighted': Weighted average F1-score across all classes.
    """
    logger.info("Computing predictions on test set with the best model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "log_loss": float(log_loss(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
    }

    logger.info(
        "Evaluation Results -> Log Loss: %.4f | Accuracy: %.4f | F1 (Weighted): %.4f",
        metrics["log_loss"],
        metrics["accuracy"],
        metrics["f1_weighted"],
    )
    return metrics


def main(
    test_features_path: Path,
    test_target_path: Path,
    model_path: Path,
    output_metrics_path: Path,
) -> None:
    """Orchestrate model evaluation and write performance metrics to a JSON file.

    Parameters
    ----------
    test_features_path : Path
        Path to the test features CSV file.
    test_target_path : Path
        Path to the test target CSV file.
    model_path : Path
        Path to the serialized joblib model file.
    output_metrics_path : Path
        Destination path for saving the generated JSON metrics report.
    """
    logger.info("Loading test data and the best model from %s...", model_path)
    X_test = pd.read_csv(test_features_path)
    y_test = pd.read_csv(test_target_path)["target"]

    model = joblib.load(model_path)

    metrics = evaluate_model(model, X_test, y_test)

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Successfully saved metrics report to %s", output_metrics_path)


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
        - model_path (Path): Path to trained model file.
        - output_metrics (Path): Path for output metrics JSON file.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-features",
        type=Path,
        default=Path("data/split_data/test_features.csv"),
        help="Path to test features CSV",
    )
    parser.add_argument(
        "--test-target",
        type=Path,
        default=Path("data/split_data/test_target.csv"),
        help="Path to test target CSV",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/best_model.joblib"),
        help="Path to trained best model file",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("metrics.json"),
        help="Path for saving output metrics JSON",
    )
    return parser.parse_args(args_list)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    main(
        test_features_path=args.test_features,
        test_target_path=args.test_target,
        model_path=args.model_path,
        output_metrics_path=args.output_metrics,
    )