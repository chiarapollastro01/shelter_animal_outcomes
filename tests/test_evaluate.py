"""Unit test suite for the model evaluation module (src.evaluate).

Verifies evaluation metric calculations, JSON report generation, and CLI argument
parsing for src.evaluate.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.evaluate import evaluate_model, main, parse_args

# =====================================================================
#                               FIXTURE
# =====================================================================


@pytest.fixture
def dummy_evaluation_setup(tmp_path: Path):
    """Provides temporary test features, test target, and a trained dummy model."""
    
    X_test = pd.DataFrame({"feature1": [1, 2, 3, 4], "feature2": [10, 20, 30, 40]})
    y_test = pd.Series(["Adoption", "Transfer", "Adoption", "Transfer"], name="target")

    # Save test files
    features_path = tmp_path / "test_features.csv"
    target_path = tmp_path / "test_target.csv"
    X_test.to_csv(features_path, index=False)
    y_test.to_frame().to_csv(target_path, index=False)

    model = DummyClassifier(strategy="prior")
    model.fit(X_test, y_test)
    model_path = tmp_path / "best_model.joblib"
    joblib.dump(model, model_path)

    return features_path, target_path, model_path


# =====================================================================
#                               TESTS
# =====================================================================


def test_evaluate_model_returns_correct_metrics(dummy_evaluation_setup):
    """GIVEN: a trained model and test data
       WHEN: evaluate_model is called
       THEN: it calculates log_loss, accuracy, and f1_weighted.
    """
    features_path, target_path, model_path = dummy_evaluation_setup
    X_test = pd.read_csv(features_path)
    y_test = pd.read_csv(target_path)["target"]
    model = joblib.load(model_path)

    metrics = evaluate_model(model, X_test, y_test)

    assert "log_loss" in metrics
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert isinstance(metrics["log_loss"], float)


def test_main_creates_metrics_json(dummy_evaluation_setup, tmp_path: Path):
    """GIVEN: valid inputs
       WHEN: main is executed
       THEN: it creates a JSON file containing evaluation metrics.
    """
    features_path, target_path, model_path = dummy_evaluation_setup
    output_metrics = tmp_path / "metrics.json"

    main(
        test_features_path=features_path,
        test_target_path=target_path,
        model_path=model_path,
        output_metrics_path=output_metrics,
    )

    assert output_metrics.exists()

    with open(output_metrics, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "log_loss" in data
    assert "accuracy" in data


def test_parse_args_defaults():
    """
    GIVEN: an empty argument list (default invocation)
    WHEN: parse_args is executed
    THEN: it assigns the correct default paths for test features, test target, model, and metrics output
    """
    args = parse_args([])

    assert args.test_features == Path("data/split_data/test_features.csv")
    assert args.test_target == Path("data/split_data/test_target.csv")
    assert args.model_path == Path("models/best_model.joblib")
    assert args.output_metrics == Path("metrics.json")


def test_parse_args_custom_values():
    """
    GIVEN: a list containing custom CLI flags (--test-features, --output-metrics, etc.)
    WHEN: parse_args is executed with this list
    THEN: it correctly overrides default values, assigning the provided custom paths
    """
    args = parse_args([
        "--test-features",
        "custom/feat.csv",
        "--test-target",
        "custom/target.csv",
        "--model-path",
        "custom/model.joblib",
        "--output-metrics",
        "custom/out.json",
    ])

    assert args.test_features == Path("custom/feat.csv")
    assert args.test_target == Path("custom/target.csv")
    assert args.model_path == Path("custom/model.joblib")
    assert args.output_metrics == Path("custom/out.json")