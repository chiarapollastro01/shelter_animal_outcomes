"""Unit tests for the model evaluation module."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from sklearn.dummy import DummyClassifier

from src import config
from src.evaluate import evaluate_model, load_test_data, main, parse_args

# =====================================================================
#                              FIXTURES
# =====================================================================


@pytest.fixture
def fitted_dummy_model() -> DummyClassifier:
    """Fixture providing a fitted classifier with predict/predict_proba/classes_."""
    X_train = pd.DataFrame(
        {
            "f1": [0, 1, 0, 1, 0, 1],
            "f2": [1, 1, 0, 0, 1, 0],
        }
    )
    y_train = pd.Series(
        ["Adoption", "Transfer", "Adoption", "Euthanasia", "Transfer", "Adoption"]
    )
    model = DummyClassifier(strategy="prior")
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def dummy_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Fixture providing a small test dataset containing both Dog and Cat rows."""
    X_test = pd.DataFrame(
        {
            config.SPECIES_COL: ["Dog", "Cat", "Dog", "Cat"],
            config.DATETIME_COL: [
                "2026-01-01 10:00:00",
                "2026-01-02 11:00:00",
                "2026-01-03 12:00:00",
                "2026-01-04 13:00:00",
            ],
            "f1": [0, 1, 0, 1],
            "f2": [1, 1, 0, 0],
        }
    )
    y_test = pd.Series(
        ["Adoption", "Transfer", "Adoption", "Euthanasia"],
        name=config.TARGET_COL,
    )
    return X_test, y_test


@pytest.fixture
def dog_only_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Fixture providing a test dataset containing only Dog rows."""
    X_test = pd.DataFrame(
        {
            config.SPECIES_COL: ["Dog", "Dog"],
            "f1": [0, 1],
            "f2": [1, 0],
            config.DATETIME_COL: [
            "2026-01-01 10:00:00", "2026-01-02 11:00:00"],
        }
    )
    y_test = pd.Series(
        ["Adoption", "Transfer"],
        name=config.TARGET_COL,
    )
    return X_test, y_test


@pytest.fixture
def saved_species_models(
    tmp_path: Path, fitted_dummy_model: DummyClassifier
) -> Path:
    """Fixture saving one dummy model per species in a temp models directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(fitted_dummy_model, models_dir / "best_shelter_model_dog.pkl")
    joblib.dump(fitted_dummy_model, models_dir / "best_shelter_model_cat.pkl")
    return models_dir


# =====================================================================
#                        evaluate_model TESTS
# =====================================================================

class TestEvaluateModel:

    def test_evaluate_model_returns_expected_metric_keys(self, fitted_dummy_model: DummyClassifier):
        """Verify that evaluate_model returns the complete set of required metric keys.

        GIVEN: a fitted classifier and a valid test dataset
        WHEN: evaluate_model is executed
        THEN: all expected metric keys are returned
        """
        X_test = pd.DataFrame({"f1": [0, 1], "f2": [1, 0]})
        y_test = pd.Series(["Adoption", "Transfer"])
        metrics = evaluate_model(fitted_dummy_model, X_test, y_test)
        assert set(metrics.keys()) == {
            "log_loss",
            "accuracy",
            "f1_macro",
            "f1_weighted",
        }

    def test_evaluate_model_returns_float_values(self, fitted_dummy_model: DummyClassifier,):
        """Ensure all metric values in the returned dictionary are floating-point numbers.

        GIVEN: a fitted classifier and a valid test dataset
        WHEN: evaluate_model is executed
        THEN: every metric value is returned as float and not a numpy scalar, 
              which json.dumps would refuse to serialise
        """
        X_test = pd.DataFrame({"f1": [0, 1], "f2": [1, 0]})
        y_test = pd.Series(["Adoption", "Transfer"])
        metrics = evaluate_model(fitted_dummy_model, X_test, y_test)
        assert all(isinstance(value, float) for value in metrics.values())

    def test_evaluate_model_handles_missing_class_in_test_split(self, fitted_dummy_model: DummyClassifier,):
        """Ensure evaluation succeeds when the test split is missing a class seen during training.

        GIVEN: a fitted classifier trained on 3 classes, and a test set missing one rare class
        WHEN: evaluate_model is executed
        THEN: log_loss still computes correctly thanks to labels=model.classes_
        """
        X_test = pd.DataFrame({"f1": [0, 1], "f2": [1, 0]})
        y_test = pd.Series(["Adoption", "Transfer"])  # no Euthanasia in test
        metrics = evaluate_model(fitted_dummy_model, X_test, y_test)
        assert "log_loss" in metrics
        assert metrics["log_loss"] >= 0.0


# =====================================================================
#                        load_test_data TESTS
# =====================================================================

class TestLoadTestData:

    def test_aligned_files_are_read_back(self, tmp_path: Path):
        """Verify that load_test_data correctly loads aligned features and targets as a DataFrame and Series.

        GIVEN: aligned feature and target CSV files
        WHEN: load_test_data is executed
        THEN: it returns a DataFrame and a Series with matching lengths
        """
        features_path = tmp_path / config.TEST_FEATURES_FILE
        target_path = tmp_path / config.TEST_TARGET_FILE
        pd.DataFrame(
            {
                config.SPECIES_COL: ["Dog", "Cat"],
                "f1": [0, 1],
            }
        ).to_csv(features_path, index=False)
        pd.DataFrame({config.TARGET_COL: ["Adoption", "Transfer"]}).to_csv(
            target_path, index=False
        )
        X_test, y_test = load_test_data(features_path, target_path)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)
        assert len(X_test) == len(y_test) == 2

    def test_misaligned_files_raise(self, tmp_path: Path):
        """Ensure load_test_data raises a ValueError when feature and target datasets have mismatched row counts.

        GIVEN: feature and target CSV files with different row counts
        WHEN: load_test_data is executed
        THEN: a ValueError is raised indicating row misalignment
        """
        features_path = tmp_path / config.TEST_FEATURES_FILE
        target_path = tmp_path / config.TEST_TARGET_FILE
        pd.DataFrame(
            {
                config.SPECIES_COL: ["Dog", "Cat", "Dog"],
                "f1": [0, 1, 0],
            }
        ).to_csv(features_path, index=False)
        pd.DataFrame({config.TARGET_COL: ["Adoption"]}).to_csv(
            target_path, index=False
        )
        with pytest.raises(ValueError, match="misaligned"):
            load_test_data(features_path, target_path)

    def test_a_test_file_without_the_target_column_raises(self, tmp_path: Path):
        """Verify that a target file under the wrong column name is refused.

        GIVEN: a test file whose column is not the configured target name
        WHEN: load_test_data is executed
        THEN: a KeyError is raised, the column name being the contract
              prepare_data and evaluate agree on through config alone
        """
        features_path = tmp_path / config.TEST_FEATURES_FILE
        target_path = tmp_path / config.TEST_TARGET_FILE
        pd.DataFrame({config.ID_COL: ["A1"]}).to_csv(features_path, index=False)
        pd.DataFrame({"outcome": ["Adoption"]}).to_csv(target_path, index=False)

        with pytest.raises(KeyError):
            load_test_data(features_path, target_path)


# =====================================================================
#                             main TESTS
# =====================================================================

class TestMain:

    def test_main_creates_metrics_json(self, dummy_test_data, saved_species_models: Path, tmp_path: Path,):
        """Verify that the evaluation pipeline generates a metrics JSON report on disk.
        
        GIVEN: valid test CSV files and per-species saved models
        WHEN: main is executed
        THEN: it creates a JSON metrics report on disk
        """
        X_test, y_test = dummy_test_data
        features_path = tmp_path / "test_features.csv"
        target_path = tmp_path / "test_target.csv"
        output_metrics_path = tmp_path / "metrics.json"
        X_test.to_csv(features_path, index=False)
        y_test.to_frame().to_csv(target_path, index=False)
        main(
            test_features_path=features_path,
            test_target_path=target_path,
            models_dir=saved_species_models,
            output_metrics_path=output_metrics_path,
        )
        assert output_metrics_path.exists()


    def test_main_writes_metrics_for_both_species(self, dummy_test_data, saved_species_models: Path, tmp_path: Path,):
        """Ensure the output report contains evaluation metrics for all present species.

        GIVEN: a test set containing both Dogs and Cats
        WHEN: main is executed
        THEN: the output JSON contains one metrics block for each species
        """
        X_test, y_test = dummy_test_data
        features_path = tmp_path / "test_features.csv"
        target_path = tmp_path / "test_target.csv"
        output_metrics_path = tmp_path / "metrics.json"

        X_test.to_csv(features_path, index=False)
        y_test.to_frame().to_csv(target_path, index=False)
        main(
            test_features_path=features_path,
            test_target_path=target_path,
            models_dir=saved_species_models,
            output_metrics_path=output_metrics_path,
        )
        metrics = json.loads(output_metrics_path.read_text())
        assert set(metrics.keys()) == {"dog", "cat"}
        expected_metrics = {"log_loss", "accuracy", "f1_macro", "f1_weighted"}
        assert set(metrics["dog"].keys()) == expected_metrics
        assert set(metrics["cat"].keys()) == expected_metrics
    


    def test_main_skips_missing_species_with_warning(self, dog_only_test_data, saved_species_models: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture,):
        """Verify that missing species in the test split are skipped with a logged warning.
        
        GIVEN: a test set containing only Dog rows
        WHEN: main is executed
        THEN: Cat is skipped with a warning and only Dog metrics are written
        """
        X_test, y_test = dog_only_test_data
        features_path = tmp_path / "test_features.csv"
        target_path = tmp_path / "test_target.csv"
        output_metrics_path = tmp_path / "metrics.json"
        X_test.to_csv(features_path, index=False)
        y_test.to_frame().to_csv(target_path, index=False)
        with caplog.at_level("WARNING"):
            main(
                test_features_path=features_path,
                test_target_path=target_path,
                models_dir=saved_species_models,
                output_metrics_path=output_metrics_path,
            )
        metrics = json.loads(output_metrics_path.read_text())
        assert "dog" in metrics
        assert "cat" not in metrics
        assert "skipping" in caplog.text.lower()


    def test_main_drops_species_column_before_prediction(self, dummy_test_data, saved_species_models: Path, tmp_path: Path,):
        """Ensure the species column is stripped from features before running model evaluation.
        
        GIVEN: a test set containing the species column (AnimalType)
        WHEN: main is executed
        THEN: the species column is removed from X before evaluate_model is called
        """
        X_test, y_test = dummy_test_data
        features_path = tmp_path / "test_features.csv"
        target_path = tmp_path / "test_target.csv"
        output_metrics_path = tmp_path / "metrics.json"

        X_test.to_csv(features_path, index=False)
        y_test.to_frame().to_csv(target_path, index=False)


        with patch("src.evaluate.evaluate_model") as mock_evaluate:
            mock_evaluate.return_value = {
                "log_loss": 0.5,
                "accuracy": 0.8,
                "f1_macro": 0.7,
                "f1_weighted": 0.7,
            }

            main(
                test_features_path=features_path,
                test_target_path=target_path,
                models_dir=saved_species_models,
                output_metrics_path=output_metrics_path,
            )

            assert mock_evaluate.call_count > 0

            for call in mock_evaluate.call_args_list:
                _, X_species, _ = call.args
                assert config.SPECIES_COL not in X_species.columns

    def test_a_test_set_without_any_known_species_raises(self, saved_species_models: Path, tmp_path: Path):
        """Verify that a ValueError is raised when no declared species are found in the test set.

        GIVEN: a test set whose species column holds no declared species
        WHEN: main is executed
        THEN: a ValueError is raised
        """
        X_test = pd.DataFrame({
            config.SPECIES_COL: ["Bird", "Bird"],
            config.DATETIME_COL: ["2026-01-01 10:00:00", "2026-01-02 11:00:00"],
            "f1": [0, 1],
        })
        y_test = pd.Series(["Adoption", "Transfer"], name=config.TARGET_COL)
        features_path = tmp_path / config.TEST_FEATURES_FILE
        target_path = tmp_path / config.TEST_TARGET_FILE
        output_path = tmp_path / config.METRICS_FILE
        X_test.to_csv(features_path, index=False)
        y_test.to_frame().to_csv(target_path, index=False)

        with pytest.raises(ValueError, match="nothing was evaluated"):
            main(features_path, target_path, saved_species_models, output_path)

        assert not output_path.exists()

    def test_a_missing_model_file_raises(self, dummy_test_data, tmp_path: Path):
        """Verify that evaluating before training is reported.

        GIVEN: a test set with both species and an empty models directory
        WHEN: main is executed
        THEN: a FileNotFoundError is raised, since the artefacts evaluate reads
              are the ones train writes and nothing else creates them
        """
        X_test, y_test = dummy_test_data
        features_path = tmp_path / config.TEST_FEATURES_FILE
        target_path = tmp_path / config.TEST_TARGET_FILE
        X_test.to_csv(features_path, index=False)
        y_test.to_frame().to_csv(target_path, index=False)
        empty_models_dir = tmp_path / "no_models"
        empty_models_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            main(features_path, target_path, empty_models_dir, tmp_path / config.METRICS_FILE)

# =====================================================================
#                          parse_args TESTS
# =====================================================================

class TestParseArgs:

    def test_parse_args_defaults(self):
        """Verify that parse_args falls back to config defaults when no arguments are provided.
        
        GIVEN: an empty argument list
        WHEN: parse_args is executed
        THEN: all default evaluation paths are assigned correctly
        """
        args = parse_args([])
        assert args.test_features == config.SPLIT_DATA_DIR / config.TEST_FEATURES_FILE
        assert args.test_target == config.SPLIT_DATA_DIR / config.TEST_TARGET_FILE
        assert args.models_dir == config.MODELS_DIR
        assert args.output_metrics == config.REPORTS_DIR / config.METRICS_FILE


    def test_parse_args_custom_values(self):
        """Verify that parse_args correctly overrides defaults with provided arguments.
    
        GIVEN: custom CLI arguments for all input and output paths
        WHEN: parse_args is executed with custom flags
        THEN: all arguments are overridden with Path objects matching the provided inputs
        """
        args = parse_args(
            [
                "--test-features",
                "custom/path/features.csv",
                "--test-target",
                "custom/path/target.csv",
                "--models-dir",
                "custom/path/models",
                "--output-metrics",
                "custom/path/metrics.json",
            ]
        )
        assert args.test_features == Path("custom/path/features.csv")
        assert args.test_target == Path("custom/path/target.csv")
        assert args.models_dir == Path("custom/path/models")
        assert args.output_metrics == Path("custom/path/metrics.json")

    def test_an_unknown_option_is_refused(self):
        """Verify that a misspelt option stops the run.

        GIVEN: an argument list holding an option the parser does not declare
        WHEN: parse_args is executed
        THEN: SystemExit is raised, argparse reporting the error itself
        """
        with pytest.raises(SystemExit):
            parse_args(["--nonexistent", "value"])