"""
Test suite for the model training module.

Validates the complete training pipeline, including hyperparameter 
configurations, data integrity, row filtering, 
species-specific model tournaments, proper artifact saving 
(.pkl models and .json metadata), and CLI parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.train import (
    build_search_spaces,
    drop_rows_missing_critical,
    load_training_data,
    main,
    run_tournament,
    train_one_species,
    parse_args
)


# =====================================================================
#                             FIXTURES
# =====================================================================

@pytest.fixture
def raw_mock_shelter_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Fixture providing an 84-row DataFrame with features (`X`) and a target Series (`y`),
    including intentional data defects to test data cleaning and filtering.
    """
    n = 84
    datetimes = list(pd.date_range("2026-01-01", periods=n, freq="h").astype(str))
    datetimes[0] = np.nan  # defective row 1: missing DateTime

    animal_types = ["Dog", "Cat"] * (n // 2)
    animal_types[5] = np.nan  # defective row 2: missing AnimalType

    X = pd.DataFrame({
        "AnimalID": [f"A{i}" for i in range(n)],
        "Name": ["Bella", "Max", None, "Luna"] * (n // 4),
        "DateTime": datetimes,
        "AnimalType": animal_types,
        "SexuponOutcome": ["Neutered Male", "Spayed Female", "Intact Male", "Unknown"] * (n // 4),
        "AgeuponOutcome": ["1 year", "2 years", "3 weeks", "5 months"] * (n // 4),
        "Breed": ["Labrador Mix", "Siamese", "Beagle", "Persian"] * (n // 4),
        "Color": ["Black", "White", "Brown", "Red"] * (n // 4),
    })
    y = pd.Series(
        ["Adoption", "Adoption", "Transfer", "Euthanasia"] * (n // 4), name="target"
    )
    return X, y


@pytest.fixture
def minimal_search_spaces() -> dict[str, dict[str, list]]:
    """
    Fixture providing single parameter grids for each registered model family
    (so tournament tests stay fast).
    """
    common = {
        "categorical_eng__max_other_ratio": [0.15],
        "smote__k_neighbors": [2],
    }
    return {
        "knn": {**common, "clf__n_neighbors": [3]},
        "logistic_regression": {**common, "clf__C": [1.0]},
        "random_forest": {**common, "clf__n_estimators": [10]},
    }


# =====================================================================
#                           UNIT TESTS: DATA
# =====================================================================


def test_build_search_spaces_structure():
    """
    GIVEN: the build_search_spaces function
    WHEN: executed
    THEN: it returns one grid per registered family, each containing the
          shared parameters merged in
    """
    spaces = build_search_spaces()

    assert set(spaces) == {"knn", "logistic_regression", "random_forest"}
    for grid in spaces.values():
        assert "categorical_eng__max_other_ratio" in grid
        assert "smote__k_neighbors" in grid


def test_load_training_data_success(tmp_path: Path):
    """
    GIVEN: feature and target CSVs with matching row counts
    WHEN: load_training_data is invoked
    THEN: it returns an aligned (DataFrame, Series) pair
    """
    features_path = tmp_path / "features.csv"
    target_path = tmp_path / "target.csv"
    pd.DataFrame({"AnimalID": ["A1", "A2"]}).to_csv(features_path, index=False)
    pd.DataFrame({"target": ["Adoption", "Transfer"]}).to_csv(target_path, index=False)

    X, y = load_training_data(features_path, target_path)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y) == 2


def test_load_training_data_misaligned_length_raises_value_error(tmp_path: Path):
    """
    GIVEN: feature and target CSVs with different row counts
    WHEN: load_training_data is executed
    THEN: a ValueError mentioning the misalignment is raised (fail fast)
    """
    features_path = tmp_path / "features.csv"
    target_path = tmp_path / "target.csv"
    pd.DataFrame({"AnimalID": ["A1", "A2", "A3"]}).to_csv(features_path, index=False)
    pd.DataFrame({"target": ["Adoption"]}).to_csv(target_path, index=False)

    with pytest.raises(ValueError, match="misaligned"):
        load_training_data(features_path, target_path)


def test_drop_rows_missing_critical(raw_mock_shelter_data: tuple[pd.DataFrame, pd.Series]):
    """"
    GIVEN: features with NaNs in DateTime (one row) and AnimalType (another)
    WHEN: drop_rows_missing_critical is executed
    THEN: exactly those two rows are removed and X/y stay index-aligned
    """
    X_raw, y_raw = raw_mock_shelter_data

    X_clean, y_clean = drop_rows_missing_critical(X_raw, y_raw)

    assert len(X_clean) == len(X_raw) - 2
    assert (X_clean.index == y_clean.index).all()
    assert X_clean["DateTime"].isna().sum() == 0
    assert X_clean["AnimalType"].isna().sum() == 0
    assert X_clean["AnimalType"].isna().sum() == 0


# =====================================================================
#                 UNIT & INTEGRATION TESTS: TOURNAMENTS
# =====================================================================

def test_run_tournament(raw_mock_shelter_data, minimal_search_spaces):
    """
    GIVEN: species-filtered clean data (as run_tournament receives it in
           production: no AnimalType column) and minimal grids
    WHEN: run_tournament is executed
    THEN: it returns the winning family name and a fitted GridSearchCV
    """
    X_raw, y_raw = raw_mock_shelter_data
    X_clean, y_clean = drop_rows_missing_critical(X_raw, y_raw)
    dog_mask = X_clean["AnimalType"] == "Dog"
    X_dogs = X_clean.loc[dog_mask].drop(columns=["AnimalType"])
    y_dogs = y_clean.loc[dog_mask]
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    with patch("src.train.build_search_spaces", return_value=minimal_search_spaces):
        best_name, best_search = run_tournament(X_dogs, y_dogs, cv)

    assert best_name in minimal_search_spaces
    assert hasattr(best_search, "best_estimator_")
    assert best_search.best_score_ > 0.0


def test_train_one_species_creates_artifacts(
    raw_mock_shelter_data, minimal_search_spaces, tmp_path: Path
):
    """
    GIVEN: species-filtered features/target and a models directory
    WHEN: train_one_species is executed
    THEN: both the .pkl model and the .json metadata sidecar are created,
          and the metadata records CV and hold-out scores
    """
    X_raw, y_raw = raw_mock_shelter_data
    X_clean, y_clean = drop_rows_missing_critical(X_raw, y_raw)
    dog_mask = X_clean["AnimalType"] == "Dog"
    X_dogs = X_clean.loc[dog_mask].drop(columns=["AnimalType"])
    y_dogs = y_clean.loc[dog_mask]

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    with patch("src.train.build_search_spaces", return_value=minimal_search_spaces):
        train_one_species(X_dogs, y_dogs, models_dir, species="Dog")

    metadata_path = models_dir / "best_shelter_model_dog.json"
    assert (models_dir / "best_shelter_model_dog.pkl").exists()
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["species"] == "Dog"
    assert metadata["model"] in minimal_search_spaces
    assert "cv_f1_macro" in metadata
    assert "holdout_f1_macro" in metadata
    assert "best_params" in metadata


def test_main_pipeline_integration(
    raw_mock_shelter_data, minimal_search_spaces, tmp_path: Path
):
    """
    GIVEN: raw CSV files on disk and a non-existent models directory
    WHEN: main is invoked
    THEN: models and metadata sidecars are produced for BOTH species and the
          models directory is created on the fly
    """
    X_raw, y_raw = raw_mock_shelter_data
    features_path = tmp_path / "raw_features.csv"
    target_path = tmp_path / "raw_target.csv"
    models_dir = tmp_path / "models"  # deliberately not created beforehand

    X_raw.to_csv(features_path, index=False)
    y_raw.to_frame().to_csv(target_path, index=False)

    with patch("src.train.build_search_spaces", return_value=minimal_search_spaces):
        main(features_path, target_path, models_dir)

    for species in ("dog", "cat"):
        assert (models_dir / f"best_shelter_model_{species}.pkl").exists()
        assert (models_dir / f"best_shelter_model_{species}.json").exists()

# =====================================================================
#                        UNIT TESTS: CLI PARSING
# =====================================================================

def test_parse_args_defaults():
    """
    GIVEN: an empty argument list (positionals declared with nargs="?")
    WHEN: parse_args is executed
    THEN: all three paths fall back to their documented defaults
    """
    args = parse_args([])

    assert args.features_path == Path("data/split_data/train_features.csv")
    assert args.target_path == Path("data/split_data/train_target.csv")
    assert args.models_dir == Path("models")


def test_parse_args_custom_values():
    """
    GIVEN: explicit positional paths and a custom --models-dir
    WHEN: parse_args is executed
    THEN: the provided values override every default
    """
    args = parse_args([
        "custom/features.csv",
        "custom/target.csv",
        "--models-dir", "custom/models",
    ])

    assert args.features_path == Path("custom/features.csv")
    assert args.target_path == Path("custom/target.csv")
    assert args.models_dir == Path("custom/models")