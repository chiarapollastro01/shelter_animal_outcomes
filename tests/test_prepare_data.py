"""Unit tests for the data preparation module."""

from pathlib import Path
import pandas as pd
import pytest
from src import config
from src.prepare_data import main, prepare_and_split_data, parse_args

# =====================================================================
#                              FIXTURE
# =====================================================================

@pytest.fixture
def dummy_raw_csv(tmp_path: Path) -> Path:
    """Fixture providing a temporary raw CSV file with dummy data."""
    df = pd.DataFrame({
        config.ID_COL: [f"A{i}" for i in range(1, 11)],
        config.TARGET_COL: ["Adoption", "Transfer"]*5,
        config.SUBTARGET_COL: ["Partner", "Foster"]*5,
        config.SPECIES_COL: ["Dog", "Cat"]*5,
        config.AGE_COL: ["1 year", "2 years"]*5
    })
    file_path = tmp_path / "raw_train.csv"
    df.to_csv(file_path, index=False)
    return file_path


# =====================================================================
#                              TESTING
# =====================================================================

def test_prepare_and_split_data_success(dummy_raw_csv: Path):
    """
    GIVEN: a valid raw CSV file containing target and leakage columns
    WHEN: prepare_and_split_data is executed
    THEN: it returns X_train, X_test, y_train, y_test, with correct split 
    proportions and OutcomeType and OutcomeSubtype columns removed
    """
    X_train, X_test, y_train, y_test = prepare_and_split_data(
        dummy_raw_csv, config.DEFAULT_TEST_SIZE, config.RANDOM_STATE
    )

    assert config.TARGET_COL not in X_train.columns
    assert config.SUBTARGET_COL not in X_train.columns
    assert config.TARGET_COL not in X_test.columns
    assert config.SUBTARGET_COL not in X_test.columns

    # 10 rows total -> 80% train (8 rows), 20% test (2 rows)
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2

    assert y_train.name == "target"
    assert y_test.name == "target"

    # Check stratification (50% Adoption, 50% Transfer in both splits)
    assert (y_train == "Adoption").sum() == 4
    assert (y_test == "Adoption").sum() == 1


def test_prepare_and_split_data_missing_target(tmp_path: Path):
    """
    GIVEN: a raw CSV file that is missing the target column (OutcomeType)
    WHEN: prepare_and_split_data is executed
    THEN: a KeyError is raised with an informative message
    """
    invalid_csv = tmp_path / "invalid.csv"
    pd.DataFrame({config.ID_COL: ["A1", "A2"]}).to_csv(invalid_csv, index=False)

    with pytest.raises(KeyError, match="missing from input file"):
        prepare_and_split_data(invalid_csv)


def test_prepare_and_split_data_handles_missing_subtype(tmp_path: Path):
    """
    GIVEN: a raw CSV file containing OutcomeType but missing OutcomeSubtype
    WHEN: prepare_and_split_data is executed
    THEN: it safely extracts the target and drops OutcomeType without raising KeyError
    """
    df = pd.DataFrame({
        config.ID_COL: [f"A{i}" for i in range(1, 11)],
        config.TARGET_COL: ["Adoption", "Transfer"]*5,
        config.SPECIES_COL: ["Dog", "Cat"]*5,
    })
    partial_csv = tmp_path / "partial.csv"
    df.to_csv(partial_csv, index=False)

    X_train, X_test, y_train, y_test = prepare_and_split_data(partial_csv)

    assert config.TARGET_COL not in X_train.columns
    assert config.SUBTARGET_COL not in X_test.columns
    assert config.SPECIES_COL in X_train.columns

def test_prepare_and_split_data_fallback_without_animal_type(tmp_path: Path):
    """Verify safe fallback to y stratification when AnimalType column is missing.

    GIVEN: a raw CSV file containing OutcomeType but lacking the AnimalType column
    WHEN: prepare_and_split_data is executed
    THEN: execution succeeds without KeyError, falling back to stratifying on y alone
    """
    df = pd.DataFrame({
        config.TARGET_COL: ["Adoption", "Transfer"] * 5,
        config.AGE_COL: ["1 year", "2 years"] * 5
    })
    csv_path = tmp_path / "no_animal_type.csv"
    df.to_csv(csv_path, index=False)

    X_train, X_test, y_train, y_test = prepare_and_split_data(
        csv_path, test_size=0.2, random_state=42
    )

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert (y_train == "Adoption").sum() == 4
    assert (y_test == "Adoption").sum() == 1


def test_prepare_and_split_data_stratifies_by_species_and_outcome(tmp_path: Path):
    """Verify that composite stratification preserves species-outcome subgroup proportions.

    GIVEN: a CSV where outcome and species are deliberately decorrelated and
           unbalanced (Adoption: 10 Dog / 5 Cat; Transfer: 5 Dog / 10 Cat)
    WHEN: prepare_and_split_data is executed with test_size=0.2
    THEN: every outcome-species subgroup keeps its exact proportion in the test split
    """
    rows = (
        [("Adoption", "Dog")] * 10 + [("Adoption", "Cat")] * 5
        + [("Transfer", "Dog")] * 5 + [("Transfer", "Cat")] * 10
    )
    df = pd.DataFrame({
        config.ID_COL: [f"A{i}" for i in range(len(rows))],
        config.TARGET_COL: [outcome for outcome, _ in rows],
        config.SPECIES_COL: [species for _, species in rows],
        config.AGE_COL: ["1 year"] * len(rows),
    })
    csv_path = tmp_path / "uncorrelated.csv"
    df.to_csv(csv_path, index=False)

    X_train, X_test, y_train, y_test = prepare_and_split_data(
        csv_path, test_size=0.2, random_state=config.RANDOM_STATE
    )

    test_combos = (y_test + "_" + X_test[config.SPECIES_COL]).value_counts()
    assert test_combos["Adoption_Dog"] == 2   # 20% of 10
    assert test_combos["Adoption_Cat"] == 1   # 20% of 5
    assert test_combos["Transfer_Dog"] == 1   # 20% of 5
    assert test_combos["Transfer_Cat"] == 2   # 20% of 10


def test_main_execution_creates_files(dummy_raw_csv: Path, tmp_path: Path):
    """
    GIVEN: valid input paths and non-existent output directories
    WHEN: main is executed
    THEN: it creates the directory and saves all 4 train/test feature and target CSV files
    """
    output_dir = tmp_path / "processed_data"

    assert not output_dir.exists()

    main(
        raw_csv_path=dummy_raw_csv,
        output_dir=output_dir,
        test_size=config.DEFAULT_TEST_SIZE,
        random_state=config.RANDOM_STATE,
        )

    assert output_dir.exists()

    train_features_path = output_dir / "train_features.csv"
    test_features_path = output_dir / "test_features.csv"
    train_target_path = output_dir / "train_target.csv"
    test_target_path = output_dir / "test_target.csv"

    assert train_features_path.is_file()
    assert test_features_path.is_file()
    assert train_target_path.is_file()
    assert test_target_path.is_file()

    X_train_saved = pd.read_csv(train_features_path)
    y_train_saved = pd.read_csv(train_target_path)

    assert config.TARGET_COL not in X_train_saved.columns
    assert y_train_saved.columns[0] == "target"
    assert len(X_train_saved) == 8


def test_parse_args_defaults():
    """
    GIVEN: an empty argument list
    WHEN: the parse_args function is executed with this list
    THEN: raw_csv_path, output_dir, test_size and random_state all fall back to their config defaults
    """
    
    args = parse_args([])

    assert args.raw_csv_path == config.RAW_DATA_PATH
    assert args.output_dir == config.SPLIT_DATA_DIR
    assert args.test_size == config.DEFAULT_TEST_SIZE
    assert args.random_state == config.RANDOM_STATE


def test_parse_args_custom_values():
    """
    GIVEN: custom arguments for output-dir, test-size, and random-state
    WHEN: the parse_args function is executed with this list
    THEN: it correctly overrides all default values
    """
    args = parse_args([
        "my_raw_data.csv",
        "--output-dir",
        "custom/dir",
        "--test-size",
        "0.3",
        "--random-state",
        "123",
    ])
    
    assert args.raw_csv_path == Path("my_raw_data.csv")
    assert args.output_dir == Path("custom/dir")
    assert args.test_size == 0.3
    assert args.random_state == 123