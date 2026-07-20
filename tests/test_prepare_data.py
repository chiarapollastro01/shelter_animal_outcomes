"""Unit tests for the data preparation module."""

from pathlib import Path
import sys
import runpy
import pandas as pd
import pytest

from src.prepare_data import main, prepare_and_split_data, parse_args

# =====================================================================
#                              FIXTURE
# =====================================================================

@pytest.fixture
def dummy_raw_csv(tmp_path: Path) -> Path:
    """Fixture providing a temporary raw CSV file with dummy data."""
    df = pd.DataFrame({
        "AnimalID": ["A1", "A2", "A3"],
        "OutcomeType": ["Adoption", "Transfer", "Euthanasia"],
        "OutcomeSubtype": ["Partner", "Foster", "Rabies"],
        "AnimalType": ["Dog", "Cat", "Dog"],
        "AgeuponOutcome": ["1 year", "2 years", "1 month"]
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
    THEN: it returns X and y, removing OutcomeType and OutcomeSubtype from X
    """
    X, y = prepare_and_split_data(dummy_raw_csv)

    assert "OutcomeType" not in X.columns
    assert "OutcomeSubtype" not in X.columns
    assert "AnimalType" in X.columns
    assert len(X) == 3
    assert len(y) == 3
    assert y.name == "target"
    assert list(y) == ["Adoption", "Transfer", "Euthanasia"]


def test_prepare_and_split_data_missing_target(tmp_path: Path):
    """
    GIVEN: a raw CSV file that is missing the target column (OutcomeType)
    WHEN: prepare_and_split_data is executed
    THEN: a KeyError is raised with an informative message
    """
    invalid_csv = tmp_path / "invalid.csv"
    pd.DataFrame({"AnimalID": ["A1", "A2"]}).to_csv(invalid_csv, index=False)

    with pytest.raises(KeyError, match="missing from input file"):
        prepare_and_split_data(invalid_csv)


def test_prepare_and_split_data_handles_missing_subtype(tmp_path: Path):
    """
    GIVEN: a raw CSV file containing OutcomeType but missing OutcomeSubtype
    WHEN: prepare_and_split_data is executed
    THEN: it safely extracts the target and drops OutcomeType without raising KeyError
    """
    df = pd.DataFrame({
        "AnimalID": ["A1"],
        "OutcomeType": ["Adoption"],
        "AnimalType": ["Dog"]
    })
    partial_csv = tmp_path / "partial.csv"
    df.to_csv(partial_csv, index=False)

    X, y = prepare_and_split_data(partial_csv)

    assert "OutcomeType" not in X.columns
    assert "AnimalType" in X.columns


def test_main_execution_creates_files(dummy_raw_csv: Path, tmp_path: Path):
    """
    GIVEN: valid input paths and non-existent output directories
    WHEN: main is executed
    THEN: it creates the necessary directories and saves X and y as CSV files
    """
    out_features = tmp_path / "processed" / "test_features.csv"
    out_target = tmp_path / "processed" / "test_target.csv"

    assert not out_features.parent.exists()

    main(dummy_raw_csv, out_features, out_target)

    assert out_features.parent.exists()
    assert out_features.is_file()
    assert out_target.is_file()

    X_saved = pd.read_csv(out_features)
    y_saved = pd.read_csv(out_target)

    assert "OutcomeType" not in X_saved.columns
    assert y_saved.columns[0] == "target"
    assert list(y_saved["target"]) == ["Adoption", "Transfer", "Euthanasia"]


def test_parse_args_defaults():
    """
    GIVEN: a list containing only the required positional argument (raw_csv_path)
    WHEN: the parse_args function is executed with this list
    THEN: the parsed namespace contains the correct positional path and assigns 
          the expected default paths to both optional arguments
    """
    args = parse_args(["my_raw_data.csv"])
    
    assert args.raw_csv_path == Path("my_raw_data.csv")
    assert args.output_features == Path("data/split_data/train_features.csv")
    assert args.output_target == Path("data/split_data/train_target.csv")


def test_parse_args_custom_values():
    """
    GIVEN: a list containing the required positional argument and custom values 
           for both optional flags (--output-features and --output-target)
    WHEN: the parse_args function is executed with this list
    THEN: the parsed namespace correctly overrides the default values, assigning 
          the custom paths provided
    """
    args = parse_args([
        "my_raw_data.csv",
        "--output-features", "custom/features.csv",
        "--output-target", "custom/target.csv"
    ])
    
    assert args.raw_csv_path == Path("my_raw_data.csv")
    assert args.output_features == Path("custom/features.csv")
    assert args.output_target == Path("custom/target.csv")


def test_script_execution_as_main(dummy_raw_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    GIVEN: a valid raw CSV and target output paths mocked in sys.argv
    WHEN: the prepare_data module is executed in-process as __main__ using run_module
    THEN: the entire entry point is executed, all CLI lines are covered,
          and the expected output files are successfully created with zero warnings
    """
    out_features = tmp_path / "cli_features.csv"
    out_target = tmp_path / "cli_target.csv"

    monkeypatch.setattr(sys, "argv", [
        "src/prepare_data.py",
        str(dummy_raw_csv),
        "--output-features", str(out_features),
        "--output-target", str(out_target)
    ])

    # Clear the module cache to simulate a completely fresh script invocation.
    # Because previous tests already imported this module into memory,
    # runpy would throw a RuntimeWarning about re-executing an active module.
    sys.modules.pop("src.prepare_data", None)
    sys.modules.pop("prepare_data", None)

    runpy.run_module("src.prepare_data", run_name="__main__")

    assert out_features.exists()
    assert out_target.exists()