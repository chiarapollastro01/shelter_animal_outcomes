"""Unit tests for the data preparation module."""

from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.prepare_data import main, parse_args, prepare_and_split_data
from src.train import load_training_data


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

@pytest.fixture
def minimal_config_file(tmp_path: Path) -> Path:
    """A parameter file holding only what prepare_data reads.

    main reads the split proportion from disk, so an integration test needs a
    real file; using the project's own would tie the assertions below to a
    value that belongs to the analysis, not to the test.
    """
    path = tmp_path / "params.yaml"
    path.write_text("test_size: 0.2\n", encoding="utf-8")
    return path

class TestPrepareAndSplitData:
    """Testing the reading, target extraction, and stratified split of the raw file."""

    def test_prepare_and_split_data_success(self, dummy_raw_csv: Path):
        """Test successful execution of data preparation and splitting.

        GIVEN: a valid raw CSV file containing target and leakage columns
        WHEN: prepare_and_split_data is executed
        THEN: it returns X_train, X_test, y_train, y_test, with correct split
        proportions and OutcomeType and OutcomeSubtype columns removed
        """
        X_train, X_test, y_train, y_test = prepare_and_split_data(
            dummy_raw_csv, 0.2, config.RANDOM_STATE
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

        assert y_train.name == config.TARGET_COL
        assert y_test.name == config.TARGET_COL

        assert (y_train == "Adoption").sum() == 4
        assert (y_test == "Adoption").sum() == 1


    def test_prepare_and_split_data_missing_target(self, tmp_path: Path):
        """Test handling of missing target column.

        GIVEN: a raw CSV file that is missing the target column (OutcomeType)
        WHEN: prepare_and_split_data is executed
        THEN: a KeyError is raised with an informative message
        """
        invalid_csv = tmp_path / "invalid.csv"
        pd.DataFrame({config.ID_COL: ["A1", "A2"]}).to_csv(invalid_csv, index=False)

        with pytest.raises(KeyError, match="missing from input file"):
            prepare_and_split_data(invalid_csv, 0.2, config.RANDOM_STATE)

    def test_a_missing_target_value_raises(self, tmp_path: Path):
        """Verify that a row with no outcome is reported rather than split.

        GIVEN: a raw file where the target column holds a missing value
        WHEN: prepare_and_split_data is executed
        THEN: a ValueError names the column and counts the missing rows
        """
        df = pd.DataFrame({
            config.TARGET_COL: ["Adoption", None, "Transfer"] * 4,
            config.SPECIES_COL: ["Dog"] * 12,
        })
        csv_path = tmp_path / "missing_target.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match=config.TARGET_COL):
            prepare_and_split_data(csv_path, test_size=0.2, random_state=config.RANDOM_STATE)

    def test_a_file_without_rows_raises(self, tmp_path: Path):
        """Verify that a header-only file fails at the split.

        GIVEN: a raw file carrying the schema but no rows
        WHEN: prepare_and_split_data is executed
        THEN: a ValueError is raised, there being nothing to split
        """
        df = pd.DataFrame(columns=[config.TARGET_COL, config.SPECIES_COL])
        csv_path = tmp_path / "header_only.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="n_samples=0"):
            prepare_and_split_data(csv_path, test_size=0.2, random_state=config.RANDOM_STATE)

    def test_prepare_and_split_data_handles_missing_subtype(self, tmp_path: Path):
        """Verify that the function handles missing outcome subtype column gracefully.

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

        X_train = prepare_and_split_data(partial_csv, 0.2, config.RANDOM_STATE)[0]

        assert config.TARGET_COL not in X_train.columns
        assert config.SPECIES_COL in X_train.columns

    def test_prepare_and_split_data_fallback_without_animal_type(self, tmp_path: Path):
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
            csv_path, test_size=0.2, random_state=config.RANDOM_STATE
        )

        assert len(X_train) == 8
        assert len(X_test) == 2
        assert (y_train == "Adoption").sum() == 4
        assert (y_test == "Adoption").sum() == 1


    def test_prepare_and_split_data_stratifies_by_species_and_outcome(self, tmp_path: Path):
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

        _, X_test, _, y_test = prepare_and_split_data(
            csv_path, test_size=0.2, random_state=config.RANDOM_STATE
        )

        test_combos = (y_test + "_" + X_test[config.SPECIES_COL]).value_counts()
        assert test_combos["Adoption_Dog"] == 2   # 20% of 10
        assert test_combos["Adoption_Cat"] == 1   # 20% of 5
        assert test_combos["Transfer_Dog"] == 1   # 20% of 5
        assert test_combos["Transfer_Cat"] == 2   # 20% of 10

    def test_a_singleton_stratum_raises(self, tmp_path: Path):
        """Verify that a subgroup too small to split is reported, not silently dropped.

        GIVEN: a raw file where one outcome-species combination has a single row
        WHEN: prepare_and_split_data is executed
        THEN: a ValueError is raised by the stratified split
        """
        df = pd.DataFrame({
            config.TARGET_COL: ["Adoption"] * 9 + ["Euthanasia"],
            config.SPECIES_COL: ["Dog"] * 10,
        })
        csv_path = tmp_path / "singleton.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="least populated"):
            prepare_and_split_data(csv_path, test_size=0.2, random_state=config.RANDOM_STATE)


class TestMain:
    """Testing the orchestration of data preparation and the artefacts it writes to disk."""

    def test_the_written_files_are_readable_by_train(
        self, dummy_raw_csv: Path, minimal_config_file: Path, tmp_path: Path):
        """Verify that the artefacts satisfy the contract the next step expects.

        GIVEN: a completed data preparation run
        WHEN: the training loader reads the two files it is pointed at
        THEN: it returns aligned features and target, the file names and the
              target column being agreed on through config alone
        """
        main(
            raw_csv_path=dummy_raw_csv,
            output_dir=tmp_path,
            config_path=minimal_config_file,
            random_state=config.RANDOM_STATE,
        )

        X, y = load_training_data(
            tmp_path / config.TRAIN_FEATURES_FILE,
            tmp_path / config.TRAIN_TARGET_FILE,
        )

        assert len(X) == len(y)

    def test_main_execution_creates_files(
        self, dummy_raw_csv: Path, minimal_config_file: Path, tmp_path: Path
    ):
        """Test that main execution creates output files in a non-existent directory.

        GIVEN: valid input paths and non-existent output directories
        WHEN: main is executed
        THEN: it creates the directory and saves all 4 train/test feature and target CSV files
        """
        output_dir = tmp_path / "processed_data"

        assert not output_dir.exists()

        main(
            raw_csv_path=dummy_raw_csv,
            output_dir=output_dir,
            config_path=minimal_config_file,
            random_state=config.RANDOM_STATE,
            )

        assert output_dir.exists()

        train_features_path = output_dir / config.TRAIN_FEATURES_FILE
        test_features_path = output_dir / config.TEST_FEATURES_FILE
        train_target_path = output_dir / config.TRAIN_TARGET_FILE
        test_target_path = output_dir / config.TEST_TARGET_FILE

        assert train_features_path.is_file()
        assert test_features_path.is_file()
        assert train_target_path.is_file()
        assert test_target_path.is_file()

        X_train_saved = pd.read_csv(train_features_path)
        y_train_saved = pd.read_csv(train_target_path)

        assert config.TARGET_COL not in X_train_saved.columns
        assert y_train_saved.columns[0] == config.TARGET_COL
        assert len(X_train_saved) == 8

class TestParseArgs:
    """Testing the command-line interface."""

    def test_parse_args_defaults(self):
        """Verify that parse_args falls back to config defaults when no arguments are provided.

        GIVEN: an empty argument list
        WHEN: the parse_args function is executed with this list
        THEN: raw_csv_path, output_dir, config_path and random_state all fall back to their config
              defaults
        """

        args = parse_args([])

        assert args.raw_csv_path == config.RAW_DATA_PATH
        assert args.output_dir == config.SPLIT_DATA_DIR
        assert args.config_path == config.CONFIG_FILE_PATH
        assert args.random_state == config.RANDOM_STATE

    def test_parse_args_custom_values(self):
        """Verify that parse_args correctly overrides defaults with provided arguments.

        GIVEN: custom arguments for output-dir, config and random-state
        WHEN: the parse_args function is executed with this list
        THEN: it correctly overrides all default values
        """
        args = parse_args([
            "my_raw_data.csv",
            "--output-dir", "custom/dir",
            "--config", "custom/params.yaml",
            "--random-state", "123",
        ])

        assert args.raw_csv_path == Path("my_raw_data.csv")
        assert args.output_dir == Path("custom/dir")
        assert args.config_path == Path("custom/params.yaml")
        assert args.random_state == 123

    def test_an_unknown_option_is_refused(self):
        """Verify that a misspelt option stops the run.

        GIVEN: an argument list holding an option the parser does not declare
        WHEN: parse_args is executed
        THEN: SystemExit is raised, argparse reporting the error itself
        """
        with pytest.raises(SystemExit):
            parse_args(["--nonexistent", "value"])
