"""Test suite for the training module (src/train.py).


"""

from __future__ import annotations

import joblib
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src import config
from src.preprocessing import drop_rows_missing_required
from src.train import (
    load_training_data,
    main,
    merge_search_grids,
    parse_args,
    run_tournament,
    train_one_species,
)

# =====================================================================
#                              FIXTURES
# =====================================================================


@pytest.fixture
def mock_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Aligned mock features and target with exactly two defective rows.

    Design constraints, deliberate: 84 rows in a period-4 pattern, so that
    after the species split each species keeps around twenty rows per class,
    enough for the hold-out split, the internal cross-validation and SMOTE
    with two neighbours. The two missing values sit on different rows, one
    lacking the timestamp and one the species, so that a test can assert that
    exactly two rows are dropped.
    """
    n = 84

    datetimes = list(pd.date_range("2026-01-01", periods=n, freq="h").astype(str))
    datetimes[0] = np.nan

    animal_types = ["Dog", "Cat"] * (n // 2)
    animal_types[5] = np.nan

    X = pd.DataFrame(
        {
            config.ID_COL: [f"A{i}" for i in range(n)],
            config.NAME_COL: ["Bella", "Max", None, "Luna"] * (n // 4),
            config.DATETIME_COL: datetimes,
            config.SPECIES_COL: animal_types,
            config.SEX_COL: [
                "Neutered Male",
                "Spayed Female",
                "Intact Male",
                "Unknown",
            ]
            * (n // 4),
            config.AGE_COL: ["1 year", "2 years", "3 weeks", "5 months"] * (n // 4),
            config.BREED_COL: ["Labrador Mix", "Siamese", "Beagle", "Persian"]
            * (n // 4),
            config.COLOR_COL: ["Black", "White", "Brown", "Red"] * (n // 4),
        }
    )
    y = pd.Series(
        ["Adoption", "Adoption", "Transfer", "Euthanasia"] * (n // 4),
        name=config.TARGET_COL,
    )
    return X, y


@pytest.fixture
def minimal_search_grids() -> dict[str, dict[str, list]]:
    """One candidate per family, so a tournament costs one fit per family."""
    common = {
        "categorical_eng__max_other_ratio": [0.15],
        "smote__k_neighbors": [2],
    }
    return {
        "knn": {**common, "clf__n_neighbors": [3]},
        "logistic_regression": {**common, "clf__C": [1.0]},
        "random_forest": {**common, "clf__n_estimators": [10]},
    }


@pytest.fixture
def minimal_run_params() -> dict:
    """Run parameters small enough to keep the tournament tests quick."""
    return {
        "holdout_size": 0.2,
        "cv_n_splits": 2,
        "scoring": ["f1_macro", "accuracy"],
        "refit": "f1_macro",
    }


@pytest.fixture
def minimal_config_file(tmp_path: Path, minimal_run_params, minimal_search_grids) -> Path:
    """A YAML parameter file mirroring the two fixtures above.

    main reads its parameters from disk rather than receiving them, so an
    integration test needs a real file; using the project's own would run the
    full search grid.
    """
    import yaml

    search_spaces = {"common": {}, **minimal_search_grids}
    path = tmp_path / "params.yaml"
    path.write_text(
        yaml.safe_dump({**minimal_run_params, "search_spaces": search_spaces}),
        encoding="utf-8",
    )
    return path


def species_slice(
    X: pd.DataFrame, y: pd.Series, species: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter one species and drop the species column, as main does.

    The duplication of main's own slicing is deliberate: a test that reused
    main would no longer be testing the function it names.
    """
    mask = X[config.SPECIES_COL] == species
    X_species = X.loc[mask].drop(columns=[config.SPECIES_COL]).reset_index(drop=True)
    y_species = y.loc[mask].reset_index(drop=True)
    return X_species, y_species


@pytest.fixture
def dog_slice(mock_training_data) -> tuple[pd.DataFrame, pd.Series]:
    """The dog rows, cleaned and sliced as the tournament receives them."""
    X, y = mock_training_data
    X_clean, y_clean = drop_rows_missing_required(X, y)
    return species_slice(X_clean, y_clean, "Dog")


# =====================================================================
#                        MERGE SEARCH GRIDS
# =====================================================================


class TestMergeSearchGrids:
    """Testing the assembly of the per-family grids from the YAML entries."""

    def test_the_shared_entry_reaches_every_family(self):
        """Verify that the common parameters are merged into each grid.

        GIVEN: search spaces holding a shared entry and two families
        WHEN: merge_search_grids is executed
        THEN: both grids carry the shared key alongside their own
        """
        spaces = {
            "common": {"smote__k_neighbors": [3]},
            "knn": {"clf__n_neighbors": [5]},
            "random_forest": {"clf__n_estimators": [10]},
        }

        grids = merge_search_grids(spaces)

        assert all("smote__k_neighbors" in grid for grid in grids.values())
        assert grids["knn"]["clf__n_neighbors"] == [5]

    def test_a_family_overrides_a_shared_key(self):
        """Verify that a family may narrow a parameter the shared entry sets.

        GIVEN: search spaces where a family repeats a key of the shared entry
        WHEN: merge_search_grids is executed
        THEN: the family's own value wins, the shared entry being a default
        """
        spaces = {
            "common": {"smote__k_neighbors": [3, 5]},
            "knn": {"smote__k_neighbors": [3]},
        }

        grids = merge_search_grids(spaces)

        assert grids["knn"]["smote__k_neighbors"] == [3]

    def test_the_shared_entry_is_not_a_family(self):
        """Verify that the shared entry does not become a grid of its own.

        GIVEN: search spaces holding the shared entry and one family
        WHEN: merge_search_grids is executed
        THEN: only the family appears among the returned grids
        """
        spaces = {"common": {"smote__k_neighbors": [3]}, "knn": {}}

        grids = merge_search_grids(spaces)

        assert set(grids) == {"knn"}

    def test_an_unknown_family_raises(self):
        """Verify that a family the pipeline cannot build is refused.

        GIVEN: search spaces naming a classifier that is not registered
        WHEN: merge_search_grids is executed
        THEN: a ValueError names it, instead of the failure surfacing minutes
              later inside the first grid search
        """
        spaces = {"common": {}, "xgboost": {"clf__eta": [0.1]}}

        with pytest.raises(ValueError, match="xgboost"):
            merge_search_grids(spaces)

    def test_a_missing_shared_entry_raises(self):
        """Verify that search spaces without the shared entry are refused.

        GIVEN: search spaces holding only families
        WHEN: merge_search_grids is executed
        THEN: a KeyError is raised, the shared entry being read unguarded
        """
        with pytest.raises(KeyError):
            merge_search_grids({"knn": {"clf__n_neighbors": [5]}})


# =====================================================================
#                        LOAD TRAINING DATA
# =====================================================================


class TestLoadTrainingData:
    """Testing the reader of the two files prepare_data writes."""

    def test_aligned_files_are_read_back(self, tmp_path: Path):
        """Verify that two aligned files come back as features and target.

        GIVEN: a features file and a target file with the same row count
        WHEN: load_training_data is executed
        THEN: it returns a frame and a series of that length
        """
        features_path = tmp_path / config.TRAIN_FEATURES_FILE
        target_path = tmp_path / config.TRAIN_TARGET_FILE
        pd.DataFrame({config.ID_COL: ["A1", "A2"]}).to_csv(features_path, index=False)
        pd.DataFrame({config.TARGET_COL: ["Adoption", "Transfer"]}).to_csv(
            target_path, index=False
        )

        X, y = load_training_data(features_path, target_path)

        assert len(X) == len(y) == 2


    def test_misaligned_files_raise(self, tmp_path: Path):
        """Verify that a row-count mismatch is reported before training.

        GIVEN: a features file and a target file of different lengths
        WHEN: load_training_data is executed
        THEN: a ValueError reports the misalignment, which would otherwise
              surface as a silent mismatch between rows and labels
        """
        features_path = tmp_path / config.TRAIN_FEATURES_FILE
        target_path = tmp_path / config.TRAIN_TARGET_FILE
        pd.DataFrame({config.ID_COL: ["A1", "A2", "A3"]}).to_csv(
            features_path, index=False
        )
        pd.DataFrame({config.TARGET_COL: ["Adoption"]}).to_csv(
            target_path, index=False
        )

        with pytest.raises(ValueError, match="misaligned"):
            load_training_data(features_path, target_path)


    def test_a_target_file_without_the_target_column_raises(self, tmp_path: Path):
        """Verify that a target file under the wrong column name is refused.

        GIVEN: a target file whose column is not the configured target name
        WHEN: load_training_data is executed
        THEN: a KeyError is raised, the column name being the contract
              prepare_data and train agree on through config alone
        """
        features_path = tmp_path / config.TRAIN_FEATURES_FILE
        target_path = tmp_path / config.TRAIN_TARGET_FILE
        pd.DataFrame({config.ID_COL: ["A1"]}).to_csv(features_path, index=False)
        pd.DataFrame({"outcome": ["Adoption"]}).to_csv(target_path, index=False)

        with pytest.raises(KeyError):
            load_training_data(features_path, target_path)


# =====================================================================
#                          RUN TOURNAMENT
# =====================================================================


class TestRunTournament:
    """Testing the grid search across the classifier families."""

    def test_a_winner_is_returned(
        self, dog_slice, minimal_search_grids, minimal_run_params
    ):
        """Verify that the tournament returns a fitted winner.

        GIVEN: the dog rows and one candidate per family
        WHEN: run_tournament is executed
        THEN: it returns a registered family name and a fitted search object
        """
        X_dogs, y_dogs = dog_slice
        cv = StratifiedKFold(
            n_splits=2, shuffle=True, random_state=config.RANDOM_STATE
        )

        best_name, best_search = run_tournament(
            X_dogs, y_dogs, cv, minimal_search_grids, minimal_run_params
        )

        assert best_name in minimal_search_grids
        assert hasattr(best_search, "best_estimator_")

    def test_every_family_is_evaluated(
        self, dog_slice, minimal_search_grids, minimal_run_params,
        caplog: pytest.LogCaptureFixture,
    ):
        """Verify that the search covers all the families, not just the winner.

        GIVEN: grids covering the three registered families
        WHEN: run_tournament is executed with INFO logging captured
        THEN: each family is named in the log, so a family silently skipped
              would not go unnoticed
        """
        X_dogs, y_dogs = dog_slice
        cv = StratifiedKFold(
            n_splits=2, shuffle=True, random_state=config.RANDOM_STATE
        )

        with caplog.at_level("INFO"):
            run_tournament(
                X_dogs, y_dogs, cv, minimal_search_grids, minimal_run_params
            )

        assert all(family in caplog.text for family in minimal_search_grids)

    def test_the_refit_metric_decides_the_winner(
        self, dog_slice, minimal_search_grids, minimal_run_params
    ):
        """Verify that the score compared across families is the refit one.

        GIVEN: run parameters naming a refit metric
        WHEN: run_tournament is executed
        THEN: the winner's best_score_ equals its mean CV score on that
              metric, which is what makes the families comparable at all
        """
        X_dogs, y_dogs = dog_slice
        cv = StratifiedKFold(
            n_splits=2, shuffle=True, random_state=config.RANDOM_STATE
        )

        _, best_search = run_tournament(
            X_dogs, y_dogs, cv, minimal_search_grids, minimal_run_params
        )

        refit = minimal_run_params["refit"]
        expected = best_search.cv_results_[f"mean_test_{refit}"][
            best_search.best_index_
        ]
        assert best_search.best_score_ == pytest.approx(expected)
        
    def test_an_empty_set_of_grids_raises(self, dog_slice, minimal_run_params):
        """Verify that a tournament with nothing to search is refused.

        GIVEN: an empty mapping of search grids
        WHEN: run_tournament is executed
        THEN: a ValueError is raised, instead of the loop returning a winner
              that was never fitted and failing later on best_estimator_
        """
        X_dogs, y_dogs = dog_slice
        cv = StratifiedKFold(
            n_splits=2, shuffle=True, random_state=config.RANDOM_STATE
        )

        with pytest.raises(ValueError, match="empty"):
            run_tournament(X_dogs, y_dogs, cv, {}, minimal_run_params)

# =====================================================================
#                        TRAIN ONE SPECIES
# =====================================================================


class TestTrainOneSpecies:
    """Testing the per-species tournament and the artefacts it writes."""

    def test_both_artefacts_are_written(
        self, dog_slice, minimal_search_grids, minimal_run_params, tmp_path: Path
    ):
        """Verify that the model and its metadata sidecar are both created.

        GIVEN: the dog rows and an existing models directory
        WHEN: train_one_species is executed
        THEN: the pickle and the JSON sidecar are both on disk, under the
              name evaluate will look for
        """
        X_dogs, y_dogs = dog_slice
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        train_one_species(
            X_dogs, y_dogs, models_dir, "Dog", minimal_search_grids, minimal_run_params
        )

        model_path = models_dir / config.MODEL_FILE_TEMPLATE.format(species="dog")
        assert model_path.is_file()
        assert model_path.with_suffix(".json").is_file()

    def test_the_sidecar_records_what_produced_the_model(
        self, dog_slice, minimal_search_grids, minimal_run_params, tmp_path: Path
    ):
        """Verify that the sidecar carries everything needed to audit the run.

        GIVEN: a completed per-species tournament
        WHEN: the sidecar is read back
        THEN: it names the species, the winning family, the metric and the two
              scores, and records the resolved run parameters, without which
              the scores would not say what produced them
        """
        X_dogs, y_dogs = dog_slice
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        train_one_species(
            X_dogs, y_dogs, models_dir, "Dog", minimal_search_grids, minimal_run_params
        )

        model_path = models_dir / config.MODEL_FILE_TEMPLATE.format(species="dog")
        metadata = json.loads(model_path.with_suffix(".json").read_text())

        assert metadata["species"] == "Dog"
        assert metadata["model"] in minimal_search_grids
        assert metadata["metric"] == minimal_run_params["refit"]
        assert metadata["n_samples"] == len(X_dogs)
        assert metadata["run_params"]["refit"] == minimal_run_params["refit"]

    def test_the_holdout_is_scored_on_the_refit_metric(
        self, dog_slice, minimal_search_grids, minimal_run_params, tmp_path: Path
    ):
        """Verify that the two recorded scores are on the same metric.

        GIVEN: a completed per-species tournament
        WHEN: the sidecar is read back
        THEN: the hold-out score is a proportion on the same metric as the
              cross-validated one, the two being meant to be compared
        """
        X_dogs, y_dogs = dog_slice
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        train_one_species(
            X_dogs, y_dogs, models_dir, "Dog", minimal_search_grids, minimal_run_params
        )

        model_path = models_dir / config.MODEL_FILE_TEMPLATE.format(species="dog")
        metadata = json.loads(model_path.with_suffix(".json").read_text())

        assert 0.0 <= metadata["holdout_score"] <= 1.0
        assert 0.0 <= metadata["cv_score"] <= 1.0

    def test_the_saved_model_can_predict_after_reloading(
        self, dog_slice, minimal_search_grids, minimal_run_params, tmp_path: Path
    ):
        """Verify that the artefact is a working model and not just a file.

        GIVEN: a completed tournament whose winner has been written to disk
        WHEN: the pickle is loaded back and asked to predict
        THEN: it returns one label per row, which is what evaluate does with
              it and the only thing that makes the file worth writing
        """
        X_dogs, y_dogs = dog_slice
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        train_one_species(
            X_dogs, y_dogs, models_dir, "Dog", minimal_search_grids, minimal_run_params
        )

        model = joblib.load(
            models_dir / config.MODEL_FILE_TEMPLATE.format(species="dog")
        )
        assert len(model.predict(X_dogs)) == len(X_dogs)

# =====================================================================
#                               MAIN
# =====================================================================


class TestMain:
    """Testing the orchestration of one tournament per species."""

    @pytest.mark.parametrize("species", config.SPECIES)
    def test_the_declared_species_gets_its_model(
        self, mock_training_data, minimal_config_file, tmp_path: Path, species: str
    ):
        """Verify that one invocation produces the model of one species.

        GIVEN: the split files on disk and a models directory that does not
           exist yet
        WHEN: main is executed for one species
        THEN: the directory holds that species' model and sidecar and nothing
              else, the other tournaments being separate invocations
        """
        X, y = mock_training_data
        features_path = tmp_path / config.TRAIN_FEATURES_FILE
        target_path = tmp_path / config.TRAIN_TARGET_FILE
        models_dir = tmp_path / "models"
        X.to_csv(features_path, index=False)
        y.to_frame().to_csv(target_path, index=False)

        main(features_path, target_path, models_dir, minimal_config_file, species)

        model_name = config.MODEL_FILE_TEMPLATE.format(species=species.lower())
        sidecar_name = Path(model_name).with_suffix(".json").name

        written = {path.name for path in models_dir.iterdir()}
        expected = {model_name, sidecar_name}

        assert written == expected

    def test_the_defective_rows_never_reach_the_tournament(
        self, mock_training_data, minimal_config_file, tmp_path: Path
    ):
        """Verify that the rows missing a required column are dropped first.

        GIVEN: split files where one dog carries no timestamp
        WHEN: main is executed for the dogs
        THEN: the sample count in the sidecar is one below the dog rows on
              disk, the filter having run before the species was selected
        """
        X, y = mock_training_data
        features_path = tmp_path / config.TRAIN_FEATURES_FILE
        target_path = tmp_path / config.TRAIN_TARGET_FILE
        models_dir = tmp_path / "models"
        X.to_csv(features_path, index=False)
        y.to_frame().to_csv(target_path, index=False)

        main(features_path, target_path, models_dir, minimal_config_file, "Dog")

        model_path = models_dir / config.MODEL_FILE_TEMPLATE.format(species="dog")
        metadata = json.loads(model_path.with_suffix(".json").read_text())

        dogs_on_disk = (X[config.SPECIES_COL] == "Dog").sum()
        assert metadata["n_samples"] == dogs_on_disk - 1

    def test_a_species_absent_from_the_data_raises(
        self, mock_training_data, minimal_config_file, tmp_path: Path
    ):
        """Verify that training on a species with no rows is refused.

        GIVEN: split files holding dogs only
        WHEN: main is executed for the cats
        THEN: a ValueError names the species
        """
        X, y = mock_training_data
        dogs_only = X[config.SPECIES_COL] == "Dog"
        features_path = tmp_path / config.TRAIN_FEATURES_FILE
        target_path = tmp_path / config.TRAIN_TARGET_FILE
        X[dogs_only].to_csv(features_path, index=False)
        y[dogs_only].to_frame().to_csv(target_path, index=False)

        with pytest.raises(ValueError, match="Cat"):
            main(
                features_path,
                target_path,
                tmp_path / "models",
                minimal_config_file,
                "Cat",
            )

# =====================================================================
#                            PARSE ARGS
# =====================================================================


class TestParseArgs:
    """Testing the command-line interface."""

    def test_the_paths_fall_back_to_the_defaults(self):
        """Verify that every path defaults to its configured location.

        GIVEN: an argument list carrying only the required species
        WHEN: parse_args is executed
        THEN: the four paths come back from config
        """
        args = parse_args(["--species", config.SPECIES[0]])

        assert args.features_path == config.SPLIT_DATA_DIR / config.TRAIN_FEATURES_FILE
        assert args.target_path == config.SPLIT_DATA_DIR / config.TRAIN_TARGET_FILE
        assert args.models_dir == config.MODELS_DIR
        assert args.config_path == config.CONFIG_FILE_PATH

    def test_parse_args_custom_values(self):
        """Verify that the provided values win over the configured ones.

        GIVEN: explicit positional paths, a models directory and a parameter
               file
        WHEN: parse_args is executed
        THEN: every value comes back as given, converted to a Path
        """
        args = parse_args(
            [
                "custom/features.csv",
                "custom/target.csv",
                "--models-dir",
                "custom/models",
                "--config",
                "custom/params.yaml",
                "--species",
                config.SPECIES[0],
            ]
        )

        assert args.features_path == Path("custom/features.csv")
        assert args.target_path == Path("custom/target.csv")
        assert args.models_dir == Path("custom/models")
        assert args.config_path == Path("custom/params.yaml")
        assert args.species == config.SPECIES[0]

    def test_a_missing_species_is_refused(self):
        """Verify that the species cannot be left out.

        GIVEN: an argument list without the species
        WHEN: parse_args is executed
        THEN: SystemExit is raised, there being no sensible species to fall
              back to when one tournament is run per invocation
        """
        with pytest.raises(SystemExit):
            parse_args([])

    def test_an_undeclared_species_is_refused(self):
        """Verify that only the declared species are accepted.

        GIVEN: an argument list naming a species config does not declare
        WHEN: parse_args is executed
        THEN: SystemExit is raised by argparse itself, before any file is read
        """
        with pytest.raises(SystemExit):
            parse_args(["--species", "Bird"])

    def test_an_unknown_option_is_refused(self):
        """Verify that a misspelt option stops the run.

        GIVEN: an argument list holding an option the parser does not declare
        WHEN: parse_args is executed
        THEN: SystemExit is raised, argparse reporting the error itself
        """
        with pytest.raises(SystemExit):
            parse_args(["--species", config.SPECIES[0], "--nonexistent", "value"])
