"""Unit tests for the project configuration module (src/config.py).

"""

from pathlib import Path

import pytest
from sklearn.metrics import get_scorer

from src import config

# The full raw Kaggle schema as declared in config, used as the reference
# set for membership checks.
RAW_SCHEMA_COLS = (
    config.ID_COL,
    config.NAME_COL,
    config.DATETIME_COL,
    config.TARGET_COL,
    config.SUBTARGET_COL,
    config.SPECIES_COL,
    config.SEX_COL,
    config.AGE_COL,
    config.BREED_COL,
    config.COLOR_COL,
)

GROUP_NAMES = (
    "LEAKAGE_COLS",
    "CATEGORICAL_COLS",
    "FILL_TARGETS",
    "COLUMNS_TO_REMOVE",
    "CRITICAL_COLS",
    "ESSENTIAL_COLS",
    "NUM_SCALE_COLS",
    "CAT_ENCODE_COLS",
    "SPECIES",
)

# =====================================================================
#                        COLUMN SCHEMA
# =====================================================================

class TestColumnSchema:

    def test_raw_schema_columns_are_unique(self):
        """Verify that no two column constants share the same name.

        GIVEN: the raw schema column constants declared in config
        WHEN: they are collected into a set
        THEN: no duplicates exist (two constants aliasing one column would
              make drop/impute steps silently overlap)
        """
        assert len(set(RAW_SCHEMA_COLS)) == len(RAW_SCHEMA_COLS)

    def test_column_names_are_nonempty_strings(self):
        """Verify that every column constant is a usable pandas column key.

        GIVEN: the raw schema column constants
        WHEN: each one is inspected
        THEN: all are non-empty strings
        """
        assert all(isinstance(col, str) and col for col in RAW_SCHEMA_COLS)


# =====================================================================
#                        FEATURE GROUPS
# =====================================================================

class TestFeatureGroups:

    @pytest.mark.parametrize("group_name", GROUP_NAMES)
    def test_groups_are_immutable_tuples(self, group_name: str):
        """Verify that every column/species group is an immutable tuple.

        GIVEN: any group constant declared in config
        WHEN: its type is inspected
        THEN: it is a tuple (immutability guards against accidental
              in-place edits at runtime)
        """
        assert isinstance(getattr(config, group_name), tuple)

    @pytest.mark.parametrize("group_name", GROUP_NAMES)
    def test_groups_contain_no_duplicates(self, group_name: str):
        """Verify that no group lists the same entry twice.

        GIVEN: any group constant declared in config
        WHEN: its entries are collected into a set
        THEN: the set has the same length as the tuple
        """
        group = getattr(config, group_name)

        assert len(set(group)) == len(group)

    def test_leakage_cols_cover_target_and_subtarget(self):
        """Verify that both target-related columns are flagged as leakage.

        GIVEN: the LEAKAGE_COLS group
        WHEN: compared with the target constants
        THEN: it contains exactly TARGET_COL and SUBTARGET_COL
        """
        assert set(config.LEAKAGE_COLS) == {config.TARGET_COL, config.SUBTARGET_COL}

    @pytest.mark.parametrize(
        "group_name",
        ["LEAKAGE_COLS", "CATEGORICAL_COLS", "FILL_TARGETS",
         "COLUMNS_TO_REMOVE", "CRITICAL_COLS", "ESSENTIAL_COLS"],
    )
    def test_raw_groups_reference_known_schema_columns(self, group_name: str):
        """Verify that raw-data groups only reference declared schema columns.

        GIVEN: any group that addresses raw dataset columns
        WHEN: its entries are compared against the declared schema
        THEN: every entry is a known raw column (a typo or a rename in one
              place but not the other fails here)
        """
        group = getattr(config, group_name)

        assert set(group) <= set(RAW_SCHEMA_COLS)

    def test_essential_cols_exclude_leakage_and_id(self):
        """Verify that model-facing columns never include leakage or the ID.

        GIVEN: the ESSENTIAL_COLS group
        WHEN: intersected with LEAKAGE_COLS and COLUMNS_TO_REMOVE
        THEN: the intersection is empty
        """
        forbidden = set(config.LEAKAGE_COLS) | set(config.COLUMNS_TO_REMOVE)

        assert not set(config.ESSENTIAL_COLS) & forbidden

    def test_scaled_and_encoded_features_are_disjoint(self):
        """Verify that no feature is both scaled and one-hot/ordinal encoded.

        GIVEN: NUM_SCALE_COLS and CAT_ENCODE_COLS
        WHEN: intersected
        THEN: the intersection is empty (a column routed to both encoder
              branches would be duplicated in the model matrix)
        """
        assert not set(config.NUM_SCALE_COLS) & set(config.CAT_ENCODE_COLS)

    def test_categorical_cols_are_all_encoded(self):
        """Verify that every grouped categorical column reaches the encoder.

        GIVEN: CATEGORICAL_COLS (rare-binned by the feature engineering)
        WHEN: compared with CAT_ENCODE_COLS
        THEN: each binned column is also listed for encoding
        """
        assert set(config.CATEGORICAL_COLS) <= set(config.CAT_ENCODE_COLS)


# =====================================================================
#                   MODEL & EXECUTION SETTINGS
# =====================================================================

class TestModelSettings:

    @pytest.mark.parametrize(
        "name", ["DEFAULT_TEST_SIZE", "HOLDOUT_FRACTION", "MAX_OTHER_RATIO"]
    )
    def test_fractions_are_valid_proportions(self, name: str):
        """Verify that every fraction setting is a proportion in (0, 1).

        GIVEN: any fraction constant (test size, hold-out, other-ratio)
        WHEN: its value is inspected
        THEN: it lies strictly between 0 and 1
        """
        value = getattr(config, name)

        assert 0.0 < value < 1.0

    def test_n_splits_allows_cross_validation(self):
        """Verify that the CV setting produces at least a 2-fold split.

        GIVEN: the N_SPLITS constant
        WHEN: its value is inspected
        THEN: it is an integer >= 2 (sklearn's minimum for cross-validation)
        """
        assert isinstance(config.N_SPLITS, int)
        assert config.N_SPLITS >= 2

    def test_random_state_and_max_iter_are_positive_ints(self):
        """Verify that reproducibility and convergence settings are sane.

        GIVEN: RANDOM_STATE and MAX_ITER
        WHEN: their values are inspected
        THEN: both are non-negative integers, with MAX_ITER strictly positive
        """
        assert isinstance(config.RANDOM_STATE, int) and config.RANDOM_STATE >= 0
        assert isinstance(config.MAX_ITER, int) and config.MAX_ITER > 0

    def test_species_are_nonempty_strings(self):
        """Verify that the species tournament roster is well-formed.

        GIVEN: the SPECIES tuple
        WHEN: its entries are inspected
        THEN: it is non-empty and every entry is a non-empty string
        """
        assert config.SPECIES
        assert all(isinstance(s, str) and s for s in config.SPECIES)

    @pytest.mark.parametrize(
        "metric_key, scorer_name", list(config.SCORING.items())
    )
    def test_scoring_entries_resolve_to_sklearn_scorers(
        self, metric_key: str, scorer_name: str
    ):
        """Verify that every scoring entry is a real sklearn scorer name.

        GIVEN: any entry of the SCORING dictionary
        WHEN: resolved through sklearn's scorer registry
        THEN: no exception is raised (a typo like 'f1-macro' fails here
              instead of deep inside GridSearchCV)
        """
        assert get_scorer(scorer_name) is not None

    def test_refit_metric_is_available(self):
        """Verify that the tournament's selection metric is in SCORING.

        GIVEN: the SCORING dictionary
        WHEN: looked up for the refit key used by the training module
        THEN: 'f1_macro' is present
        """
        assert "f1_macro" in config.SCORING


# =====================================================================
#                            PATHS
# =====================================================================

class TestPaths:

    @pytest.mark.parametrize(
        "path_name",
        ["PROJECT_ROOT", "RAW_DATA_PATH", "SPLIT_DATA_DIR",
         "MODELS_DIR", "REPORTS_DIR"],
    )
    def test_paths_are_path_objects(self, path_name: str):
        """Verify that every declared path is a pathlib.Path.

        GIVEN: any path constant declared in config
        WHEN: its type is inspected
        THEN: it is a Path (never a raw string)
        """
        assert isinstance(getattr(config, path_name), Path)

    @pytest.mark.parametrize(
        "path_name",
        ["RAW_DATA_PATH", "SPLIT_DATA_DIR", "MODELS_DIR", "REPORTS_DIR"],
    )
    def test_paths_live_under_project_root(self, path_name: str):
        """Verify that every data/artifact path stays inside the project.

        GIVEN: any non-root path constant
        WHEN: compared against PROJECT_ROOT
        THEN: the path is a descendant of the project root (no accidental
              absolute path escaping the repository)
        """
        path = getattr(config, path_name)

        assert config.PROJECT_ROOT in path.parents

    def test_raw_data_path_points_to_a_csv(self):
        """Verify that the raw data path targets a CSV file.

        GIVEN: the RAW_DATA_PATH constant
        WHEN: its suffix is inspected
        THEN: it ends in '.csv'
        """
        assert config.RAW_DATA_PATH.suffix == ".csv"
