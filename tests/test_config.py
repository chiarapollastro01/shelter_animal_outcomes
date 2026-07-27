"""Unit tests for the configuration module."""

from pathlib import Path

from src import config


def test_config_types_and_paths():
    """Verify type correctness of paths and global configuration constants.

    GIVEN: the src.config module
    WHEN: inspecting paths and data structures
    THEN: paths are Path objects, CSV extension is present, and types are valid
    """
    assert isinstance(config.RAW_DATA_PATH, Path)
    assert isinstance(config.SPLIT_DATA_DIR, Path)
    assert config.RAW_DATA_PATH.suffix == ".csv"

    assert isinstance(config.RANDOM_STATE, int)
    assert isinstance(config.DEFAULT_TEST_SIZE, float)
    assert isinstance(config.MAX_OTHER_RATIO, float)
    assert isinstance(config.N_SPLITS, int)


def test_config_numeric_ranges_sanity():
    """Verify that numeric thresholds and fractions fall within valid mathematical bounds.

    GIVEN: the src.config numerical parameters
    WHEN: validating ranges for split sizes, ratios, and random seeds
    THEN: all values fall strictly within their allowed domain boundaries
    """
    assert 0.0 < config.DEFAULT_TEST_SIZE < 1.0
    assert 0.0 < config.MAX_OTHER_RATIO < 1.0
    assert 0.0 < config.HOLDOUT_FRACTION < 1.0
    assert config.N_SPLITS >= 2
    assert config.RANDOM_STATE >= 0


def test_config_schema_internal_consistency():
    """Verify cross-field logical integrity and schema relationships.

    GIVEN: the schema definitions in src.config
    WHEN: checking relationships between target, leakage, and feature lists
    THEN: target columns are present in leakage tuples, and required features are grouped correctly
    """

    assert config.TARGET_COL in config.LEAKAGE_COLS
    assert config.SUBTARGET_COL in config.LEAKAGE_COLS

    assert config.BREED_COL in config.CATEGORICAL_COLS
    assert config.COLOR_COL in config.CATEGORICAL_COLS

    assert config.BREED_COL in config.CAT_ENCODE_COLS
    assert config.COLOR_COL in config.CAT_ENCODE_COLS

    assert config.ID_COL in config.COLUMNS_TO_REMOVE

    assert len(config.SCORING) > 0
    assert "f1_macro" in config.SCORING

def test_config_feature_columns_disjoint_sets():
    """Verify that numerical features, categorical features, removed columns, and leakage columns are mutually exclusive.

    GIVEN: the column sets defined in src.config
    WHEN: checking intersections between categorical, numerical, removed, and leakage sets
    THEN: all column sets are completely disjoint with zero overlap
    """
    num_cols = set(config.NUM_SCALE_COLS)
    cat_cols = set(config.CAT_ENCODE_COLS)
    remove_cols = set(config.COLUMNS_TO_REMOVE)
    leakage_cols = set(config.LEAKAGE_COLS)

    assert num_cols.isdisjoint(
        cat_cols
    ), f"Overlap found between NUM and CAT: {num_cols & cat_cols}"

    assert remove_cols.isdisjoint(
        num_cols
    ), f"Overlap found between REMOVE and NUM: {remove_cols & num_cols}"
    assert remove_cols.isdisjoint(
        cat_cols
    ), f"Overlap found between REMOVE and CAT: {remove_cols & cat_cols}"

    assert leakage_cols.isdisjoint(
        num_cols
    ), f"Overlap found between LEAKAGE and NUM: {leakage_cols & num_cols}"
    assert leakage_cols.isdisjoint(
        cat_cols
    ), f"Overlap found between LEAKAGE and CAT: {leakage_cols & cat_cols}"