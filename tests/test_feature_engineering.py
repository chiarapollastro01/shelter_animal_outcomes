# pylint: disable=too-many-lines
"""
Unit tests for the feature engineering module.
"""

from functools import partial
import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src import config
from src.feature_engineering import (
    CategoricalFeaturesEngineer,
    NameFeaturesExtractor,
    RareCategoriesGrouper,
    SexFeaturesExtractor,
    TemporalFeaturesExtractor,
    extract_primary_breed,
    extract_primary_color,
    require_columns
)


TWO_PI= 2* np.pi


TRANSFORMER_FACTORIES = [
    pytest.param(TemporalFeaturesExtractor, id="temporal"),
    pytest.param(SexFeaturesExtractor, id="sex"),
    pytest.param(NameFeaturesExtractor, id="name"),
    pytest.param(CategoricalFeaturesEngineer, id="categorical_engineer"),
    pytest.param(
        partial(RareCategoriesGrouper, columns=(config.BREED_COL, config.COLOR_COL)),
        id="rare_grouper",
    ),
]


@pytest.fixture
def full_feature_frame() -> pd.DataFrame:
    """Frame carrying every column any transformer consumes, with custom index."""
    return pd.DataFrame(
        {
            config.DATETIME_COL: ["2026-07-06 12:00:00", "2026-07-11 09:30:00"],
            config.SEX_COL: ["Neutered Male", "Intact Female"],
            config.NAME_COL: ["Bella", None],
            config.BREED_COL: ["Labrador Mix", "Siamese"],
            config.COLOR_COL: ["Black/White", "Orange"],
        },
        index=[10, 20],
    )


@pytest.mark.parametrize("make_transformer", TRANSFORMER_FACTORIES)
def test_fit_returns_self_and_ignores_y(make_transformer, full_feature_frame):
    """Verify the sklearn fit contract for every transformer in the module.

    GIVEN: any unfitted transformer and a frame containing all consumable columns
    WHEN: fit is executed, passing an arbitrary y alongside X
    THEN: the same instance is returned and the extra y is accepted and ignored
    """
    transformer = make_transformer()
    y_dummy = pd.Series(range(len(full_feature_frame)), index=full_feature_frame.index)

    assert transformer.fit(full_feature_frame, y=y_dummy) is transformer


@pytest.mark.parametrize("make_transformer", TRANSFORMER_FACTORIES)
def test_transform_does_not_mutate_input(make_transformer, full_feature_frame):
    """Verify the purity contract for every transformer in the module.

    GIVEN: any transformer and a raw input frame
    WHEN: fit_transform is executed
    THEN: the original input frame remains completely unchanged
    """
    original = full_feature_frame.copy(deep=True)

    make_transformer().fit_transform(full_feature_frame)

    pd.testing.assert_frame_equal(full_feature_frame, original)


@pytest.mark.parametrize(
    "make_transformer, custom_col, sample_value, expected_col",
    [
        pytest.param(
            partial(TemporalFeaturesExtractor, datetime_col="my_date"),
            "my_date",
            "2026-07-06 12:00:00",
            config.IS_WEEKEND_COL,
             id="temporal_custom",
        ),
        pytest.param(
            partial(SexFeaturesExtractor, sex_col="my_sex"),
            "my_sex",
            "Neutered Male",
            config.REPRODUCTIVE_STATUS_COL,
            id="sex_custom",
        ),
        pytest.param(
            partial(NameFeaturesExtractor, name_col="my_name"),
            "my_name",
            "Charles",
            config.HAS_NAME_COL,
            id="name_custom",
        ),
    ],
)
def test_extractors_support_custom_column_names(
    make_transformer, custom_col, sample_value, expected_col
):
    """Verify the drop-source/create-feature contract with non-default column names.

    GIVEN: a stateless extractor configured with a custom source column name
    WHEN: fit_transform is executed
    THEN: the custom source column is dropped and the expected feature is created
    """
    X = pd.DataFrame({custom_col: [sample_value]})
    X_transformed = make_transformer().fit_transform(X)

    assert custom_col not in X_transformed.columns
    assert expected_col in X_transformed.columns


# =====================================================================
#                       REQUIRE COLUMNS TESTS 
# =====================================================================

class TestRequireColumns:
    """Testing the guard every transformer calls before touching a frame."""

    def test_all_columns_present_returns_none(self):
        """Verify that a satisfied requirement is silent.

        GIVEN: a DataFrame carrying every required column
        WHEN: require_columns is executed
        THEN: nothing is raised and nothing is returned, the guard being there
              only to interrupt
        """
        X = pd.DataFrame({"a": [1], "b": [2]})

        assert require_columns(X, ("a", "b"), "nothing works") is None


    def test_every_missing_column_is_listed(self):
        """Verify that the message names all the culprits, not just the first.

        GIVEN: a DataFrame missing two of the three required columns
        WHEN: require_columns is executed
        THEN: both missing names appear in the message, so that a caller fixes
              the frame once instead of discovering the columns one at a time
        """
        X = pd.DataFrame({"a": [1]})

        with pytest.raises(ValueError) as excinfo:
            require_columns(X, ("a", "b", "c"), "the frequent categories cannot be learned")

        message = str(excinfo.value)
        assert "b" in message
        assert "c" in message

    def test_the_message_reads_the_same_whatever_the_count(self):
        """Verify that the wording does not assume a single missing column.

        GIVEN: one frame missing a single column and one missing two
        WHEN: require_columns is executed on both
        THEN: neither message contains a pronoun agreeing with a count, the
              reason being appended after a colon in both cases
        """
        X = pd.DataFrame({"a": [1]})
        reason = "nothing works"

        with pytest.raises(ValueError) as one:
            require_columns(X, ("b",), reason)
        with pytest.raises(ValueError) as two:
            require_columns(X, ("b", "c"), reason)

        assert str(one.value).endswith(f": {reason}.")
        assert str(two.value).endswith(f": {reason}.")


# =====================================================================
#                       TEMPORAL FEATURE TESTS (Stateless)
# =====================================================================
class TestTemporalFeaturesExtractor:
    """Testing the cyclic and weekend features derived from the raw datetime column."""

    def test_temporal_extractor_success(self):
        """Verify that TemporalFeaturesExtractor drops raw datetime and generates cyclic columns.

        GIVEN: a DataFrame containing a raw datetime string column
        WHEN: fit_transform is executed
        THEN: raw datetime column is dropped, expected cyclic columns and weekend indicator are
              added,
          and original index is preserved
        """
        X_train = pd.DataFrame(
         {config.DATETIME_COL: ["2026-07-06 12:00:00", "2026-07-10 12:00:00"]},
         index=[10, 20],
         )
        expected_columns = {
          config.HOUR_SIN_COL,
          config.HOUR_COS_COL,
          config.WDAY_SIN_COL,
          config.WDAY_COS_COL,
          config.DOY_SIN_COL,
          config.DOY_COS_COL,
          config.IS_WEEKEND_COL,
         }

        X_transformed = TemporalFeaturesExtractor().fit_transform(X_train)

        assert config.DATETIME_COL not in X_transformed.columns
        assert set(X_transformed.columns) == expected_columns
        pd.testing.assert_index_equal(X_transformed.index, X_train.index)

    def test_temporal_extractor_is_weekend_logic(self):
        """Verify that weekend indicator correctly identifies weekdays and weekend days.

        GIVEN: a DataFrame with dates representing Monday, Friday, Saturday, and Sunday
        WHEN: fit_transform is executed
        THEN: weekend indicator is assigned 0.0 for weekdays and 1.0 for weekends, preserving custom
              index
        """
        X_train = pd.DataFrame(
            {
            config.DATETIME_COL: [
                "2026-07-06 12:00:00",  # Monday (0) -> 0.0
                "2026-07-10 12:00:00",  # Friday (4) -> 0.0
                "2026-07-11 12:00:00",  # Saturday (5) -> 1.0
                "2026-07-12 12:00:00",  # Sunday (6) -> 1.0
                ]
            },
            index=[10, 20, 30, 40],
        )
        expected_weekend = pd.Series(
            [0.0, 0.0, 1.0, 1.0], index=[10, 20, 30, 40], name=config.IS_WEEKEND_COL
        )

        X_transformed = TemporalFeaturesExtractor().fit_transform(X_train)

        pd.testing.assert_series_equal(
            X_transformed[config.IS_WEEKEND_COL], expected_weekend, check_dtype=False
        )

    def test_temporal_extractor_cyclic_hours(self):
        """Verify the mathematical correctness of sine and cosine transformations for hours.

        GIVEN: a DataFrame with precise timestamps representing Midnight (hour 0), 6 AM (hour 6),
               and Noon (hour 12)
               and a custom non-default index
        WHEN: the transform method of TemporalFeaturesExtractor is executed
        THEN: Hour_sin and Hour_cos calculate accurate values at boundaries (0, pi/2, pi)
              respectively,
              preserving the original index and handling float precision tolerances
        """
        X_train = pd.DataFrame({
            config.DATETIME_COL: [
                "2026-01-01 00:00:00",
                "2026-01-01 06:00:00",
                "2026-01-01 12:00:00",
                ]
            },
            index=[11, 22, 33],
        )
        expected_sin = pd.Series(
            np.sin(TWO_PI * np.array([0, 6, 12]) / 24), index=[11, 22, 33], name=config.HOUR_SIN_COL
        )
        expected_cos = pd.Series(
            np.cos(TWO_PI* np.array([0, 6, 12]) / 24), index=[11, 22, 33], name=config.HOUR_COS_COL
        )

        X_transformed = TemporalFeaturesExtractor().fit_transform(X_train)

        pd.testing.assert_series_equal(
            X_transformed[config.HOUR_SIN_COL], expected_sin, check_exact=False, atol=1e-7
        )
        pd.testing.assert_series_equal(
            X_transformed[config.HOUR_COS_COL], expected_cos, check_exact=False, atol=1e-7
        )


    def test_temporal_extractor_cyclic_weekdays(self):
        """Verify the mathematical correctness of sine and cosine transformations for weekdays.

        GIVEN: a DataFrame containing a Monday (weekday 0) and a Sunday (weekday 6) with custom
               non-default indices
        WHEN: the transform method of TemporalFeaturesExtractor is executed
        THEN: Wday_sin and Wday_cos are computed accurately based on the weekday, preserving the
              index
              and handling float precision tolerances
        """
        X_train = pd.DataFrame({config.DATETIME_COL: ["2026-07-06", "2026-07-12"]}, index=[15, 25])
        expected_sin = pd.Series(
            np.sin(TWO_PI * np.array([0, 6]) / 7), index=[15, 25], name=config.WDAY_SIN_COL
        )
        expected_cos = pd.Series(
            np.cos(TWO_PI * np.array([0, 6]) / 7), index=[15, 25], name=config.WDAY_COS_COL
        )

        X_transformed = TemporalFeaturesExtractor().fit_transform(X_train)

        pd.testing.assert_series_equal(
            X_transformed[config.WDAY_SIN_COL], expected_sin, check_exact=False, atol=1e-7
        )
        pd.testing.assert_series_equal(
            X_transformed[config.WDAY_COS_COL], expected_cos, check_exact=False, atol=1e-7
        )


    def test_temporal_extractor_cyclic_day_of_year(self):
        """Verify the mathematical correctness of sine and cosine transformations for the day of the
           year.

        GIVEN: a DataFrame containing dates representing Day of Year 1 (Jan 1) and 100 (Apr 10) in a
               non-leap year (2026),
           with custom non-default indices
        WHEN: the transform method of TemporalFeaturesExtractor is executed
        THEN: DoY_sin and DoY_cos columns are computed accurately based on the day of the year,
              preserving the index
              and handling float precision tolerances
        """
        X_train = pd.DataFrame({config.DATETIME_COL: ["2026-01-01", "2026-04-10"]}, index=[11, 22])
        expected_sin = pd.Series(
            np.sin(TWO_PI * np.array([1, 100]) / 365.25), index=[11, 22], name=config.DOY_SIN_COL
        )
        expected_cos = pd.Series(
            np.cos(TWO_PI * np.array([1, 100]) / 365.25), index=[11, 22], name=config.DOY_COS_COL
        )

        X_transformed = TemporalFeaturesExtractor().fit_transform(X_train)

        pd.testing.assert_series_equal(
            X_transformed[config.DOY_SIN_COL], expected_sin, check_exact=False, atol=1e-7
        )
        pd.testing.assert_series_equal(
            X_transformed[config.DOY_COS_COL], expected_cos, check_exact=False, atol=1e-7
        )


    def test_temporal_extractor_already_datetime_type(self):
        """Verify that the transformer produces identical values regardless of whether the input is
           raw string or datetime64.

        GIVEN: two identical DataFrames, one with raw string dates and one pre-converted to
               datetime64[ns]
        WHEN: transform is executed on both
        THEN: both executions succeed, producing perfectly identical DataFrames in both schema and
              values,
              preserving the index and dropping datetime column
        """
        X_strings = pd.DataFrame({config.DATETIME_COL: ["2026-07-06 12:00:00"]}, index=[99])
        X_datetime = pd.DataFrame(
            {config.DATETIME_COL: pd.to_datetime(["2026-07-06 12:00:00"])}, index=[99]
        )

        res_strings = TemporalFeaturesExtractor().fit_transform(X_strings)
        res_datetime = TemporalFeaturesExtractor().fit_transform(X_datetime)

        pd.testing.assert_frame_equal(res_strings, res_datetime)

    def test_temporal_extractor_invalid_inputs(self):
        """Verify safe fallback to NaN when encountering completely invalid date strings.

        GIVEN: a DataFrame containing an invalid/unparseable date string
        WHEN: transform is executed on TemporalFeaturesExtractor
        THEN: execution succeeds without raising exceptions, resulting in NaN values for cyclic
              features
        """
        X_train = pd.DataFrame({config.DATETIME_COL: ["invalid_date_string"]}, index=[1])

        X_transformed = TemporalFeaturesExtractor().fit_transform(X_train)

        assert np.isnan(X_transformed[config.HOUR_SIN_COL].iloc[0])
        assert np.isnan(X_transformed[config.IS_WEEKEND_COL].iloc[0])


    def test_temporal_extractor_empty_dataframe_with_columns(self):
        """Verify that an empty DataFrame with the target column is processed safely without
           crashing.

        GIVEN: an empty DataFrame with only the datetime column in its schema
        WHEN: transform is executed on TemporalFeaturesExtractor
        THEN: the returned DataFrame is empty, datetime column is dropped, and the expected empty
              schema is preserved
        """
        X_train = pd.DataFrame(columns=(config.DATETIME_COL,))

        expected_cols = {
        config.IS_WEEKEND_COL,
        config.HOUR_SIN_COL,
        config.HOUR_COS_COL,
        config.WDAY_SIN_COL,
        config.WDAY_COS_COL,
        config.DOY_SIN_COL,
        config.DOY_COS_COL,
    }

        X_transformed = TemporalFeaturesExtractor().fit_transform(X_train)

        assert X_transformed.empty
        assert set(X_transformed.columns) == expected_cols

    def test_temporal_extractor_handles_null_values(self):
        """Verify that missing or invalid date values do not crash the execution and propagate as
           NaN.

        GIVEN: a DataFrame with valid datetime values and a None value, under custom indices
        WHEN: transform is executed on TemporalFeaturesExtractor
        THEN: the valid date is calculated, and the missing value safely propagates as NaN in the
              cyclic columns
        """
        X_train = pd.DataFrame(
            {config.DATETIME_COL: ["2026-07-12 12:00:00", None]}, index=[101, 102]
        )

        X_transformed = TemporalFeaturesExtractor().fit_transform(X_train)

        assert not np.isnan(X_transformed[config.HOUR_SIN_COL].loc[101])

        assert np.isnan(X_transformed[config.HOUR_SIN_COL].loc[102])
        assert np.isnan(X_transformed[config.WDAY_COS_COL].loc[102])
        assert np.isnan(X_transformed[config.DOY_SIN_COL].loc[102])

        assert X_transformed[config.IS_WEEKEND_COL].loc[101] == 1.0
        assert np.isnan(X_transformed[config.IS_WEEKEND_COL].loc[102])


    def test_temporal_extractor_missing_column(self):
        """Verify that a missing datetime column is an error.

        GIVEN: a DataFrame that does not carry the datetime column
        WHEN: transform is executed
        THEN: a ValueError names it
        """
        X = pd.DataFrame({config.NAME_COL: ["Bella"]})

        with pytest.raises(ValueError, match=config.DATETIME_COL):
            TemporalFeaturesExtractor().transform(X)

    def test_temporal_extractor_all_null(self):
        """Verify processing when the datetime column consists entirely of NaN values.

        GIVEN: a DataFrame where all entries in DateTime are NaN
        WHEN: fit_transform is executed on TemporalFeaturesExtractor
        THEN: all extracted cyclic features and weekend indicator is safely resolve to NaN,
              preserving the index
        """
        X_train = pd.DataFrame({config.DATETIME_COL: [np.nan, np.nan]}, index=[10, 20])

        X_transformed = TemporalFeaturesExtractor().fit_transform(X_train)

        assert X_transformed[config.HOUR_SIN_COL].isna().all()
        assert X_transformed[config.IS_WEEKEND_COL].isna().all()


# =====================================================================
#                           SEX FEATURE TESTS
# =====================================================================

class TestSexFeaturesExtractor:
    """Testing the reproductive status and sex features derived from the raw sex column."""

    def test_sex_extractor_success(self):
        """Verify that SexFeaturesExtractor successfully extracts the reproductive status column and
           drops the raw column.

        GIVEN: a DataFrame with raw sex values containing neutered,
           spayed, intact, unknown, and NaN entries, with a custom index
        WHEN: transform is executed on SexFeaturesExtractor
        THEN: the raw column is dropped and the reproductive status
          is accurately mapped, preserving the original index
        """
        X_train = pd.DataFrame({
            config.SEX_COL: ["Neutered Male", "Spayed Female", "Intact Male", "Unknown", np.nan]
        }, index=[10, 20, 30, 40, 50])


        expected = pd.Series(
            ["Neutered/Spayed", "Neutered/Spayed", "Intact", "Unknown", "Unknown"],
            index=[10, 20, 30, 40, 50],
            name=config.REPRODUCTIVE_STATUS_COL
        )

        X_transformed = SexFeaturesExtractor().fit_transform(X_train)

        assert config.SEX_COL not in X_transformed.columns
        pd.testing.assert_series_equal(X_transformed[config.REPRODUCTIVE_STATUS_COL], expected)

    def test_sex_extractor_case_insensitivity(self):
        """Verify robust parsing of lower/uppercase variations in sex strings.

        GIVEN: a DataFrame with sex strings in mixed casing
        WHEN: transform is executed on SexFeaturesExtractor
        THEN: categories are mapped correctly regardless of letter casing
        """
        X_train = pd.DataFrame(
            {config.SEX_COL: ["spayed female", "NEUTERED MALE", "  intact male  "]}, index=[1, 2, 3]
        )
        expected_status = pd.Series(
            ["Neutered/Spayed", "Neutered/Spayed", "Intact"],
            index=[1, 2, 3],
            name=config.REPRODUCTIVE_STATUS_COL,
        )

        X_transformed = SexFeaturesExtractor().fit_transform(X_train)

        pd.testing.assert_series_equal(
            X_transformed[config.REPRODUCTIVE_STATUS_COL], expected_status
        )

    def test_sex_extractor_invalid_inputs(self):
        """Verify safe fallback to 'Unknown' when encountering invalid sex strings.

        GIVEN: a DataFrame containing 'Unknown', and completely unrecognized text ('5 units',
               'Random')
        WHEN: fit_transform is executed on SexFeaturesExtractor
        THEN: all unrecognized inputs safely resolve to 'Unknown' without breaking execution
        """
        X_train = pd.DataFrame(
            {config.SEX_COL: ["Unknown", "5 units", "Random Text"]},
            index=[2, 3, 4],
        )
        expected = pd.Series(
             ["Unknown", "Unknown", "Unknown"],
            index=[2, 3, 4],
            name=config.REPRODUCTIVE_STATUS_COL,
        )

        X_transformed = SexFeaturesExtractor().fit_transform(X_train)

        pd.testing.assert_series_equal(
            X_transformed[config.REPRODUCTIVE_STATUS_COL], expected
        )

    def test_sex_extractor_empty_dataframe_with_columns(self):
        """Verify that an empty DataFrame with the target sex column is processed safely.

        GIVEN: an empty DataFrame with only the sex column in its schema
        WHEN: transform is executed on SexFeaturesExtractor
        THEN: the returned DataFrame is empty, sex is dropped, and the reproductive status feature
              is created
        """
        X_train = pd.DataFrame(columns=(config.SEX_COL,))

        X_transformed = SexFeaturesExtractor().fit_transform(X_train)

        assert X_transformed.empty
        assert set(X_transformed.columns) == {config.REPRODUCTIVE_STATUS_COL}


    def test_sex_extractor_handles_null_values(self):
        """Verify that missing (NaN) sex values map safely to 'Unknown'.

        GIVEN: a DataFrame containing a valid sex string and a NaN entry under custom indices
        WHEN: transform is executed on SexFeaturesExtractor
        THEN: NaN entries resolve to 'Unknown' without raising exceptions
        """
        X_train = pd.DataFrame({config.SEX_COL: ["Intact Male", np.nan]}, index=[101, 102])

        X_transformed = SexFeaturesExtractor().fit_transform(X_train)

        assert X_transformed.loc[101, config.REPRODUCTIVE_STATUS_COL] == "Intact"
        assert X_transformed.loc[102, config.REPRODUCTIVE_STATUS_COL] == "Unknown"


    def test_sex_extractor_missing_column(self):
        """Verify that a missing sex column is an error.

        GIVEN: a DataFrame that does not carry the sex column
        WHEN: transform is executed
        THEN: a ValueError names it
        """
        X = pd.DataFrame({config.NAME_COL: ["Bella"]})

        with pytest.raises(ValueError, match=config.SEX_COL):
            SexFeaturesExtractor().transform(X)

    def test_sex_extractor_all_null(self):
        """Verify processing when the sex column consists entirely of NaN values.

        GIVEN: a DataFrame where all entries in sex are NaN
        WHEN: fit_transform is executed on SexFeaturesExtractor
        THEN: the reproductive status feature resolves to 'Unknown' for all rows, preserving the
              index
        """
        X_train = pd.DataFrame({config.SEX_COL: [np.nan, np.nan]}, index=[10, 20])
        expected = pd.Series(
            ["Unknown", "Unknown"], index=[10, 20], name=config.REPRODUCTIVE_STATUS_COL
        )

        X_transformed = SexFeaturesExtractor().fit_transform(X_train)
        pd.testing.assert_series_equal(X_transformed[config.REPRODUCTIVE_STATUS_COL], expected)


# =====================================================================
#                         NAME FEATURE TESTS
# =====================================================================
class TestNameFeaturesExtractor:
    """Testing the has-a-name indicator derived from the raw name column."""

    def test_name_extractor_success(self):
        """Verify that NameFeaturesExtractor correctly creates the name indicator and drops the raw
           column.

        GIVEN: a DataFrame with standard valid names and the placeholder "Unknown",
               with custom non-default indices
        WHEN: fit_transform is executed on NameFeaturesExtractor
        THEN: the raw name column is dropped, name indicator is 1 for valid names and 0 for
              'Unknown',
              and the original index is preserved
        """
        X_train = pd.DataFrame(
            {config.NAME_COL: ["Bella", "Max", "Unknown"]},
            index=[10, 20, 30],
        )
        expected = pd.Series(
            [1, 1, 0], index=[10, 20, 30], name=config.HAS_NAME_COL
        )

        X_transformed = NameFeaturesExtractor().fit_transform(X_train)

        assert config.NAME_COL not in X_transformed.columns
        pd.testing.assert_series_equal(
             X_transformed[config.HAS_NAME_COL], expected, check_dtype=False
        )

    def test_name_extractor_case_insensitivity_and_formatting(self):
        """Verify robust parsing of whitespace variations and case-insensitivity for 'Unknown'.

        GIVEN: a DataFrame with 'Unknown' in various casings and extra surrounding spaces
        WHEN: fit_transform is executed on NameFeaturesExtractor
        THEN: the name indicator resolves correctly to 0 for unknown/empty variations and 1 for
              valid names
        """
        X_train = pd.DataFrame(
             {config.NAME_COL: ["  Bella  ", "unknown", "  UNKNOWN  ", "   "]},
            index=[1, 2, 3, 4],
        )
        expected = pd.Series([1, 0, 0, 0], index=[1, 2, 3, 4], name=config.HAS_NAME_COL)

        X_transformed = NameFeaturesExtractor().fit_transform(X_train)

        pd.testing.assert_series_equal(
             X_transformed[config.HAS_NAME_COL], expected, check_dtype=False
        )

    def test_name_extractor_empty_dataframe_with_columns(self):
        """Verify that an empty DataFrame with the target name column is processed safely.

        GIVEN: an empty DataFrame with only the name column in its schema
        WHEN: transform is executed on NameFeaturesExtractor
        THEN: the returned DataFrame is empty, name is dropped, and name indicator is created
        """
        X_train = pd.DataFrame(columns=(config.NAME_COL,))

        X_transformed = NameFeaturesExtractor().fit_transform(X_train)

        assert X_transformed.empty
        assert list(X_transformed.columns) == [config.HAS_NAME_COL]


    def test_name_extractor_handles_null_values(self):
        """Verify that missing (NaN/None) name values safely resolve to name indicator = 0.

        GIVEN: a DataFrame containing a valid name and NaN/None entries under custom indices
        WHEN: fit_transform is executed on NameFeaturesExtractor
        THEN: NaN/None entries map to 0 in the name indicator without raising exceptions
        """
        X_train = pd.DataFrame({config.NAME_COL: ["Max", np.nan, None]}, index=[101, 102, 103])
        expected = pd.Series([1, 0, 0], index=[101, 102, 103], name=config.HAS_NAME_COL)

        X_transformed = NameFeaturesExtractor().fit_transform(X_train)

        pd.testing.assert_series_equal(
             X_transformed[config.HAS_NAME_COL], expected, check_dtype=False
        )

    def test_name_extractor_missing_column(self):
        """Verify that a missing name column is an error.

        GIVEN: a DataFrame that does not carry the name column
        WHEN: transform is executed
        THEN: a ValueError names it
        """
        X = pd.DataFrame({config.SEX_COL: ["Neutered Male"]})

        with pytest.raises(ValueError, match=config.NAME_COL):
            NameFeaturesExtractor().transform(X)

    def test_name_extractor_all_null(self):
        """Verify processing when the Name column consists entirely of NaN values.

        GIVEN: a DataFrame where all entries in the name column are NaN
        WHEN: fit_transform is executed on NameFeaturesExtractor
        THEN: the name indicator is set to 0 for all rows, preserving the original index
        """
        X_train = pd.DataFrame({config.NAME_COL: [np.nan, np.nan]}, index=[10, 20])
        expected = pd.Series([0, 0], index=[10, 20], name="has_name")

        X_transformed = NameFeaturesExtractor().fit_transform(X_train)

        pd.testing.assert_series_equal(
            X_transformed[config.HAS_NAME_COL], expected, check_dtype=False
        )

# =====================================================================
#                  COLOR AND BREED FEATURES TESTS
# =====================================================================
# =====================================================================
#                     HELPER FUNCTIONS TESTS
# =====================================================================

# ----------------- EXTRACT PRIMARY COLOR -----------------
class TestHelperFunctionColor:
    """Testing extract_primary_color, the helper isolating the first color in a slash-separated
       list."""

    def test_extract_primary_color_logic(self):
        """Verify that only the primary color is extracted by splitting on '/'.

        GIVEN: a Series containing bicolor entries ('Black/White'), tricolor entries
               ('Brown Tabby/White'), and solid colors ('Blue') with custom indices
        WHEN: extract_primary_color is executed
        THEN: only the first color before the slash is kept, removing trailing
              spaces, preserving the original index
        """

        color_series = pd.Series(
            ["Black/White", "Brown Tabby/White", "Blue"], index=[10, 20, 30]
        )
        expected = pd.Series(["Black", "Brown Tabby", "Blue"], index=[10, 20, 30])

        result = extract_primary_color(color_series)

        pd.testing.assert_series_equal(result, expected)

    def test_extract_primary_color_handles_nulls(self):
        """Verify primary color extraction on a Series containing both valid strings and NaN
           entries.

        GIVEN: a Series with a valid color string ('Black/White') and a missing value (NaN)
        WHEN: extract_primary_color is executed
        THEN: the valid string is parsed correctly and the NaN entry remains NaN, preserving index
        """
        color_series = pd.Series(["Black/White", np.nan], index=[10, 20])
        expected = pd.Series(["Black", np.nan], index=[10, 20])

        result = extract_primary_color(color_series)

        pd.testing.assert_series_equal(result, expected)


    def test_extract_primary_color_all_null_series(self):
        """Verify that a Series composed entirely of NaNs is handled safely.

        GIVEN: a Series containing only NaN values with custom non-default indices
        WHEN: extract_primary_color is executed
        THEN: all elements remain NaN, preserving the original length and custom index
              (returned unchanged, since a float64 Series of NaN has no .str accessor to work with)
        """
        nan_series = pd.Series([np.nan, np.nan], index=[10, 20])
        expected = pd.Series([np.nan, np.nan], index=[10, 20])

        result = extract_primary_color(nan_series)
        pd.testing.assert_series_equal(result, expected)


    def test_extract_primary_color_empty_series(self):
        """Verify behavior when processing an empty pandas Series.

        GIVEN: a completely empty pandas Series (length 0)
        WHEN: extract_primary_color is executed
        THEN: an empty Series is returned without errors
        """
        empty_series = pd.Series([], dtype=object)

        result = extract_primary_color(empty_series)

        pd.testing.assert_series_equal(result, empty_series)


    def test_extract_primary_color_formatting_variations(self):
        """Verify primary color extraction with multiple slashes, extra spaces, and single colors.

        GIVEN: a Series containing multiple slashes ('Black/White/Brown'), surrounding spaces, and
               single colors
        WHEN: extract_primary_color is executed
        THEN: only the first color component is extracted and trimmed, preserving the index
        """
        color_series = pd.Series(
            ["  Black / White / Tan  ", "Blue", " Red / Yellow "],
            index=[10, 20, 30],
        )
        expected = pd.Series(["Black", "Blue", "Red"], index=[10, 20, 30])

        result = extract_primary_color(color_series)

        pd.testing.assert_series_equal(result, expected)

# ----------------- EXTRACT PRIMARY BREED -----------------

class TestHelperFunctionBreed:
    """Testing extract_primary_breed, the helper isolating the first breed and stripping the 'Mix'
       suffix."""

    def test_extract_primary_breed_logic(self):
        """Verify that the primary breed is correctly extracted by splitting '/' and stripping
           'Mix'.

        GIVEN: a Series containing crossbreeds with slashes, 'Mix' suffixes, and purebreds,
           with custom indices
        WHEN: extract_primary_breed is executed
        THEN: the 'Mix' keyword and second breeds are removed, leaving only the
              clean primary breed, preserving the index
        """

        breed_series = pd.Series(
            [
            "Labrador Retriever/German Shepherd",
            "Chihuahua Shorthair Mix",
            "Siamese",
            ],
            index=[15, 25, 35],
        )
        expected = pd.Series(
            ["Labrador Retriever", "Chihuahua Shorthair", "Siamese"],
            index=[15, 25, 35],
        )

        result = extract_primary_breed(breed_series)

        pd.testing.assert_series_equal(result, expected)

    def test_extract_primary_breed_handles_nulls(self):
        """Verify primary breed extraction on a Series containing both valid strings and NaN
           entries.

        GIVEN: a Series with a valid breed string ('Chihuahua Mix') and a missing value (NaN)
        WHEN: extract_primary_breed is executed
        THEN: the valid breed is parsed correctly and the NaN entry remains NaN, preserving index
        """
        breed_series = pd.Series(["Chihuahua Mix", np.nan], index=[101, 102])
        expected = pd.Series(["Chihuahua", np.nan], index=[101, 102])

        result = extract_primary_breed(breed_series)

        pd.testing.assert_series_equal(result, expected)

    def test_extract_primary_breed_all_null_series(self):
        """Verify that a Series composed entirely of NaNs is handled safely.

        GIVEN: a Series containing only NaN values with custom non-default indices
        WHEN: extract_primary_breed is executed
        THEN: all elements remain NaN, preserving the original length and custom index
            (returned unchanged, since a float64 Series of NaN has no .str accessor to work with)
        """
        nan_series = pd.Series([np.nan, np.nan], index=[101, 102])
        expected = pd.Series([np.nan, np.nan], index=[101, 102])

        result = extract_primary_breed(nan_series)

        pd.testing.assert_series_equal(result, expected)


    def test_extract_primary_breed_empty_series(self):
        """Verify behavior when processing an empty pandas Series.

        GIVEN: a completely empty pandas Series (length 0)
        WHEN: extract_primary_breed is executed
        THEN: an empty Series is returned without errors
        """
        empty_series = pd.Series([], dtype=object)

        result = extract_primary_breed(empty_series)

        pd.testing.assert_series_equal(result, empty_series)


    def test_extract_primary_breed_formatting_variations(self):
        """Verify primary breed extraction with case variations of 'mix', multiple slashes, and
           clean breeds.

        GIVEN: a Series with 'mix' in different casing ('MIX', 'mix'), multiple slashes, and pure
               breeds
        WHEN: extract_primary_breed is executed
        THEN: trailing 'mix' keywords and secondary breeds are stripped regardless of casing,
              preserving the index
        """
        breed_series = pd.Series(
            [
            "Chihuahua MIX",
            "Labrador Retriever mix",
            "German Shepherd/Siberian Husky/Poodle",
            "Beagle",
            ],
            index=[10, 20, 30, 40],
        )
        expected = pd.Series(
            ["Chihuahua", "Labrador Retriever", "German Shepherd", "Beagle"],
            index=[10, 20, 30, 40],
        )
        result = extract_primary_breed(breed_series)

        pd.testing.assert_series_equal(result, expected)



# =====================================================================
#                      RareCategoriesGrouper TESTS
# =====================================================================

# -------------------- FIT TESTS --------------------

class TestRareCategoriesGrouperFit:
    """Testing the frequent categories RareCategoriesGrouper learns from the training frame."""

    def test_rare_categories_grouper_learns_frequent_categories(self):
        """Verify that fit learns frequent categories above threshold.

        GIVEN: a DataFrame with categorical data where 'A' is frequent and 'C' is rare
        WHEN: fit is executed with max_other_ratio=0.25
        THEN: frequent categories are learned and stored in frequent_categories_
        """
        X_train = pd.DataFrame(
            {config.BREED_COL: ["A", "A", "B", "B", "C"]},
        )
        grouper = RareCategoriesGrouper(columns=(config.BREED_COL,), max_other_ratio=0.25)
        grouper.fit(X_train)

        assert set(grouper.frequent_categories_[config.BREED_COL]) == {"A", "B"}

    def test_rare_categories_grouper_fit_raises_on_missing_column(self):
        """Verify that RareCategoriesGrouper raises a ValueError if a required column is missing
           during fit.

        GIVEN: a DataFrame missing a required column specified in the grouper's target configuration
        WHEN: the fit method is executed
        THEN: it must immediately raise a ValueError indicating the required column is missing
        """
        X_train = pd.DataFrame({config.COLOR_COL: ["Black", "White"]})
        grouper = RareCategoriesGrouper(columns=(config.BREED_COL, config.COLOR_COL))

        with pytest.raises(ValueError, match=config.BREED_COL):
            grouper.fit(X_train)

    def test_rare_categories_grouper_fit_learns_nothing_from_an_empty_column(
        self, caplog: pytest.LogCaptureFixture
    ):
        """Verify that RareCategoriesGrouper handles completely empty/NaN columns during fit safely
           with a warning.

        GIVEN: a training DataFrame containing only missing values (NaN) in the target column
        WHEN: the fit method is executed with WARNING logging level captured
        THEN: a warning log is recorded, execution doesn't crash, and frequent categories is set to
              empty tuple
        """
        X_train = pd.DataFrame({config.BREED_COL: [np.nan, np.nan]})
        grouper = RareCategoriesGrouper(columns=(config.BREED_COL,))

        with caplog.at_level(logging.WARNING):
            grouper.fit(X_train)

        assert "is empty or contains only NaNs" in caplog.text
        assert grouper.frequent_categories_ == {config.BREED_COL: ()}

    def test_rare_categories_grouper_fit_empty_dataframe(self):
        """Verify that a frame with no rows is fitted without crashing.

        GIVEN: a training DataFrame carrying the declared columns but no rows
        WHEN: fit is executed
        THEN: an empty set of frequent categories is learned for each column,
              which is what a degenerate cross-validation fold would produce
        """
        X_train = pd.DataFrame(columns=(config.BREED_COL, config.COLOR_COL))
        grouper = RareCategoriesGrouper(
            columns=(config.BREED_COL, config.COLOR_COL)
        )

        grouper.fit(X_train)

        assert grouper.frequent_categories_ == {config.BREED_COL: (), config.COLOR_COL: ()}

# -------------------- TRANSFORM TESTS --------------------

class TestRareCategoriesGrouperTransform:
    """Testing the collapsing of infrequent categories into 'Other' once RareCategoriesGrouper is
       fitted."""

    def test_rare_categories_grouper_raises_not_fitted_error_when_unfitted(self):
        """Verify that RareCategoriesGrouper raises a NotFittedError if transform is called before
           fit.

        GIVEN: a RareCategoriesGrouper instance that has not been fitted yet
        WHEN: the transform method is executed on a DataFrame
        THEN: it must immediately raise a NotFittedError indicating the instance is not fitted
        """
        X_train = pd.DataFrame({config.BREED_COL: ["Labrador Retriever", "Poodle"]})
        grouper = RareCategoriesGrouper(columns=(config.BREED_COL,))

        with pytest.raises(
            NotFittedError,
            match="RareCategoriesGrouper instance is not fitted",
        ):
            grouper.transform(X_train)

    def test_rare_categories_grouper_transform_raises_on_missing_column(self):
        """Verify that a fitted column absent from the data is an error.

        GIVEN: a grouper fitted on two columns
        WHEN: transform is called on a frame carrying only one of them
        THEN: a ValueError is raised, since silently skipping the column
              would let unbinned categories reach the encoder
        """
        X = pd.DataFrame(
            {config.BREED_COL: ["Lab"] * 8 + ["Poodle"] * 2,
             config.COLOR_COL: ["Black"] * 8 + ["Red"] * 2}
        )
        grouper = RareCategoriesGrouper(
            columns=(config.BREED_COL, config.COLOR_COL)
        ).fit(X)

        with pytest.raises(ValueError, match=config.COLOR_COL):
            grouper.transform(X[[config.BREED_COL]])

    def test_rare_categories_grouper_success(self):
        """Verify that categories are dynamically preserved to keep 'Other' below the max ratio.

        GIVEN: a fitted RareCategoriesGrouper instance and a DataFrame with frequent and rare
               categories
        WHEN: transform is executed
        THEN: frequent categories are preserved and rare categories are replaced with 'Other'
        """
        X_train = pd.DataFrame(
            {config.BREED_COL: ["A", "A", "B", "B", "C"]}, index=[10, 20, 30, 40, 50]
        )
        expected = pd.Series(
            ["A", "A", "B", "B", "Other"], index=[10, 20, 30, 40, 50], name=config.BREED_COL
        )

        grouper = RareCategoriesGrouper(columns=(config.BREED_COL,), max_other_ratio=0.25)
        grouper.fit(X_train)
        X_clean = grouper.transform(X_train)

        pd.testing.assert_series_equal(X_clean[config.BREED_COL], expected)

    def test_rare_categories_grouper_preserves_nans(self):
        """Verify that NaN values are untouched and not converted to 'Other' during transform.

        GIVEN: a RareCategoriesGrouper with pre-learned frequent categories ('A' and 'B') and input
               with NaNs
        WHEN: transform is executed
        THEN: NaN values remain NaN, preserving the index
        """
        X_train = pd.DataFrame(
            {config.BREED_COL: ["A", "A", "B", "B", np.nan]}, index=[10, 20, 30, 40, 50]
        )
        expected = pd.Series(
            ["A", "A", "B", "B", np.nan], index=[10, 20, 30, 40, 50], name=config.BREED_COL
        )

        grouper = RareCategoriesGrouper(columns=(config.BREED_COL,), max_other_ratio=0.25)
        grouper.fit(X_train)
        X_clean = grouper.transform(X_train)

        pd.testing.assert_series_equal(X_clean[config.BREED_COL], expected)

    def test_rare_categories_grouper_preserves_index_and_non_target_columns(self):
        """Verify that transform preserves original row indices and untouched columns.

        GIVEN: a fitted grouper and a DataFrame with target and non-target columns and custom
               indices
        WHEN: transform is executed
        THEN: the original index and non-target columns remain completely unchanged
        """
        X_train = pd.DataFrame(
            {config.BREED_COL: ["Labrador"] * 8 + ["Chihuahua"] * 2, "Other_Col": range(10)},
            index=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        )
        grouper = RareCategoriesGrouper(
            columns=(config.BREED_COL,), max_other_ratio=0.25
        ).fit(X_train)

        X_clean = grouper.transform(X_train)

        pd.testing.assert_index_equal(X_clean.index, X_train.index)
        pd.testing.assert_series_equal(X_clean["Other_Col"], X_train["Other_Col"])

    def test_rare_categories_grouper_logs_debug_on_drift(self, caplog: pytest.LogCaptureFixture):
        """Verify that transform emits a DEBUG log when 'Other' proportion exceeds max ratio due to
           data drift.

        GIVEN: a trained grouper and a test dataset containing only rare categories
        WHEN: transform is executed on the test dataset
        THEN: a DEBUG log is emitted indicating the ratio exceeds threshold
        """
        X_train = pd.DataFrame({config.BREED_COL: ["A"] * 90 + ["B"] * 10})
        X_test = pd.DataFrame({config.BREED_COL: ["B"] * 100})

        grouper = RareCategoriesGrouper(columns=(config.BREED_COL,), max_other_ratio=0.15)
        grouper.fit(X_train)

        with caplog.at_level(logging.DEBUG):
            grouper.transform(X_test)

        assert "exceeds the configured max_other_ratio" in caplog.text

    def test_rare_categories_grouper_empty_dataframe(self):
        """Verify that transform returns an empty DataFrame untouched if the input is empty.

        GIVEN: a properly fitted RareCategoriesGrouper instance and a completely empty test
               DataFrame schema
        WHEN: fit and transform methods are executed
        THEN: the input DataFrame is returned completely unmodified, preserving its empty structure
        """
        X_train = pd.DataFrame({config.BREED_COL: ["A", "B"]})
        X_empty = pd.DataFrame(columns=(config.BREED_COL,))

        grouper = RareCategoriesGrouper(columns=(config.BREED_COL,))
        grouper.fit(X_train)

        result = grouper.transform(X_empty)
        assert result.empty
        assert list(result.columns) == [config.BREED_COL]

# -------------------- END TO END, CUSTOM & EDGE CASES --------------------

class TestRareCategoriesGrouperCustomAndE2E:
    """Testing RareCategoriesGrouper end to end, including leakage prevention and degenerate
       frames."""

    def test_rare_categories_grouper_all_nan(self):
        """Verify robust handling of a DataFrame column composed entirely of missing entries.

        GIVEN: a DataFrame where all elements in target column are missing (NaN)
        WHEN: fit and transform are executed
        THEN: all NaN values remain NaN and no exception is raised
        """
        X_mock = pd.DataFrame({config.BREED_COL: [np.nan, np.nan]}, index=[10, 20])

        grouper = RareCategoriesGrouper(columns=(config.BREED_COL,))
        X_clean = grouper.fit_transform(X_mock)

        pd.testing.assert_frame_equal(X_clean, X_mock, check_dtype=False)

    def test_rare_categories_grouper_prevents_data_leakage(self):
        """Verify that test set grouping strictly uses frequent categories learned during training.

        GIVEN: a grouper trained on X_train with breed and color columns and a test set with unseen
               labels
        WHEN: transform is called on the test set
        THEN: rare/unseen labels are mapped to 'Other' based solely on train statistics
        """
        X_train = pd.DataFrame({
            config.BREED_COL: ["Labrador"] * 8 + ["Poodle"] * 2,
            config.COLOR_COL: ["Black"] * 10,
            })
        X_test = pd.DataFrame({
            config.BREED_COL: ["Beagle", "Chihuahua"],
            config.COLOR_COL: ["Black", "Black"],
        })
        grouper = RareCategoriesGrouper(
            columns=(config.BREED_COL, config.COLOR_COL,), max_other_ratio=0.25
        ).fit(X_train)
        X_test_clean = grouper.transform(X_test)

        expected_breed = pd.Series(["Other", "Other"], name=config.BREED_COL)
        pd.testing.assert_series_equal(X_test_clean[config.BREED_COL], expected_breed)

        expected_color = pd.Series(["Black", "Black"], name=config.COLOR_COL)
        pd.testing.assert_series_equal(X_test_clean[config.COLOR_COL], expected_color)

    @pytest.mark.parametrize(
    "max_ratio, expected_frequent",
    [
            (0.0, {"A", "B", "C"}),
            (0.25, {"A", "B"}),
            (0.50, {"A"}),
        ],
    )
    def test_rare_categories_grouper_custom_max_other_ratio(
        self, max_ratio, expected_frequent
    ):
        """Verify that custom max_other_ratio values dynamically alter the learned frequent
           categories.

        GIVEN: a DataFrame with unbalanced categorical frequencies (A: 60%, B: 20%, C: 20%)
        WHEN: fit is executed with different max_other_ratio thresholds
        THEN: frequent_categories_ correctly adapts to the configured ratio threshold
        """
        X_train = pd.DataFrame(
            {config.BREED_COL: ["A"] * 6 + ["B"] * 2 + ["C"] * 2,
             config.COLOR_COL: ["Black"] * 10,
             }
        )
        grouper = RareCategoriesGrouper(
            columns=(config.BREED_COL, config.COLOR_COL,), max_other_ratio=max_ratio
        )
        grouper.fit(X_train)

        assert set(grouper.frequent_categories_[config.BREED_COL]) == expected_frequent
        assert set(grouper.frequent_categories_[config.COLOR_COL]) == {"Black"}


# =====================================================================
#                   CategoricalFeaturesEngineer TESTS
# =====================================================================

# -------------------- FIT TESTS --------------------

class TestCategoricalFeaturesEngineerFit:
    """Testing the internal RareCategoriesGrouper that CategoricalFeaturesEngineer fits."""

    def test_categorical_features_engineer_initializes_and_fits_grouper(self):
        """Verify that fit initializes and fits the internal RareCategoriesGrouper instance.

        GIVEN: a DataFrame with target categorical columns
        WHEN: fit is executed
        THEN: grouper_ attribute is instantiated as a fitted RareCategoriesGrouper
        """
        X_train = pd.DataFrame(
            {
                config.BREED_COL: ["Labrador Mix"] * 8 + ["Poodle"] * 2,
                config.COLOR_COL: ["Black/White"] * 8 + ["Red"] * 2,
            }
        )
        engineer = CategoricalFeaturesEngineer().fit(X_train)

        assert isinstance(engineer.grouper_, RareCategoriesGrouper)
        assert set(engineer.grouper_.frequent_categories_) == {config.BREED_COL, config.COLOR_COL}

    def test_categorical_features_engineer_fit_learns_nothing_from_an_empty_column(
        self, caplog: pytest.LogCaptureFixture
    ):
        """Verify that an all-missing column is fitted safely, with a warning.

        GIVEN: a training DataFrame containing only missing values (NaN) in target columns
        WHEN: fit is executed with WARNING logging level captured
        THEN: a warning is recorded and the internal grouper learns an empty
              set of frequent categories rather than crashing
        """
        X_train = pd.DataFrame(
            {config.BREED_COL: [np.nan, np.nan], config.COLOR_COL: [np.nan, np.nan]}
        )
        engineer = CategoricalFeaturesEngineer(columns=(config.BREED_COL, config.COLOR_COL))

        with caplog.at_level(logging.WARNING):
            engineer.fit(X_train)

        assert "is empty or contains only NaNs" in caplog.text
        assert engineer.grouper_ is not None
        assert engineer.grouper_.frequent_categories_ == {
            config.BREED_COL: (),
            config.COLOR_COL: (),
        }


    def test_categorical_features_engineer_fit_empty_dataframe(self):
        """Verify fit execution when training DataFrame is completely empty.

        GIVEN: an empty training DataFrame with valid column headers
        WHEN: fit is executed on CategoricalFeaturesEngineer
        THEN: execution succeeds safely and learns default empty states
        """
        X_train = pd.DataFrame(columns=(config.BREED_COL, config.COLOR_COL))
        engineer = CategoricalFeaturesEngineer(columns=(config.BREED_COL, config.COLOR_COL))

        fitted_engineer = engineer.fit(X_train)

        assert fitted_engineer.grouper_.frequent_categories_ == {
            config.BREED_COL: (),
            config.COLOR_COL: (),
        }

    def test_categorical_features_engineer_fit_raises_on_missing_column(self):
        """Verify that CategoricalFeaturesEngineer raises an error when target columns are missing.

        GIVEN: a DataFrame that lacks both breed and color target columns
        WHEN: fit is executed on CategoricalFeaturesEngineer
        THEN: a ValueError names the missing column
        """
        X_train = pd.DataFrame({config.NAME_COL: ["Bella", "Max"]})
        engineer = CategoricalFeaturesEngineer(
            columns=(config.BREED_COL, config.COLOR_COL)
        )

        with pytest.raises(ValueError, match=config.BREED_COL):
            engineer.fit(X_train)

# -------------------- TRANSFORM TESTS --------------------

class TestCategoricalFeaturesEngineerTransform:
    """Testing the mix indicator, primary-form reduction, and rare-category binning once fitted."""

    def test_categorical_features_engineer_success(self):
        """Verify that the mix indicator, the primary forms and the binning combine.

        GIVEN: a frame where a slash-separated breed and a "Mix" one both reduce
               to the same primary value, plus a rare breed and a missing one
        WHEN: fit_transform is executed with max_other_ratio=0.25
        THEN: the mix indicator flags the first two rows, both collapse onto the
              same primary breed, the rare one becomes "Other" and the missing
              one stays missing
        """
        X_train = pd.DataFrame(
            {
                config.BREED_COL: ["A/B", "A Mix", "A", "A", "C", np.nan],
                config.COLOR_COL: ["Black/White", "Black", "Black", "Black", "Blue", np.nan],
            },
            index=[10, 20, 30, 40, 50, 60],
        )

        expected_is_mix = pd.Series(
            [1, 1, 0, 0, 0, 0], index=[10, 20, 30, 40, 50, 60], name=config.IS_MIX_COL
        )
        expected_breed = pd.Series(
            ["A", "A", "A", "A", "Other", np.nan],
            index=[10, 20, 30, 40, 50, 60],
            name=config.BREED_COL,
        )
        expected_color = pd.Series(
            ["Black", "Black", "Black", "Black", "Other", np.nan],
            index=[10, 20, 30, 40, 50, 60],
            name=config.COLOR_COL,
        )

        engineer = CategoricalFeaturesEngineer(
            columns=(config.BREED_COL, config.COLOR_COL), max_other_ratio=0.25
        )
        X_clean = engineer.fit_transform(X_train)

        pd.testing.assert_series_equal(X_clean[config.IS_MIX_COL], expected_is_mix)
        pd.testing.assert_series_equal(X_clean[config.BREED_COL], expected_breed)
        pd.testing.assert_series_equal(X_clean[config.COLOR_COL], expected_color)

    def test_categorical_features_engineer_raises_not_fitted_error_when_unfitted(self):
        """Verify that CategoricalFeaturesEngineer raises a NotFittedError if transform is called
           before fit.

        GIVEN: a CategoricalFeaturesEngineer instance that has not been fitted yet
        WHEN: the transform method is executed on a DataFrame
        THEN: it must immediately raise a NotFittedError indicating the instance is not fitted
        """
        X_train = pd.DataFrame({config.BREED_COL: ["A", "B"], config.COLOR_COL: ["Black", "White"]})
        engineer = CategoricalFeaturesEngineer()

        with pytest.raises(
            NotFittedError, match="CategoricalFeaturesEngineer instance is not fitted"
        ):
            engineer.transform(X_train)

    def test_categorical_features_engineer_preserves_nans(self):
        """Verify that missing values (NaN) in breed and color columns are preserved and mix
           indicator is 0.

        GIVEN: a DataFrame containing missing values (NaN) in breed and color columns
        WHEN: transform is executed
        THEN: NaNs are preserved in target columns and mix indicator is 0 for those rows
        """
        X_train = pd.DataFrame(
            {
                config.BREED_COL: ["Labrador", np.nan],
                config.COLOR_COL: ["Black", np.nan],
            },
            index=[10, 20],
        )
        engineer = CategoricalFeaturesEngineer().fit(X_train)

        X_clean = engineer.transform(X_train)

        assert pd.isna(X_clean.loc[20, config.BREED_COL])
        assert pd.isna(X_clean.loc[20, config.COLOR_COL])
        assert X_clean.loc[20, config.IS_MIX_COL] == 0

    def test_categorical_features_engineer_preserves_index_and_non_target_columns(self):
        """Verify that transform leaves the index and the other columns alone.

        GIVEN: a fitted engineer and a frame carrying a column it does not
               address, under a non-default index
        WHEN: transform is executed
        THEN: that column and the index come back untouched, the engineer
              only rewriting its own columns and appending the mix indicator
        """
        X = pd.DataFrame(
            {
                config.BREED_COL: ["Labrador Mix"] * 8 + ["Poodle"] * 2,
                config.COLOR_COL: ["Black/White"] * 10,
                "Other_Col": range(10),
            },
            index=range(100, 110),
        )

        X_transformed = CategoricalFeaturesEngineer(max_other_ratio=0.25).fit_transform(X)

        pd.testing.assert_series_equal(X_transformed["Other_Col"], X["Other_Col"])
        assert list(X_transformed.index) == list(range(100, 110))

    def test_categorical_features_engineer_empty_dataframe_with_columns(self):
        """Verify that CategoricalFeaturesEngineer handles empty DataFrames safely without crashing.

        GIVEN: a fitted CategoricalFeaturesEngineer and an empty DataFrame with valid schema
        WHEN: transform is executed
        THEN: the returned DataFrame is empty, and expected schema (breed, color, mix indicator) is
              created
        """
        X_train = pd.DataFrame(
            {config.BREED_COL: ["Labrador Mix"], config.COLOR_COL: ["Black/White"]}
        )
        X_empty = pd.DataFrame(columns=(config.BREED_COL, config.COLOR_COL))

        engineer = CategoricalFeaturesEngineer(
            columns=(config.BREED_COL, config.COLOR_COL)
        ).fit(X_train)
        X_transformed = engineer.transform(X_empty)

        assert X_transformed.empty
        assert set(X_transformed.columns) == {config.BREED_COL, config.COLOR_COL, config.IS_MIX_COL}

    def test_categorical_features_engineer_transform_raises_on_missing_column(self):
        """Verify transform behavior when one fitted column is missing during inference.

        GIVEN: a CategoricalFeaturesEngineer fitted on both breed and color, and a test DataFrame
               missing breed
        WHEN: transform is executed on the test DataFrame
        THEN: a ValueError names the missing column, since going ahead would
              silently drop both the mix indicator and the binning of breed
        """
        X_train = pd.DataFrame(
            {config.BREED_COL: ["Labrador"], config.COLOR_COL: ["Black"]}, index=[10]
        )
        X_test = pd.DataFrame({config.COLOR_COL: ["Black"]}, index=[20])

        engineer = CategoricalFeaturesEngineer(
            columns=(config.BREED_COL, config.COLOR_COL)).fit(X_train)

        with pytest.raises(ValueError, match=config.BREED_COL):
            engineer.transform(X_test)

    def test_categorical_features_engineer_logs_debug_on_drift(
        self, caplog: pytest.LogCaptureFixture
    ):
        """Verify debug log emission on data drift.

        GIVEN: a fitted CategoricalFeaturesEngineer instance, and a test DataFrame experiencing data
               drift
        WHEN: transform is executed on the test dataset with DEBUG logging level captured
        THEN: a DEBUG log message is recorded indicating that 'Other' ratio exceeds max_other_ratio
        """
        X_train = pd.DataFrame(
            {config.BREED_COL: ["A"] * 90 + ["B"] * 10, config.COLOR_COL: ["Black"] * 100}
        )
        X_test = pd.DataFrame({config.BREED_COL: ["B"] * 100, config.COLOR_COL: ["Black"] * 100})

        engineer = CategoricalFeaturesEngineer(
            columns=(config.BREED_COL, config.COLOR_COL), max_other_ratio=0.15
        )
        engineer.fit(X_train)

        with caplog.at_level(logging.DEBUG):
            engineer.transform(X_test)

        assert "exceeds the configured max_other_ratio" in caplog.text


# -------------------- END TO END, CUSTOM & EDGE CASES --------------------

class TestCategoricalFeaturesEngineerCustomAndE2E:
    """Testing CategoricalFeaturesEngineer end to end on degenerate all-missing and empty frames."""

    def test_categorical_features_engineer_all_nan(self):
        """Verify robust handling when breed and color columns contain only missing entries.

        GIVEN: a DataFrame where breed and color columns contain only NaNs
        WHEN: fit and transform are executed
        THEN: NaNs are preserved, mix indicator contains all 0s, and no exception is raised
        """
        X_mock = pd.DataFrame(
            {
                config.BREED_COL: [np.nan, np.nan],
                config.COLOR_COL: [np.nan, np.nan],
            },
            index=[10, 20],
        )

        engineer = CategoricalFeaturesEngineer()
        X_clean = engineer.fit_transform(X_mock)

        assert (X_clean[config.IS_MIX_COL] == 0).all()
        assert X_clean[config.BREED_COL].isna().all()
        assert X_clean[config.COLOR_COL].isna().all()


    def test_categorical_features_engineer_prevents_data_leakage(self):
        """Verify that test set category grouping strictly uses statistics learned during training.

        GIVEN: an engineer trained on X_train (frequent breed: 'Labrador') and a test set with
               unseen labels
        WHEN: transform is called on the test set
        THEN: unseen breeds are mapped to 'Other' based strictly on training set statistics
        """
        X_train = pd.DataFrame({
            config.BREED_COL: ["Labrador Mix"] * 8 + ["Poodle"] * 2,
            config.COLOR_COL: ["Black"] * 10,
        })
        X_test = pd.DataFrame({
            config.BREED_COL: ["Beagle", "Chihuahua"],
            config.COLOR_COL: ["Black", "Black"],
        })

        engineer = CategoricalFeaturesEngineer(max_other_ratio=0.25).fit(X_train)
        X_test_clean = engineer.transform(X_test)

        expected_is_mix = pd.Series([0, 0], name=config.IS_MIX_COL)
        pd.testing.assert_series_equal(X_test_clean[config.IS_MIX_COL], expected_is_mix)

        expected_breed = pd.Series(["Other", "Other"], name=config.BREED_COL)
        pd.testing.assert_series_equal(X_test_clean[config.BREED_COL], expected_breed)

        expected_color = pd.Series(["Black", "Black"], name=config.COLOR_COL)
        pd.testing.assert_series_equal(X_test_clean[config.COLOR_COL], expected_color)

    def test_categorical_features_engineer_custom_max_other_ratio(self):
        """Verify that max_other_ratio custom parameter is passed down to internal
           RareCategoriesGrouper.

        GIVEN: a custom max_other_ratio value (0.5) and a training dataset with breed and color
        WHEN: CategoricalFeaturesEngineer executes fit
        THEN: internal grouper_ receives and uses the custom max_other_ratio
        """
        X_train = pd.DataFrame(
            {config.BREED_COL: ["Labrador"] * 6 + ["Poodle"] * 2 + ["Chihuahua"] * 2,
            config.COLOR_COL: ["Black"] * 10,
            })

        engineer = CategoricalFeaturesEngineer(max_other_ratio=0.5).fit(X_train)

        assert engineer.grouper_.max_other_ratio == 0.5
        assert engineer.grouper_.frequent_categories_[config.BREED_COL] == ("Labrador",)
        assert engineer.grouper_.frequent_categories_[config.COLOR_COL] == ("Black",)

    def test_categorical_features_engineer_bins_a_column_without_a_cleaner(self):
        """Verify that a custom column is grouped even with no cleaning rule.

        GIVEN: an engineer declaring a column that PRIMARY_EXTRACTORS ignores
        WHEN: fit_transform is executed
        THEN: its rare values are still collapsed into "Other", the cleaning
              step applying only to the columns that have a rule
        """
        X = pd.DataFrame({"my_cat": ["a"] * 8 + ["b"] * 2})

        X_out = CategoricalFeaturesEngineer(
            columns=("my_cat",), max_other_ratio=0.25
        ).fit_transform(X)

        assert X_out["my_cat"].value_counts().to_dict() == {"a": 8, "Other": 2}

    def test_categorical_features_engineer_leaves_undeclared_columns_alone(self):
        """Verify that a present but undeclared column is not binned.

        GIVEN: an engineer declaring only the breed column, on a frame that
               also carries the colour one
        WHEN: fit_transform is executed
        THEN: the colour column comes back with its raw values, "columns"
              bounding what the engineer touches rather than what it sees
        """
        X = pd.DataFrame(
            {
                config.BREED_COL: ["Labrador Mix"] * 8 + ["Poodle"] * 2,
                config.COLOR_COL: ["Black/White"] * 8 + ["Red"] * 2,
            }
        )

        X_transformed = CategoricalFeaturesEngineer(
            columns=(config.BREED_COL,), max_other_ratio=0.25
        ).fit_transform(X)

        pd.testing.assert_series_equal(X_transformed[config.COLOR_COL], X[config.COLOR_COL])
