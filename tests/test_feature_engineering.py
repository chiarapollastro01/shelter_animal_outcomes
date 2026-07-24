"""
Unit tests for the feature engineering module.
"""

import numpy as np
import pandas as pd
import pytest
import logging
from src.feature_engineering import (
    TemporalFeaturesExtractor,
    CategoricalFeaturesEngineer,
    RareCategoriesGrouper,
    extract_primary_breed,
    extract_primary_color,
    SexFeaturesExtractor,
    NameFeaturesExtractor
)

two_pi= 2* np.pi # GLOBAL CONSTANT

# =====================================================================
#                       PARAMETRIZED TESTS
# =====================================================================

@pytest.mark.parametrize(
    "transformer, custom_col, mock_df, expected_col",
    [
        (
            TemporalFeaturesExtractor(datetime_col="my_date"),
            "my_date",
            pd.DataFrame({"my_date": ["2026-07-06 12:00:00"]}),
            "IsWeekend",
        ),
        (
            SexFeaturesExtractor(sex_col="my_sex"),
            "my_sex",
            pd.DataFrame({"my_sex": ["Intact Male"]}),
            "Reproductive_Status",
        ),
        (
            NameFeaturesExtractor(name_col="my_name"),
            "my_name",
            pd.DataFrame({"my_name": ["Bella"]}),
            "has_name",
        ),
    ],
)
def test_extractors_support_custom_column_names(transformer, custom_col, mock_df, expected_col):
    """Verify that extractors correctly process data when target column names are customized.

    GIVEN: an extractor configured with a non-default column name and a DataFrame containing that column
    WHEN: fit_transform is executed
    THEN: the custom input column is dropped and the expected feature column is created
    """
    df_transformed = transformer.fit_transform(mock_df)

    assert custom_col not in df_transformed.columns
    assert expected_col in df_transformed.columns


@pytest.mark.parametrize(
    "transformer",
    [
        TemporalFeaturesExtractor(),
        SexFeaturesExtractor(),
        NameFeaturesExtractor(),
        CategoricalFeaturesEngineer(),
        RareCategoriesGrouper(columns=["Breed", "Color"]), 
    ],
)
def test_transformers_do_not_mutate_input(transformer):
    """Verify that feature transformers do not mutate the input DataFrame in-place.

    GIVEN: any feature transformer and a raw input DataFrame
    WHEN: fit_transform is executed
    THEN: the original input DataFrame remains completely unchanged
    """
    df_mock = pd.DataFrame({
        "DateTime": ["2026-07-06 12:00:00"],
        "SexuponOutcome": ["Neutered Male"],
        "Name": ["Bella"],
        "Breed": ["Labrador Mix"],
        "Color": ["Black/White"],
    })
    original = df_mock.copy(deep=True)

    transformer.fit_transform(df_mock)

    pd.testing.assert_frame_equal(df_mock, original)


@pytest.mark.parametrize(
    "transformer",
    [
        TemporalFeaturesExtractor(),
        SexFeaturesExtractor(),
        NameFeaturesExtractor(),
        CategoricalFeaturesEngineer(),
        RareCategoriesGrouper(columns=["Breed", "Color"]),
    ],
)
def test_transformers_fit_returns_self(transformer):
    """Verify that the fit method of any feature transformer returns the instance itself.

    GIVEN: any feature transformer and a valid mockup DataFrame containing required features
    WHEN: the fit method is executed
    THEN: the returned object is exactly the same instance of the transformer
    """
    df_mock = pd.DataFrame({
        "DateTime": ["2026-07-06 12:00:00"],
        "SexuponOutcome": ["Neutered Male"],
        "Name": ["Bella"],
        "Breed": ["Labrador Mix"],
        "Color": ["Black/White"],
    })

    fitted_transformer = transformer.fit(df_mock)

    assert fitted_transformer is transformer


# =====================================================================
#                       TEMPORAL FEATURE TESTS
# =====================================================================

def test_temporal_extractor_success():
    """Verify that TemporalFeaturesExtractor correctly extracts cyclic features and IsWeekend.

    GIVEN: a DataFrame with raw string dates (Monday, Friday, Saturday, Sunday) and custom indices
    WHEN: transform is executed
    THEN: DateTime is dropped, IsWeekend is correctly assigned (0 for weekdays, 1 for weekends),
          cyclic columns are created, and the original index is preserved
    """
    df_mock = pd.DataFrame(
        {
            "DateTime": [
                "2026-07-06 12:00:00",  # Monday (0) -> IsWeekend = 0
                "2026-07-10 12:00:00",  # Friday (4) -> IsWeekend = 0
                "2026-07-11 12:00:00",  # Saturday (5) -> IsWeekend = 1
                "2026-07-12 12:00:00",  # Sunday (6) -> IsWeekend = 1
            ]
        },
        index=[10, 20, 30, 40],
    )
    expected_weekend = pd.Series([0.0, 0.0, 1.0, 1.0], index=[10, 20, 30, 40], name="IsWeekend")
    expected_columns = {"Hour_sin", "Hour_cos", "Wday_sin", "Wday_cos", "DoY_sin", "DoY_cos", "IsWeekend"}

    df_transformed = TemporalFeaturesExtractor().fit_transform(df_mock)

    assert "DateTime" not in df_transformed.columns
    assert set(df_transformed.columns) == expected_columns

    pd.testing.assert_series_equal(df_transformed["IsWeekend"], expected_weekend, check_dtype=False)


def test_temporal_extractor_cyclic_hours():
   """Verify the mathematical correctness of sine and cosine transformations for hours.

    GIVEN: a DataFrame with precise timestamps representing Midnight (hour 0), 6 AM (hour 6), and Noon (hour 12)
           and a custom non-default index
    WHEN: the transform method of TemporalFeaturesExtractor is executed
    THEN: Hour_sin and Hour_cos calculate accurate values at boundaries (0, pi/2, pi) respectively,
          preserving the original index and handling float precision tolerances
    """
   df_mock = pd.DataFrame({
            "DateTime": [
                "2026-01-01 00:00:00",  
                "2026-01-01 06:00:00",  
                "2026-01-01 12:00:00", 
            ]
        },
        index=[11, 22, 33],
    )
   expected_sin = pd.Series(np.sin(two_pi * np.array([0, 6, 12]) / 24), index=[11, 22, 33], name="Hour_sin")
   expected_cos = pd.Series(np.cos(two_pi* np.array([0, 6, 12]) / 24), index=[11, 22, 33], name="Hour_cos")

   df_transformed = TemporalFeaturesExtractor().fit_transform(df_mock)

   pd.testing.assert_series_equal(df_transformed["Hour_sin"], expected_sin, check_exact=False, atol=1e-7)
   pd.testing.assert_series_equal(df_transformed["Hour_cos"], expected_cos, check_exact=False, atol=1e-7)


def test_temporal_extractor_cyclic_weekdays():
     """Verify the mathematical correctness of sine and cosine transformations for weekdays.

    GIVEN: a DataFrame containing a Monday (weekday 0) and a Sunday (weekday 6) with custom non-default indices
    WHEN: the transform method of TemporalFeaturesExtractor is executed
    THEN: Wday_sin and Wday_cos are computed accurately based on the weekday, preserving the index
          and handling float precision tolerances
    """
     df_mock = pd.DataFrame({"DateTime": ["2026-07-06", "2026-07-12"]}, index=[15, 25])  # Lunedì, Domenica
     expected_sin = pd.Series(np.sin(two_pi * np.array([0, 6]) / 7), index=[15, 25], name="Wday_sin")
     expected_cos = pd.Series(np.cos(two_pi * np.array([0, 6]) / 7), index=[15, 25], name="Wday_cos")

     df_transformed = TemporalFeaturesExtractor().fit_transform(df_mock)

     pd.testing.assert_series_equal(df_transformed["Wday_sin"], expected_sin, check_exact=False, atol=1e-7)
     pd.testing.assert_series_equal(df_transformed["Wday_cos"], expected_cos, check_exact=False, atol=1e-7)


def test_temporal_extractor_cyclic_day_of_year():
    """Verify the mathematical correctness of sine and cosine transformations for the day of the year.

    GIVEN: a DataFrame containing dates representing Day of Year 1 (Jan 1) and 100 (Apr 10) in a non-leap year (2026),
           with custom non-default indices
    WHEN: the transform method of TemporalFeaturesExtractor is executed
    THEN: DoY_sin and DoY_cos columns are computed accurately based on the day of the year, preserving the index
          and handling float precision tolerances
    """
    df_mock = pd.DataFrame({"DateTime": ["2026-01-01", "2026-04-10"]}, index=[11, 22])
    expected_sin = pd.Series(np.sin(two_pi * np.array([1, 100]) / 365.25), index=[11, 22], name="DoY_sin")
    expected_cos = pd.Series(np.cos(two_pi * np.array([1, 100]) / 365.25), index=[11, 22], name="DoY_cos")

    df_transformed = TemporalFeaturesExtractor().fit_transform(df_mock)

    pd.testing.assert_series_equal(df_transformed["DoY_sin"], expected_sin, check_exact=False, atol=1e-7)
    pd.testing.assert_series_equal(df_transformed["DoY_cos"], expected_cos, check_exact=False, atol=1e-7)


def test_temporal_extractor_already_datetime_type():
    """Verify that the transformer produces identical values regardless of whether the input is raw string or datetime64.

    GIVEN: two identical DataFrames, one with raw string dates and one pre-converted to datetime64[ns]
    WHEN: transform is executed on both
    THEN: both executions succeed, producing perfectly identical DataFrames in both schema and values, 
          preserving the index and dropping DateTime
    """
    df_strings = pd.DataFrame({"DateTime": ["2026-07-06 12:00:00"]}, index=[99])
    df_datetime = pd.DataFrame({"DateTime": pd.to_datetime(["2026-07-06 12:00:00"])}, index=[99])
    
    res_strings = TemporalFeaturesExtractor().fit_transform(df_strings)
    res_datetime = TemporalFeaturesExtractor().fit_transform(df_datetime)
    
    pd.testing.assert_frame_equal(res_strings, res_datetime)


def test_temporal_extractor_empty_dataframe_with_columns():
    """Verify that an empty DataFrame with the target column is processed safely without crashing.

    GIVEN: an empty DataFrame with only the DateTime column in its schema
    WHEN: transform is executed on TemporalFeaturesExtractor
    THEN: the returned DataFrame is empty, DateTime is dropped, and the expected empty schema is preserved
    """
    df_mock = pd.DataFrame(columns=["DateTime"])

    expected_cols = {"IsWeekend", "Hour_sin", "Hour_cos", "Wday_sin", "Wday_cos", "DoY_sin", "DoY_cos"}
    
    df_transformed = TemporalFeaturesExtractor().fit_transform(df_mock)
    
    assert df_transformed.empty
    assert set(df_transformed.columns) == expected_cols

def test_temporal_extractor_handles_null_values():
    """Verify that missing or invalid date values do not crash the execution and propagate as NaN.

    GIVEN: a DataFrame with a valid DateTime and a None value, under custom indices
    WHEN: transform is executed on TemporalFeaturesExtractor
    THEN: the valid date is calculated, and the missing value safely propagates as NaN in the cyclic columns
    """
    df_mock = pd.DataFrame({"DateTime": ["2026-07-12 12:00:00", None]}, index=[101, 102])
    
    df_transformed = TemporalFeaturesExtractor().fit_transform(df_mock)
    

    assert not np.isnan(df_transformed["Hour_sin"].loc[101])

    assert np.isnan(df_transformed["Hour_sin"].loc[102])
    assert np.isnan(df_transformed["Wday_cos"].loc[102])
    assert np.isnan(df_transformed["DoY_sin"].loc[102])
    
    assert df_transformed["IsWeekend"].loc[101] == 1.0
    assert np.isnan(df_transformed["IsWeekend"].loc[102])


def test_temporal_extractor_missing_column():
    """Verify that the base transformer returns the DataFrame unchanged when
    the target column is missing.

    GIVEN: a DataFrame that lacks the specified target DateTime column, with a
    custom index
    WHEN: the transform method is executed
    THEN: the input DataFrame is returned completely unmodified, preserving its
    index and values
    """

    df_mock = pd.DataFrame({"Name": ["Bella", "Max"]}, index=[100, 200])

    df_transformed = TemporalFeaturesExtractor(datetime_col="DateTime").fit_transform(
        df_mock
    )

    pd.testing.assert_frame_equal(df_transformed, df_mock)


# =====================================================================
#                           SEX FEATURE TESTS
# =====================================================================
def test_sex_features_extractor_success():
    """Verify that SexFeaturesExtractor successfully extracts Reproductive_Status and drops the raw column.

    GIVEN: a DataFrame with raw SexuponOutcome values containing neutered,
           spayed, intact, unknown, and NaN entries, with a custom index
    WHEN: transform is executed on SexFeaturesExtractor
    THEN: the raw column is dropped and Reproductive_Status 
          is accurately mapped, preserving the original index
    """
    df_mock = pd.DataFrame({
        "SexuponOutcome": ["Neutered Male", "Spayed Female", "Intact Male", "Unknown", np.nan]
    }, index=[10, 20, 30, 40, 50])
    

    expected = pd.Series(
        ["Neutered/Spayed", "Neutered/Spayed", "Intact", "Unknown", "Unknown"],
        index=[10, 20, 30, 40, 50],
        name="Reproductive_Status"
    )
    
    df_transformed = SexFeaturesExtractor().fit_transform(df_mock)
    
    assert "SexuponOutcome" not in df_transformed.columns
    pd.testing.assert_series_equal(df_transformed["Reproductive_Status"], expected)

def test_sex_extractor_case_insensitivity_and_formatting():
    """Verify robust parsing of lower/uppercase variations in sex strings.

    GIVEN: a DataFrame with lowercase, uppercase, and untrimmed sex strings
    WHEN: transform is executed on SexFeaturesExtractor
    THEN: categories are mapped correctly regardless of letter casing
    """
    df_mock = pd.DataFrame(
        {"SexuponOutcome": ["spayed female", "NEUTERED MALE", "intact male"]}, index=[1, 2, 3]
    )
    expected_status = pd.Series(
        ["Neutered/Spayed", "Neutered/Spayed", "Intact"], index=[1, 2, 3], name="Reproductive_Status"
    )

    df_transformed = SexFeaturesExtractor().fit_transform(df_mock)

    pd.testing.assert_series_equal(df_transformed["Reproductive_Status"], expected_status)


def test_sex_extractor_empty_dataframe_with_columns():
    """Verify that an empty DataFrame with the target sex column is processed safely.

    GIVEN: an empty DataFrame with only the 'SexuponOutcome' column in its schema
    WHEN: transform is executed on SexFeaturesExtractor
    THEN: the returned DataFrame is empty, 'SexuponOutcome' is dropped, and 'Reproductive_Status' is created
    """
    df_mock = pd.DataFrame(columns=["SexuponOutcome"])

    df_transformed = SexFeaturesExtractor().fit_transform(df_mock)

    assert df_transformed.empty
    assert list(df_transformed.columns) == ["Reproductive_Status"]


def test_sex_extractor_handles_null_values():
    """Verify that missing (NaN) sex values map safely to 'Unknown'.

    GIVEN: a DataFrame containing a valid sex string and a NaN entry under custom indices
    WHEN: transform is executed on SexFeaturesExtractor
    THEN: NaN entries resolve to 'Unknown' without raising exceptions
    """
    df_mock = pd.DataFrame({"SexuponOutcome": ["Intact Male", np.nan]}, index=[101, 102])

    df_transformed = SexFeaturesExtractor().fit_transform(df_mock)

    assert df_transformed.loc[101, "Reproductive_Status"] == "Intact"
    assert df_transformed.loc[102, "Reproductive_Status"] == "Unknown"

    
def test_sex_extractor_missing_column():
    """Verify that SexFeaturesExtractor returns the DataFrame untouched when the target column is missing.

    GIVEN: a DataFrame that does not contain the required SexuponOutcome column,
           with a custom index
    WHEN: transform is executed on SexFeaturesExtractor
    THEN: the input DataFrame is returned completely unmodified, preserving its
          index and values
    """
  
    df_mock = pd.DataFrame({"Name": ["Bella", "Max"]}, index=[100, 200])

    df_transformed = SexFeaturesExtractor(sex_col="SexuponOutcome").fit_transform(
        df_mock
    )

    pd.testing.assert_frame_equal(df_transformed, df_mock)

# =====================================================================
#                         NAME FEATURE TESTS
# =====================================================================

def test_name__extractor_success():
    """Verify that NameFeaturesExtractor correctly creates 'has_name' and drops the raw column.

    GIVEN: a DataFrame with standard valid names and the placeholder "Unknown",
           with custom non-default indices
    WHEN: fit_transform is executed on NameFeaturesExtractor
    THEN: the raw 'Name' column is dropped, 'has_name' is 1 for valid names and 0 for 'Unknown',
          and the original index is preserved
    """
    df_mock = pd.DataFrame(
        {"Name": ["Bella", "Max", "Unknown"]},
        index=[10, 20, 30],
    )
    expected = pd.Series(
        [1, 1, 0], index=[10, 20, 30], name="has_name"
    )

    df_transformed = NameFeaturesExtractor().fit_transform(df_mock)

    assert "Name" not in df_transformed.columns
    pd.testing.assert_series_equal(
        df_transformed["has_name"], expected, check_dtype=False
    )


def test_name_extractor_case_insensitivity_and_formatting():
    """Verify robust parsing of whitespace variations and case-insensitivity for 'Unknown'.

    GIVEN: a DataFrame with 'Unknown' in various casings and extra surrounding spaces
    WHEN: fit_transform is executed on NameFeaturesExtractor
    THEN: 'has_name' resolves correctly to 0 for unknown/empty variations and 1 for valid names
    """
    df_mock = pd.DataFrame(
        {"Name": ["  Bella  ", "unknown", "  UNKNOWN  ", "   "]},
        index=[1, 2, 3, 4],
    )
    expected = pd.Series([1, 0, 0, 0], index=[1, 2, 3, 4], name="has_name")

    df_transformed = NameFeaturesExtractor().fit_transform(df_mock)

    pd.testing.assert_series_equal(
        df_transformed["has_name"], expected, check_dtype=False
    )

def test_name_extractor_empty_dataframe_with_columns():
    """Verify that an empty DataFrame with the target name column is processed safely.

    GIVEN: an empty DataFrame with only the 'Name' column in its schema
    WHEN: transform is executed on NameFeaturesExtractor
    THEN: the returned DataFrame is empty, 'Name' is dropped, and 'has_name' is created
    """
    df_mock = pd.DataFrame(columns=["Name"])

    df_transformed = NameFeaturesExtractor().fit_transform(df_mock)

    assert df_transformed.empty
    assert list(df_transformed.columns) == ["has_name"]


def test_name_extractor_handles_null_values():
    """Verify that missing (NaN/None) name values safely resolve to has_name = 0.

    GIVEN: a DataFrame containing a valid name and NaN/None entries under custom indices
    WHEN: fit_transform is executed on NameFeaturesExtractor
    THEN: NaN/None entries map to 0 in 'has_name' without raising exceptions
    """
    df_mock = pd.DataFrame({"Name": ["Max", np.nan, None]}, index=[101, 102, 103])
    expected = pd.Series([1, 0, 0], index=[101, 102, 103], name="has_name")

    df_transformed = NameFeaturesExtractor().fit_transform(df_mock)

    pd.testing.assert_series_equal(
        df_transformed["has_name"], expected, check_dtype=False
    )

def test_name_extractor_missing_column():
    """Verify that NameFeaturesExtractor returns the DataFrame untouched when

    the Name column is missing.

    GIVEN: a DataFrame that does not contain the required Name column, with a
    custom index
    WHEN: transform is executed on NameFeaturesExtractor
    THEN: the input DataFrame is returned completely unmodified, preserving values
    """
    df_mock = pd.DataFrame({"Age": [10, 20]}, index=[100, 200])

    df_transformed = NameFeaturesExtractor(name_col="Name").fit_transform(df_mock)

    pd.testing.assert_frame_equal(df_transformed, df_mock)

# =====================================================================
#                     HELPER FUNCTIONS TESTS
# =====================================================================

# ----------------- EXTRACT PRIMARY COLOR -----------------

def test_extract_primary_color_logic():
    """Verify that only the primary color is extracted by splitting on '/'.

    GIVEN: a Series containing bicolor entries ('Black/White'), tricolor entries
           ('Brown Tabby/White'), and solid colors ('Blue') with custom indices
    WHEN: extract_primary_color is executed
    THEN: only the first color before the slash is kept, removing trailing
    spaces,
          preserving the original index
    """

    color_series = pd.Series(
        ["Black/White", "Brown Tabby/White", "Blue"], index=[10, 20, 30]
    )
    expected = pd.Series(["Black", "Brown Tabby", "Blue"], index=[10, 20, 30])

    result = extract_primary_color(color_series)


    pd.testing.assert_series_equal(result, expected)


def test_extract_primary_color_all_null_series():
    """Verify that a Series composed entirely of NaNs is handled safely.

    GIVEN: a Series containing only NaN values with custom non-default indices
    WHEN: extract_primary_color is executed
    THEN: all elements remain NaN, preserving the original length and custom index
    """
    nan_series = pd.Series([np.nan, np.nan], index=[10, 20])
    expected = pd.Series([np.nan, np.nan], index=[10, 20])

    result = extract_primary_color(nan_series)

    pd.testing.assert_series_equal(result, expected)


def test_extract_primary_color_empty_series():
    """Verify behavior when processing an empty pandas Series.

    GIVEN: a completely empty pandas Series (length 0)
    WHEN: extract_primary_color is executed
    THEN: an empty Series is returned without errors
    """
    empty_series = pd.Series([], dtype=object)

    result = extract_primary_color(empty_series)

    assert result.empty
    pd.testing.assert_series_equal(result, empty_series)


def test_extract_primary_color_formatting_variations():
    """Verify primary color extraction with multiple slashes, extra spaces, and single colors.

    GIVEN: a Series containing multiple slashes ('Black/White/Brown'), surrounding spaces, and single colors
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

def test_extract_primary_breed_logic():
    """Verify that the primary breed is correctly extracted by splitting '/' and

    stripping 'Mix'.

    GIVEN: a Series containing crossbreeds with slashes, 'Mix' suffixes, and
    purebreds,
           with custom indices
    WHEN: extract_primary_breed is executed
    THEN: the 'Mix' keyword and second breeds are removed, leaving only the
    clean primary breed,
          preserving the index
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

def test_extract_primary_breed_all_null_series():
    """Verify that a Series composed entirely of NaNs is handled safely.

    GIVEN: a Series containing only NaN values with custom non-default indices
    WHEN: extract_primary_breed is executed
    THEN: all elements remain NaN, preserving the original length and custom index
    """
    nan_series = pd.Series([np.nan, np.nan], index=[101, 102])
    expected = pd.Series([np.nan, np.nan], index=[101, 102])

    result = extract_primary_breed(nan_series)

    pd.testing.assert_series_equal(result, expected)


def test_extract_primary_breed_empty_series():
    """Verify behavior when processing an empty pandas Series.

    GIVEN: a completely empty pandas Series (length 0)
    WHEN: extract_primary_breed is executed
    THEN: an empty Series is returned without errors
    """
    empty_series = pd.Series([], dtype=object)

    result = extract_primary_breed(empty_series)

    assert result.empty
    pd.testing.assert_series_equal(result, empty_series)


def test_extract_primary_breed_formatting_variations():
    """Verify primary breed extraction with case variations of 'mix', multiple slashes, and clean breeds.

    GIVEN: a Series with 'mix' in different casing ('MIX', 'mix'), multiple slashes, and pure breeds
    WHEN: extract_primary_breed is executed
    THEN: trailing 'mix' keywords and secondary breeds are stripped regardless of casing, preserving the index
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
#                  COLOR AND BREED FEATURES TESTS
# =====================================================================

# ----------------- RareCategoriesGrouper -----------------

def test_rare_categories_grouper_success():
    """Verify that categories are dynamically preserved to keep 'Other' below the max ratio.

    GIVEN: a DataFrame with a categorical column where 'A' and 'B' represent 40% each,
           and 'C' represents 20% of the dataset, with custom indices
    WHEN: fit and transform are sequentially executed with max_other_ratio=0.25 
          (meaning we must keep at least 75% of the data)
    THEN: 'A' and 'B' are preserved (covering 80% of the data), and 'C' is replaced 
          with 'Other' (representing 20% of the data, which is safely below 25%), 
          preserving the index
    """
    df_mock = pd.DataFrame(
        {"Breed": ["A", "A", "B", "B", "C"]}, index=[10, 20, 30, 40, 50]
    )
    expected = pd.Series(
        ["A", "A", "B", "B", "Other"], index=[10, 20, 30, 40, 50], name="Breed"
    )

    grouper = RareCategoriesGrouper(columns=["Breed"], max_other_ratio=0.25)
    grouper.fit(df_mock)
    df_clean = grouper.transform(df_mock)

    pd.testing.assert_series_equal(df_clean["Breed"], expected)


def test_rare_categories_grouper_logs_debug_on_drift(caplog):
    """Verify that transform emits a DEBUG log when the proportion of
    'Other' exceeds the configured max ratio due to data drift.

    GIVEN: a trained grouper, and a new test dataset containing only rare categories
           (which will all be mapped to 'Other', resulting in 100% 'Other' ratio)
    WHEN: transform is executed on the test dataset
    THEN: a DEBUG log is emitted indicating the ratio exceeds threshold
    """
    df_train = pd.DataFrame({"Breed": ["A"] * 90 + ["B"] * 10})
    df_test = pd.DataFrame({"Breed": ["B"] * 100})

    grouper = RareCategoriesGrouper(columns=["Breed"], max_other_ratio=0.15)
    grouper.fit(df_train)

    with caplog.at_level(logging.DEBUG):
        grouper.transform(df_test)

    assert "exceeds the configured max_other_ratio" in caplog.text


def test_rare_categories_grouper_raises_value_error_on_missing_fit_column():
    """Verify that RareCategoriesGrouper raises a ValueError if a required column is missing during fit.

    GIVEN: a DataFrame missing a required column specified in the grouper's target configuration
    WHEN: the fit method is executed
    THEN: it must immediately raise a ValueError indicating the required column is missing from the training DataFrame
    """
    df_mock = pd.DataFrame({"Color": ["Black", "White"]})  # 'Breed' is missing
    grouper = RareCategoriesGrouper(columns=["Breed"])
    
    with pytest.raises(ValueError, match="missing from the training DataFrame during fit"):
        grouper.fit(df_mock)

def test_rare_categories_grouper_raises_runtime_error_when_unfitted():
    """Verify that RareCategoriesGrouper raises a RuntimeError if transform is called before fit.

    GIVEN: a RareCategoriesGrouper instance that has not been fitted yet
    WHEN: the transform method is executed on a DataFrame
    THEN: it must immediately raise a RuntimeError indicating the instance is not fitted
    """
    df_mock = pd.DataFrame({"Breed": ["Labrador Retriever", "Poodle"]})
    grouper = RareCategoriesGrouper(columns=["Breed"])

    with pytest.raises(
        RuntimeError,
        match="RareCategoriesGrouper instance is not fitted. Call 'fit' before 'transform'.",
    ):
        grouper.transform(df_mock)

def test_rare_categories_grouper_preserves_nan():
    """Verify that NaN values are untouched and not converted to the 'Other'
       placeholder.

    GIVEN: a DataFrame with a categorical column containing missing NaN values
    WHEN: fit and transform are executed
    THEN: NaN values remain as NaN, preventing them from being grouped into
    'Other'
    """

    df_mock = pd.DataFrame(
        {"Breed": ["A", "A", "B", "B", np.nan]}, index=[10, 20, 30, 40, 50]
    )
    expected = pd.Series(
        ["A", "A", "B", "B", np.nan], index=[10, 20, 30, 40, 50], name="Breed"
    )

    grouper = RareCategoriesGrouper(columns=["Breed"], max_other_ratio=0.25)
    grouper.fit(df_mock)
    df_clean = grouper.transform(df_mock)

    pd.testing.assert_series_equal(df_clean["Breed"], expected)


def test_rare_categories_grouper_fit_empty_column(caplog):
    """Verify that RareCategoriesGrouper handles completely empty/NaN columns during fit safely with a warning.

    GIVEN: a training DataFrame containing only missing values (NaN) in the target column
    WHEN: the fit method is executed with WARNING logging level captured
    THEN: a warning log is recorded, the execution doesn't crash, and the frequent categories list is set to empty
    """

    df_mock = pd.DataFrame({"Breed": [np.nan, np.nan]})
    
    grouper = RareCategoriesGrouper(columns=["Breed"])
    
    with caplog.at_level(logging.WARNING):
        grouper.fit(df_mock)
        
    assert "is empty or contains only NaNs" in caplog.text
    assert grouper.frequent_categories_["Breed"] == []


def test_rare_categories_grouper_missing_column_transform():
    """Verify that RareCategoriesGrouper returns the DataFrame untouched if the target column is missing during transform.

    GIVEN: a fitted RareCategoriesGrouper and a test DataFrame lacking the target column
    WHEN: transform is executed
    THEN: the input DataFrame is returned completely unmodified, preserving its index and values
    """
    df_train = pd.DataFrame({"Breed": ["A", "B"]})
    df_test = pd.DataFrame({"Color": ["Black", "White"]}, index=[10, 20])

    grouper = RareCategoriesGrouper(columns=["Breed"]).fit(df_train)
    df_transformed = grouper.transform(df_test)

    pd.testing.assert_frame_equal(df_transformed, df_test)


def test_rare_categories_grouper_transform_empty_dataframe():
    """Verify that transform returns an empty DataFrame untouched if the input is empty.

    GIVEN: a properly fitted RareCategoriesGrouper instance and a completely empty test DataFrame schema
    WHEN: the transform method is executed
    THEN: the input DataFrame is returned completely unmodified, preserving its empty structure
    """
    df_train = pd.DataFrame({"Breed": ["A", "B"]})
    df_empty = pd.DataFrame(columns=["Breed"])
    
    grouper = RareCategoriesGrouper(columns=["Breed"])
    grouper.fit(df_train)
    
    result = grouper.transform(df_empty)
    
    assert result.empty

# ----------------- CategoricalFeaturesEngineer -----------------

def test_categorical_features_engineer_success():
    """Verify that CategoricalFeaturesEngineer correctly extracts 'is_mix' and
       groups rare labels.

    GIVEN: a DataFrame with Breed and Color columns, with custom index
           (max_other_ratio=0.25)
    WHEN: fit and transform are sequentially executed
    THEN: 'is_mix' is extracted, Breed and Color are converted to primary, and
          rare categories are successfully grouped, preserving the index
     """
    df_mock = pd.DataFrame(
        {
            "Breed": ["A/B", "A Mix", "A", "A", "C", np.nan],
            "Color": ["Black/White", "Black", "Black", "Black", "Blue", np.nan],
        },
        index=[10, 20, 30, 40, 50, 60],
    )

    expected_is_mix = pd.Series(
            [1, 1, 0, 0, 0, 0], index=[10, 20, 30, 40, 50, 60], name="is_mix"
    )
    expected_breed = pd.Series(
            ["A", "A", "A", "A", "Other", np.nan], index=[10, 20, 30, 40, 50, 60], name="Breed"
    )
    expected_color = pd.Series(
            ["Black", "Black", "Black", "Black", "Other", np.nan],
            index=[10, 20, 30, 40, 50, 60],
            name="Color",
    )
    
    engineer = CategoricalFeaturesEngineer(
            columns=["Breed", "Color"], max_other_ratio=0.25
        )
    df_clean = engineer.fit_transform(df_mock)
    
    pd.testing.assert_series_equal(df_clean["is_mix"], expected_is_mix)
    pd.testing.assert_series_equal(df_clean["Breed"], expected_breed)
    pd.testing.assert_series_equal(df_clean["Color"], expected_color)


def test_categorical_features_engineer_raises_runtime_error_when_unfitted():
    """Verify that CategoricalFeaturesEngineer raises a RuntimeError if transform is called before fit.

    GIVEN: a CategoricalFeaturesEngineer instance that has not been fitted yet
    WHEN: the transform method is executed on a DataFrame
    THEN: it must immediately raise a RuntimeError indicating the instance is not fitted
    """
    df_mock = pd.DataFrame({"Breed": ["A", "B"], "Color": ["Black", "White"]})
    engineer = CategoricalFeaturesEngineer()
    
    with pytest.raises(RuntimeError, match="instance is not fitted. Call 'fit' before 'transform'"):
        engineer.transform(df_mock)


def test_categorical_features_engineer_missing_columns():
    """Verify that CategoricalFeaturesEngineer returns the DataFrame untouched
       when both Breed and Color columns are missing.

    GIVEN: a DataFrame that lacks both 'Breed' and 'Color' target columns,
           with a custom index
    WHEN: fit_transform is executed on CategoricalFeaturesEngineer
    THEN: the input DataFrame is returned completely unmodified, preserving its
          index and values
    """

    df_mock = pd.DataFrame({"Name": ["Bella", "Max"]}, index=[100, 200])

    engineer = CategoricalFeaturesEngineer(
        columns=["Breed", "Color"], max_other_ratio=0.25
    )
    df_transformed = engineer.fit_transform(df_mock)

    pd.testing.assert_frame_equal(df_transformed, df_mock)


def test_categorical_features_engineer_fit_empty_column(caplog):
    """Verify that CategoricalFeaturesEngineer handles completely empty/NaN columns during fit safely with a warning.

    GIVEN: a training DataFrame containing only missing values (NaN) in target columns
    WHEN: fit is executed with WARNING logging level captured
    THEN: warning logs are recorded, execution doesn't crash, and columns are processed safely
    """
    df_mock = pd.DataFrame({"Breed": [np.nan, np.nan], "Color": [np.nan, np.nan]})
    engineer = CategoricalFeaturesEngineer(columns=["Breed", "Color"])

    with caplog.at_level(logging.WARNING):
        engineer.fit(df_mock)

    assert "is empty or contains only NaNs" in caplog.text

def test_categorical_features_engineer_empty_dataframe_with_columns():
    """Verify that CategoricalFeaturesEngineer handles empty DataFrames safely without crashing.

    GIVEN: a fitted CategoricalFeaturesEngineer and an empty DataFrame with valid schema
    WHEN: transform is executed
    THEN: the returned DataFrame is empty, and the expected schema ('Breed', 'Color', 'is_mix') is created
    """
    df_train = pd.DataFrame({"Breed": ["Labrador Mix"], "Color": ["Black/White"]})
    df_empty = pd.DataFrame(columns=["Breed", "Color"])

    engineer = CategoricalFeaturesEngineer(columns=["Breed", "Color"]).fit(df_train)
    df_transformed = engineer.transform(df_empty)

    assert df_transformed.empty
    assert set(df_transformed.columns) == {"Breed", "Color", "is_mix"}

def test_categorical_features_engineer_logs_debug_on_drift(caplog):
    """Verify debug log emission on data drift.

    GIVEN: a fitted CategoricalFeaturesEngineer instance, and a test DataFrame experiencing data drift
           where rare categories dominate
    WHEN: transform is executed on the test dataset with DEBUG logging level captured
    THEN: a DEBUG log message is recorded indicating that the 'Other' category ratio exceeds max_other_ratio
    """
    df_train = pd.DataFrame(
        {"Breed": ["A"] * 90 + ["B"] * 10, "Color": ["Black"] * 100}
    )
    df_test = pd.DataFrame({"Breed": ["B"] * 100, "Color": ["Black"] * 100})

    engineer = CategoricalFeaturesEngineer(
        columns=["Breed", "Color"], max_other_ratio=0.15
    )
    engineer.fit(df_train)

    with caplog.at_level(logging.DEBUG):
        engineer.transform(df_test)

    assert "exceeds the configured max_other_ratio" in caplog.text


def test_categorical_features_engineer_missing_column_transform():
    """Verify transform behavior when one fitted column is missing during inference.

    GIVEN: a CategoricalFeaturesEngineer fitted on both 'Breed' and 'Color', and a test DataFrame 
           missing the 'Breed' column
    WHEN: transform is executed on the test DataFrame
    THEN: 'is_mix' extraction is safely skipped, while the present 'Color' column is processed 
          and returned without crashing
    """
    df_train = pd.DataFrame(
        {"Breed": ["Labrador"], "Color": ["Black"]}, index=[1, 2]
    )
    df_test = pd.DataFrame({"Color": ["Black"]}, index=[10, 20])

    engineer = CategoricalFeaturesEngineer(columns=["Breed", "Color"]).fit(
        df_train
    )
    df_transformed = engineer.transform(df_test)

    assert "is_mix" not in df_transformed.columns
    assert "Color" in df_transformed.columns