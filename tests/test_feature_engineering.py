import numpy as np
import pandas as pd
import pytest
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
#                       TEMPORAL FEATURE
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

    df_transformed = TemporalFeaturesExtractor().transform(df_mock)

    assert "DateTime" not in df_transformed.columns
    assert set(df_transformed.columns) == expected_columns

    pd.testing.assert_series_equal(df_transformed["IsWeekend"], expected_weekend, check_dtype=False)


def test_temporal_cyclic_hours():
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

   df_transformed = TemporalFeaturesExtractor().transform(df_mock)

   pd.testing.assert_series_equal(df_transformed["Hour_sin"], expected_sin, check_exact=False, atol=1e-7)
   pd.testing.assert_series_equal(df_transformed["Hour_cos"], expected_cos, check_exact=False, atol=1e-7)


def test_temporal_cyclic_weekdays():
     """Verify the mathematical correctness of sine and cosine transformations for weekdays.

    GIVEN: a DataFrame containing a Monday (weekday 0) and a Sunday (weekday 6) with custom non-default indices
    WHEN: the transform method of TemporalFeaturesExtractor is executed
    THEN: Wday_sin and Wday_cos are computed accurately based on the weekday, preserving the index
          and handling float precision tolerances
    """
     df_mock = pd.DataFrame({"DateTime": ["2026-07-06", "2026-07-12"]}, index=[15, 25])  # Lunedì, Domenica
     expected_sin = pd.Series(np.sin(two_pi * np.array([0, 6]) / 7), index=[15, 25], name="Wday_sin")
     expected_cos = pd.Series(np.cos(two_pi * np.array([0, 6]) / 7), index=[15, 25], name="Wday_cos")

     df_transformed = TemporalFeaturesExtractor().transform(df_mock)

     pd.testing.assert_series_equal(df_transformed["Wday_sin"], expected_sin, check_exact=False, atol=1e-7)
     pd.testing.assert_series_equal(df_transformed["Wday_cos"], expected_cos, check_exact=False, atol=1e-7)


def test_temporal_cyclic_day_of_year():
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

    df_transformed = TemporalFeaturesExtractor().transform(df_mock)

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
    
    res_strings = TemporalFeaturesExtractor().transform(df_strings)
    res_datetime = TemporalFeaturesExtractor().transform(df_datetime)
    
    pd.testing.assert_frame_equal(res_strings, res_datetime)


def test_temporal_extractor_empty_dataframe_with_columns():
    """Verify that an empty DataFrame with the target column is processed safely without crashing.

    GIVEN: an empty DataFrame with only the DateTime column in its schema
    WHEN: transform is executed on TemporalFeaturesExtractor
    THEN: the returned DataFrame is empty, DateTime is dropped, and the expected empty schema is preserved
    """
    df_mock = pd.DataFrame(columns=["DateTime"])

    expected_cols = {"IsWeekend", "Hour_sin", "Hour_cos", "Wday_sin", "Wday_cos", "DoY_sin", "DoY_cos"}
    
    df_transformed = TemporalFeaturesExtractor().transform(df_mock)
    
    assert df_transformed.empty
    assert set(df_transformed.columns) == expected_cols

def test_temporal_extractor_handles_null_values():
    """Verify that missing or invalid date values do not crash the execution and propagate as NaN.

    GIVEN: a DataFrame with a valid DateTime and a None value, under custom indices
    WHEN: transform is executed on TemporalFeaturesExtractor
    THEN: the valid date is calculated, and the missing value safely propagates as NaN in the cyclic columns
    """
    df_mock = pd.DataFrame({"DateTime": ["2026-07-12 12:00:00", None]}, index=[101, 102])
    
    df_transformed = TemporalFeaturesExtractor().transform(df_mock)
    

    assert not np.isnan(df_transformed["Hour_sin"].loc[101])

    assert np.isnan(df_transformed["Hour_sin"].loc[102])
    assert np.isnan(df_transformed["Wday_cos"].loc[102])
    assert np.isnan(df_transformed["DoY_sin"].loc[102])
    
    assert df_transformed["IsWeekend"].loc[101] == 1.0
    assert np.isnan(df_transformed["IsWeekend"].loc[102])



def test_temporal_base_extractor_missing_column():
    """Verify that the base transformer returns the DataFrame unchanged when
    the target column is missing.

    GIVEN: a DataFrame that lacks the specified target DateTime column, with a
    custom index
    WHEN: the transform method is executed
    THEN: the input DataFrame is returned completely unmodified, preserving its
    index and values
    """

    df_mock = pd.DataFrame({"Name": ["Bella", "Max"]}, index=[100, 200])

    df_transformed = TemporalFeaturesExtractor(datetime_col="DateTime").transform(
        df_mock
    )

    pd.testing.assert_frame_equal(df_transformed, df_mock)

# =====================================================================
#                           COLOR AND BREED 
# =====================================================================


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

def test_rare_categories_grouper_dynamic_preservation():
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


def test_rare_categories_grouper_emits_warning_on_drift():
    """Verify that transform emits a RuntimeWarning when the proportion of
       'Other' exceeds the configured max ratio due to data drift.

    GIVEN: a trained grouper, and a new test dataset containing only rare
    categories
           (which will all be mapped to 'Other', resulting in 100% 'Other' ratio)
    WHEN: transform is executed on the test dataset
    THEN: a RuntimeWarning is correctly emitted, indicating a dataset shift
    """
    df_train = pd.DataFrame({"Breed": ["A"] * 90 + ["B"] * 10})

    df_test = pd.DataFrame({"Breed": ["B"] * 100})


    grouper = RareCategoriesGrouper(columns=["Breed"], max_other_ratio=0.15)
    grouper.fit(df_train)

    with pytest.warns(
        RuntimeWarning, match="exceeds the configured max_other_ratio"
    ):
        grouper.transform(df_test)



def test_categorical_features_engineer_fit_transform():
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


# =====================================================================
#                         Sex
# =====================================================================
def test_sex_features_extractor_reproductive_status():
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
    

    df_transformed = SexFeaturesExtractor().transform(df_mock)
    
    assert "SexuponOutcome" not in df_transformed.columns
    
    pd.testing.assert_series_equal(df_transformed["Reproductive_Status"], expected)


  
def test_sex_features_extractor_missing_column():
    """Verify that SexFeaturesExtractor returns the DataFrame untouched when the target column is missing.

    GIVEN: a DataFrame that does not contain the required SexuponOutcome column,
           with a custom index
    WHEN: transform is executed on SexFeaturesExtractor
    THEN: the input DataFrame is returned completely unmodified, preserving its
          index and values
    """
  
    df_mock = pd.DataFrame({"Name": ["Bella", "Max"]}, index=[100, 200])

    df_transformed = SexFeaturesExtractor(sex_col="SexuponOutcome").transform(
        df_mock
    )

    pd.testing.assert_frame_equal(df_transformed, df_mock)


# =====================================================================
#                         Name
# =====================================================================
def test_name_features_extractor_presence():
    """Verify that NameFeaturesExtractor correctly creates 'has_name' and drops
       the raw column.

    GIVEN: a DataFrame with a Name column containing a valid name, whitespace,
           NaN, and variations of "Unknown" with custom indices
    WHEN: transform is executed on NameFeaturesExtractor
    THEN: the raw Name column is dropped, and 'has_name' is 1 only for valid names
          and 0 for empty/unknown entries, preserving the original index
    """
    # 1. GIVEN: Prepariamo l'input sporco con nomi validi, spazi, NaN e varianti di Unknown [2, 21]
    df_mock = pd.DataFrame(
        {"Name": ["Bella", "   ", np.nan, "Unknown", "UNKNOWN"]},
        index=[10, 20, 30, 40, 50],
    )
    expected = pd.Series(
        [1, 0, 0, 0, 0], index=[10, 20, 30, 40, 50], name="has_name"
    )

    df_transformed = NameFeaturesExtractor().transform(df_mock)


    assert "Name" not in df_transformed.columns

    pd.testing.assert_series_equal(
        df_transformed["has_name"], expected, check_dtype=False
    )


def test_name_features_extractor_missing_column():
    """Verify that NameFeaturesExtractor returns the DataFrame untouched when

    the Name column is missing.

    GIVEN: a DataFrame that does not contain the required Name column, with a
    custom index
    WHEN: transform is executed on NameFeaturesExtractor
    THEN: the input DataFrame is returned completely unmodified, preserving values
    """
    df_mock = pd.DataFrame({"Age": [10, 20]}, index=[100, 200])

    df_transformed = NameFeaturesExtractor(name_col="Name").transform(df_mock)

    pd.testing.assert_frame_equal(df_transformed, df_mock)