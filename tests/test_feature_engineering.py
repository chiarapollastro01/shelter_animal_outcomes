import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import (
    AdvancedTemporalFeaturesExtractor,
    CategoricalFeaturesEngineer,
    RareCategoriesGrouper,
    extract_primary_breed,
    extract_primary_color,
    SexFeaturesExtractor,
    NameFeaturesExtractor
)
# =====================================================================
#                       TEMPORAL FEATURE
# =====================================================================
def test_temporal_weekend_boundaries():
    """Verify that IsWeekend correctly identifies the weekday/weekend boundary.

    GIVEN: a DataFrame with explicit dates containing a Monday, a Friday, a
    Saturday, and a Sunday 
    WHEN: the transform method of
    TemporalFeaturesExtractor is executed 
    THEN: IsWeekend returns 0 for Monday
    and Friday, and 1 for Saturday and Sunday,
          preserving the custom index
    """

    df_mock = pd.DataFrame(
        {
            "DateTime": [
                "2026-07-06 12:00:00",  
                "2026-07-10 12:00:00",  
                "2026-07-11 12:00:00",  
                "2026-07-12 12:00:00",  
            ]
        },
        index=[10, 20, 30, 40],
    )
    expected = pd.Series([0, 0, 1, 1], index=[10, 20, 30, 40], name="IsWeekend")

    df_transformed = AdvancedTemporalFeaturesExtractor().transform(df_mock)

    pd.testing.assert_series_equal(df_transformed["IsWeekend"], expected)



def test_temporal_time_of_day_operational_binning():
    """Verify that hours are correctly classified into the 4 operational

    shelter categories.

    GIVEN: a DataFrame containing timestamps exactly at the transition hours
           (0, 5, 6, 11, 12, 16, 17, 23)
    WHEN: the transform method is executed
    THEN: TimeOfDay classifies boundaries correctly into Night, Morning,
    Afternoon, or Evening
    """
    df_mock = pd.DataFrame(
        {
            "DateTime": [
                "2026-07-09 00:00:00",  
                "2026-07-09 05:00:00",  
                "2026-07-09 06:00:00",  
                "2026-07-09 11:00:00",  
                "2026-07-09 12:00:00",  
                "2026-07-09 16:00:00",  
                "2026-07-09 17:00:00",  
                "2026-07-09 23:00:00",  
            ]
        },
        index=[100, 101, 102, 103, 104, 105, 106, 107],
    )
    expected = pd.Series(
        ["Night", "Night", "Morning", "Morning", "Afternoon", "Afternoon", "Evening", "Evening"],
        index=[100, 101, 102, 103, 104, 105, 106, 107],
        name="TimeOfDay",
    )

    df_transformed = AdvancedTemporalFeaturesExtractor().transform(df_mock)

    pd.testing.assert_series_equal(df_transformed["TimeOfDay"], expected)


def test_temporal_features_kitten_season_conditional():
    """Verify that IsKittenSeason is only extracted when explicitly enabled.

    GIVEN: a DataFrame with a DateTime in July (Kitten Season)
    WHEN: transform is called with add_kitten_season=True and
    add_kitten_season=False
    THEN: IsKittenSeason is correctly calculated in the first case, and absent
    in the second
    """
    df_mock = pd.DataFrame({"DateTime": ["2026-07-09"]}, index=[10])

    df_enabled = AdvancedTemporalFeaturesExtractor(
        add_kitten_season=True
    ).transform(df_mock)
    assert "IsKittenSeason" in df_enabled.columns
    assert df_enabled.loc[10, "IsKittenSeason"] == 1

 
    df_disabled = AdvancedTemporalFeaturesExtractor(
        add_kitten_season=False
    ).transform(df_mock)
    assert "IsKittenSeason" not in df_disabled.columns


def test_temporal_features_kitten_season_boundaries():
    """Verify the exact day-of-year boundary transitions for Kitten Season.

    GIVEN: a DataFrame with dates exactly at the boundaries of April (91) 
           and October (304) in a non-leap year (2026)
    WHEN: transform is called with add_kitten_season=True
    THEN: IsKittenSeason is 1 from April 1 to Oct 31, and 0 otherwise, preserving the index
    """
    df_mock = pd.DataFrame(
        {
            "DateTime": [
                "2026-03-31",  
                "2026-04-01",  
                "2026-10-31",  
                "2026-11-01", 
            ]
        },
        index=[11, 22, 33, 44]
    )
    expected = pd.Series([0, 1, 1, 0], index=[11, 22, 33, 44], name="IsKittenSeason")

    df_transformed = AdvancedTemporalFeaturesExtractor(add_kitten_season=True).transform(df_mock)

    pd.testing.assert_series_equal(df_transformed["IsKittenSeason"], expected)



def test_advanced_temporal_missing_parent_columns():
    """Verify that AdvancedTemporalFeaturesExtractor exits early if base columns (Hour/Weekday) are missing.

    GIVEN: a DataFrame that bypasses the parent's datetime_col check but where 
           Hour or Weekday are dropped (or missing)
    WHEN: transform is executed on the subclass
    THEN: the transform method exits early, returning the DataFrame untouched without raising a KeyError
    """

    df_mock = pd.DataFrame({"Name": ["Bella", "Max"]}, index=[100, 200])

    df_transformed = AdvancedTemporalFeaturesExtractor(datetime_col="MissingDateTime").transform(df_mock)

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