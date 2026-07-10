import numpy as np
import pandas as pd
from src.feature_engineering import AdvancedTemporalFeaturesExtractor


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



def test_advanced_temporal_missing_parent_columns_failsafe():
    """Verify that AdvancedTemporalFeaturesExtractor exits early if base columns (Hour/Weekday) are missing.

    GIVEN: a DataFrame that bypasses the parent's datetime_col check but where 
           Hour or Weekday are dropped (or missing)
    WHEN: transform is executed on the subclass
    THEN: the transform method exits early, returning the DataFrame untouched without raising a KeyError
    """

    df_mock = pd.DataFrame({"Name": ["Bella", "Max"]}, index=[100, 200])

    df_transformed = AdvancedTemporalFeaturesExtractor(datetime_col="MissingDateTime").transform(df_mock)

    pd.testing.assert_frame_equal(df_transformed, df_mock)