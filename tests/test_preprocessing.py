"""
Unit tests for the preprocessing module (extract_age_in_days, DataCleaner).

"""
import logging
import numpy as np
import pytest
import pandas as pd
from src.preprocessing import extract_age_in_days, DataCleaner

# =====================================================================
#                              FIXTURES 
# =====================================================================

@pytest.fixture
def train_df() -> pd.DataFrame:
    """Training frame with known statistics: sex mode = 'Neutered Male',
    median age = 730 days. Covers every column the cleaner touches."""
    return pd.DataFrame(
        {
            "AnimalID": ["A1", "A2", "A3", "A4"],
            "DateTime": ["2026-01-01 10:00:00", "2026-01-02 11:00:00", "2026-01-03 12:00:00", "2026-01-04 13:00:00"],
            "Name": ["Bella", np.nan, "Luna", "Max"],
            "Breed": ["Chihuahua Mix", np.nan, "Beagle", "Beagle"],
            "Color": [np.nan, "Black", "White", "Black"],
            "SexuponOutcome": ["Neutered Male", "Neutered Male", "Intact Female", np.nan],
            "AgeuponOutcome": ["1 year", "2 years", "3 years", np.nan],
        },
        index=[10, 20, 30, 40],
    )


@pytest.fixture
def fitted_cleaner(train_df: pd.DataFrame) -> DataCleaner:
    """A DataCleaner already fitted on the training fixture."""
    return DataCleaner().fit(train_df)

# =====================================================================
#                      AGE EXTRACTION TESTS
# =====================================================================

def test_extract_age_typical_cases():
    """
    GIVEN: a pandas Series with well-formatted, lowercase age strings (years, months, weeks, days) with custom indices
    WHEN: the extract_age_in_days function is executed
    THEN: the numeric values are correctly converted to float days and the index is preserved
    """
    age_series = pd.Series(["1 year", "2 months", "3 weeks", "5 days"], index=[101, 102, 103, 104])
    
    expected = pd.Series([365.0, 60.0, 21.0, 5.0], index=[101, 102, 103, 104])
    
    result = extract_age_in_days(age_series)

        # check_names=False: the function does not set a name on the output Series
    pd.testing.assert_series_equal(result, expected, check_names=False)
  

def test_extract_age_formatting_variations():
    """
    GIVEN: a Series with uppercase units, singular terms, and leading/trailing spaces with custom indices
    WHEN: the extract_age_in_days function is executed
    THEN: values are parsed correctly, ignoring case and surrounding spacing, preserving the index
    """
    age_series = pd.Series(["1 YEAR", "  2 months  ", "1 day", "10 DAYS"], index=[10, 20, 30, 40])

    expected = pd.Series([365.0, 60.0, 1.0, 10.0], index=[10, 20, 30, 40])
    
    result = extract_age_in_days(age_series)
    
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_extract_age_invalid_and_null_inputs():
    """
    GIVEN: a Series containing a missing value (NaN), an unparseable string ("Unknown") and a number with an invalid unit, with custom indices
    WHEN: the extract_age_in_days function is executed
    THEN: all inputs safely resolve to NaN without breaking the execution, preserving the index
    """
    age_series = pd.Series([np.nan, "Unknown", "5 units"], index=[7,8,9])
    expected = pd.Series([np.nan, np.nan, np.nan], index=[7,8,9])
    
    result = extract_age_in_days(age_series)
    
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_extract_age_all_null_series():
    """

    GIVEN: a Series where all elements are NaN with custom indeces (early-return branch)
    WHEN: the extract_age_in_days function is executed
    THEN: it returns a Series of the same length containing NaN with a float dtype, preserving the index
    """
    age_series = pd.Series([np.nan, np.nan], index=[10, 20])
    expected = pd.Series([np.nan, np.nan], index=[10, 20], dtype=float)
    
    result = extract_age_in_days(age_series)
    
    pd.testing.assert_series_equal(result, expected)

def test_extract_age_empty_series():
    """
    GIVEN: a completely empty pandas Series (length 0) 
    WHEN: the extract_age_in_days function is executed
    THEN: an empty Series is returned immediately, preserving the empty structure and float dtype
    """

    empty_series = pd.Series([], dtype=object)
    expected = pd.Series([], dtype=float)
    
    result = extract_age_in_days(empty_series)
    
    assert result.empty
    pd.testing.assert_series_equal(result, expected)


# =====================================================================
#                       DATA CLEANER TESTS
# =====================================================================
def test_datacleaner_fit_returns_self(train_df):
    """
    GIVEN: an unfitted DataCleaner
    WHEN: fit is executed
    THEN: the same instance is returned
    """
    cleaner = DataCleaner()
    assert cleaner.fit(train_df) is cleaner

def test_datacleaner_custom_columns_to_remove(train_df):
    """
    GIVEN: a DataCleaner initialised with a custom columns_to_remove list
    WHEN: fit and transform are executed
    THEN: only the requested columns are dropped 
    """
    cleaner = DataCleaner(columns_to_remove=["Name"])
    df_clean = cleaner.fit(train_df).transform(train_df)

    assert "Name" not in df_clean.columns
    assert "AnimalID" in df_clean.columns 


def test_datacleaner_column_dropping(fitted_cleaner, train_df):
    """
    GIVEN: a fitted cleaner and a frame containing AnimalID
    WHEN: transform is executed
    THEN: AnimalID and AgeuponOutcome are removed, log_age_in_days is added
    """

    df_clean = fitted_cleaner.transform(train_df)

    assert "AnimalID" not in df_clean.columns
    assert "AgeuponOutcome" not in df_clean.columns
    assert "log_age_in_days" in df_clean.columns


def test_datacleaner_prevents_data_leakage(train_df):
    """
    GIVEN: a training DataFrame with specific median age (2 years = 730 days) and sex mode (Neutered Male)
           and a test DataFrame with missing values
    WHEN: fit is called on train, and transform is called on test
    THEN: the test DataFrame is imputed using train statistics, completely preventing data leakage
    """

    df_test = pd.DataFrame(
        {"SexuponOutcome": [np.nan], "AgeuponOutcome": [np.nan]}, index=[99]
    )

    cleaner = DataCleaner().fit(train_df)
    df_test_clean = cleaner.transform(df_test)

    assert df_test_clean["SexuponOutcome"].iloc[0] == "Neutered Male"
    assert np.isclose(
        df_test_clean["log_age_in_days"].iloc[0], np.log1p(730.0), atol=1e-7
    )


def test_datacleaner_raises_runtime_error_if_unfitted():
    """
    GIVEN: a DataCleaner instance that has not been fitted
    WHEN: transform is executed
    THEN: a RuntimeError is raised with an informative error message
    """
    cleaner = DataCleaner()

    with pytest.raises(RuntimeError, match="not fitted"):
        cleaner.transform(pd.DataFrame({"Name": ["Bella"]}))


def test_datacleaner_name_imputation(fitted_cleaner, train_df):
    """
    GIVEN: a DataFrame with some null values (NaN) in the Name column
    WHEN: transform is executed
    THEN: all missing values in the Name column are replaced with "Unknown",
    preserving the original index
    """
    df_clean = fitted_cleaner.transform(train_df)

    assert df_clean["Name"].isnull().sum() == 0
    assert df_clean.loc[20, "Name"] == "Unknown"


def test_datacleaner_sex_imputation(fitted_cleaner, train_df):
    """
    GIVEN: a DataFrame with a missing value in SexuponOutcome where "Neutered
    Male" is the mode
    WHEN: transform is executed
    THEN: the missing value is replaced by the mode "Neutered Male", preserving
    the index
    """
    df_clean = fitted_cleaner.transform(train_df)

    assert df_clean.loc[40, "SexuponOutcome"] == "Neutered Male"
   

def test_datacleaner_age_imputation(fitted_cleaner, train_df):
    """
    GIVEN: a DataFrame with AgeuponOutcome values whose valid median in days is 730.0
    WHEN: transform is executed
    THEN: AgeuponOutcome is converted, dropped, its NaNs are filled with the median (730.0),
          and the resulting column is log-transformed, preserving the index
    """
    expected_days = np.array([365.0, 730.0, 1095.0, 730.0])
    expected = pd.Series(
        np.log1p(expected_days), index=[10, 20, 30, 40], name="log_age_in_days"
    )

    df_clean = fitted_cleaner.transform(train_df)

    pd.testing.assert_series_equal(
        df_clean["log_age_in_days"], expected, check_exact=False, atol=1e-7
    )

def test_datacleaner_age_log_transformation():
    """
    GIVEN: a DataFrame with valid AgeuponOutcome entries ("0 days", "7 days") 
           that do not need imputation
    WHEN: fit and transform are executed
    THEN: the resulting log_age_in_days column contains mathematically correct 
          log1p values, and the original column is dropped, preserving index (covers the no-missing-age branch)
    """
    
    df_mock = pd.DataFrame({"AgeuponOutcome": ["0 days", "7 days"]}, index=[10, 20])
    
    
    expected = pd.Series([np.log1p(0.0), np.log1p(7.0)], index=[10, 20], name="log_age_in_days")

    df_clean = DataCleaner().fit(df_mock).transform(df_mock)

    pd.testing.assert_series_equal(df_clean["log_age_in_days"], expected, check_exact=False, atol=1e-7)


def test_datacleaner_breed_color_preventive_imputation(fitted_cleaner, train_df):
    """
    GIVEN: a DataFrame containing missing values (NaN) in both Breed and Color columns,
           with custom non-default indices
    WHEN: transform is executed
    THEN: all missing values (NaN) in both columns are successfully replaced with "Unknown",
          preserving the original index
    """
    df_clean = fitted_cleaner.transform(train_df)

    assert df_clean.loc[20, "Breed"] == "Unknown"
    assert df_clean.loc[10, "Color"] == "Unknown"



def test_datacleaner_fit_without_sex_and_age_columns():
    """
    GIVEN: a DataFrame without SexuponOutcome and AgeuponOutcome 
    WHEN: fit is executed
    THEN: safe fallbacks are learned ('Unknown', 0.0) (covers the
          missing-column branches of fit)
    """
    df_mock = pd.DataFrame({"Breed": ["Beagle"], "Color": ["Black"]})

    cleaner = DataCleaner().fit(df_mock)

    assert cleaner.sex_mode_ == "Unknown"
    assert cleaner.age_median_ == 0.0


def test_datacleaner_transform_without_sex_and_age_columns(fitted_cleaner):
    """
    GIVEN: a fitted cleaner and a DataFrame without SexuponOutcome and AgeuponOutcome
    WHEN: transform is executed
    THEN: log_age_in_days column isn't created and nothing raises (covers
          the missing-column branches of transform)
    """
    df_mock = pd.DataFrame({"Name": ["Bella", np.nan]}, index=[1, 2])

    df_clean = fitted_cleaner.transform(df_mock)

    assert "log_age_in_days" not in df_clean.columns
    assert df_clean.loc[2, "Name"] == "Unknown"


def test_datacleaner_all_nan():
    """
    GIVEN: a DataFrame where all elements are missing
    WHEN: fit and transform are executed
    THEN: Name and SexuponOutcome fall back to "Unknown", and log_age_in_days
          is safely imputed with 0.0, preserving the original index (covers the empty-mode and
          empty-median fallback branches)
    """
    df_mock = pd.DataFrame(
        {
            "Name": [np.nan, np.nan],
            "SexuponOutcome": [np.nan, np.nan],
            "AgeuponOutcome": [np.nan, np.nan],
        },
        index=[100, 200],
    )
    expected = pd.DataFrame(
        {
            "Name": ["Unknown", "Unknown"],
            "SexuponOutcome": ["Unknown", "Unknown"],
            "log_age_in_days": [0.0, 0.0], 
        },
        index=[100, 200],
    )

    df_clean = DataCleaner().fit(df_mock).transform(df_mock)

    pd.testing.assert_frame_equal(df_clean, expected)


def test_datacleaner_empty_dataframe():
    """
    GIVEN: an empty DataFrame with valid column headers
    WHEN: fit and transform are executed
    THEN: the output DataFrame is empty, schema is preserved, and default fallback states are learned
    """
    df_mock = pd.DataFrame(columns=["Name", "SexuponOutcome", "AgeuponOutcome"])
    
    cleaner = DataCleaner().fit(df_mock)
    df_clean=cleaner.transform(df_mock)

    assert cleaner.sex_mode_ == "Unknown"
    assert cleaner.age_median_ == 0.0    
    assert df_clean.empty
    assert "log_age_in_days" in df_clean.columns
    assert "AgeuponOutcome" not in df_clean.columns

def test_datacleaner_does_not_mutate_input(fitted_cleaner, train_df):
    """
    GIVEN: a fitted cleaner and an input DataFrame
    WHEN: transform is executed
    THEN: the input frame is unchanged 
    """
    original = train_df.copy(deep=True)

    fitted_cleaner.transform(train_df)

    pd.testing.assert_frame_equal(train_df, original)


def test_datacleaner_logs_imputation(fitted_cleaner, train_df, caplog):
    """
    GIVEN: a fitted cleaner and a DataFrame with missing values
    WHEN: transform is executed with INFO logging captured
    THEN: the imputation events are logged 
    """
    with caplog.at_level(logging.INFO):
        fitted_cleaner.transform(train_df)

    assert any("Imputed" in record.message for record in caplog.records)