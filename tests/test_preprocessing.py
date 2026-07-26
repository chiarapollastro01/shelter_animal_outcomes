"""
Unit tests for the preprocessing module.
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
def train_X() -> pd.DataFrame:
    """Training frame with known statistics: sex mode = 'Neutered Male',
    median age = 730 days. Covers every column manipulated by the cleaner."""
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
def fitted_cleaner(train_X: pd.DataFrame) -> DataCleaner:
    """A DataCleaner already fitted on the training fixture."""
    return DataCleaner().fit(train_X)

# =====================================================================
#                      AGE EXTRACTION TESTS
# =====================================================================

def test_extract_age_typical_cases():
    """ Verify parsing of well-formatted textual age strings into float days.

    GIVEN: a pandas Series with well-formatted, lowercase age strings (years, months, weeks, days) with custom indices
    WHEN: extract_age_in_days is executed
    THEN: the numeric values are correctly converted to float days and the index is preserved
    """
    age_series = pd.Series(["1 year", "2 months", "3 weeks", "5 days"], index=[101, 102, 103, 104])
    
    expected = pd.Series([365.0, 60.0, 21.0, 5.0], index=[101, 102, 103, 104])
    
    result = extract_age_in_days(age_series)

    pd.testing.assert_series_equal(result, expected, check_names=False)
  

def test_extract_age_formatting_variations():
    """ Verify parsing validity with case variations and surrounding spaces.

    GIVEN: a Series with uppercase units, singular terms, and leading/trailing spaces with custom indices
    WHEN: extract_age_in_days is executed
    THEN: values are parsed correctly, ignoring case and spacing, preserving the index
    """
    age_series = pd.Series(["1 YEAR", "  2 months  ", "1 day", "10 DAYS"], index=[10, 20, 30, 40])

    expected = pd.Series([365.0, 60.0, 1.0, 10.0], index=[10, 20, 30, 40])
    
    result = extract_age_in_days(age_series)
    
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_extract_age_decimal_and_float_values():
    """ Verify parsing of decimal/float numbers in textual age strings.

    GIVEN: a pandas Series containing age strings with float numbers (e.g., '1.5 years', '0.5 months') with custom indices
    WHEN: extract_age_in_days is executed
    THEN: decimal numbers are extracted correctly and multiplied by unit factors, preserving the float index
    """
    age_series = pd.Series(
        ["1.5 years", "0.5 months", "2.5 weeks", "0.5 days"], 
        index=[10, 20, 30, 40]
    )

    expected = pd.Series([547.5, 15.0, 17.5, 0.5], index=[10, 20, 30, 40])

    result = extract_age_in_days(age_series)

    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_extract_age_word_boundaries_false_positives():
    """ Verify that partial word matches containing unit substrings are ignored.

    GIVEN: a pandas Series containing compound words with unit substrings (e.g., 'weekday', 'midweek') with custom indices
    WHEN: extract_age_in_days is executed
    THEN: compound words safely resolve to NaN without triggering unit multipliers, preserving the index
    """
    age_series = pd.Series(
        ["1 weekday", "5 workdays", "2 midweeks", "1 day"], 
        index=[100, 200, 300, 400]
    )

    expected = pd.Series([np.nan, np.nan, np.nan, 1.0], index=[100, 200, 300, 400])

    result = extract_age_in_days(age_series)

    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_extract_age_invalid_and_null_inputs():
    """ Verify safe fallback to NaN when encountering null/invalid cases.

    GIVEN: a Series containing a missing value (NaN), an unparseable string ("Unknown") and a number with an invalid unit
    WHEN: the extract_age_in_days function is executed
    THEN: all inputs safely resolve to NaN without breaking the execution, preserving the index
    """
    age_series = pd.Series([np.nan, "Unknown", "5 units"], index=[7,8,9])
    expected = pd.Series([np.nan, np.nan, np.nan], index=[7,8,9])
    
    result = extract_age_in_days(age_series)
    
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_extract_age_all_null_series():
    """ Verify early-return optimization when processing a Series composed entirely of NaNs.

    GIVEN: a Series where all elements are NaN with custom indeces (early-return branch)
    WHEN: extract_age_in_days is executed
    THEN: it returns a Series of the same length containing NaN with a float dtype, preserving the index
    """
    age_series = pd.Series([np.nan, np.nan], index=[10, 20])
    expected = pd.Series([np.nan, np.nan], index=[10, 20], dtype=float)
    
    result = extract_age_in_days(age_series)
    
    pd.testing.assert_series_equal(result, expected)

def test_extract_age_empty_series():
    """ Verify behavior when processing an empty pandas Series.

    GIVEN: a completely empty pandas Series (length 0) 
    WHEN: extract_age_in_days is executed
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

def test_datacleaner_fit_returns_self(train_X):
    """ Verify that DataCleaner.fit returns the fitted transformer instance.
    
    GIVEN: an unfitted DataCleaner
    WHEN: fit is executed
    THEN: the same instance is returned
    """
    cleaner = DataCleaner()
    assert cleaner.fit(train_X) is cleaner

def test_datacleaner_custom_columns_to_remove(train_X):
    """ Verify column dropping logic in DataCleaner.

    GIVEN: a DataCleaner initialised with a custom columns_to_remove list
    WHEN: fit and transform are executed
    THEN: only the requested columns are dropped 
    """
    cleaner = DataCleaner(columns_to_remove=["Name"])
    X_clean = cleaner.fit(train_X).transform(train_X)

    assert "Name" not in X_clean.columns
    assert "AnimalID" in X_clean.columns 


def test_datacleaner_column_dropping(fitted_cleaner, train_X):
    """Verify automatic removal of identifier and raw age columns.

    GIVEN: a fitted cleaner and a frame containing AnimalID
    WHEN: transform is executed
    THEN: AnimalID and AgeuponOutcome are removed, log_age_in_days is added
    """

    X_clean = fitted_cleaner.transform(train_X)

    assert "AnimalID" not in X_clean.columns
    assert "AgeuponOutcome" not in X_clean.columns
    assert "log_age_in_days" in X_clean.columns


def test_datacleaner_prevents_data_leakage(train_X):
    """ Verify that test set imputation uses learned statistics from training data.

    GIVEN: a training DataFrame with specific median age (2 years = 730 days) and sex mode (Neutered Male)
           and a test DataFrame with missing values
    WHEN: fit is called on train, and transform is called on test
    THEN: the test DataFrame is imputed using train statistics, completely preventing data leakage
    """

    X_test = pd.DataFrame(
        {"SexuponOutcome": [np.nan], "AgeuponOutcome": [np.nan]}, index=[99]
    )

    cleaner = DataCleaner().fit(train_X)
    X_test_clean = cleaner.transform(X_test)

    assert X_test_clean["SexuponOutcome"].iloc[0] == "Neutered Male"
    assert np.isclose(
        X_test_clean["log_age_in_days"].iloc[0], np.log1p(730.0), atol=1e-7
    )


def test_datacleaner_raises_runtime_error_if_unfitted():
    """ Verify that transform raises a RuntimeError if executed before fitting.

    GIVEN: a DataCleaner instance that has not been fitted
    WHEN: transform is executed
    THEN: a RuntimeError is raised with an informative error message
    """
    cleaner = DataCleaner()

    with pytest.raises(RuntimeError, match="not fitted"):
        cleaner.transform(pd.DataFrame({"Name": ["Bella"]}))


def test_datacleaner_name_imputation(fitted_cleaner, train_X):
    """ Verify missing value imputation in the Name column.

    GIVEN: a DataFrame with some null values (NaN) in the Name column
    WHEN: transform is executed
    THEN: all missing values in the Name column are replaced with "Unknown",
    preserving the original index
    """
    X_clean = fitted_cleaner.transform(train_X)

    assert X_clean["Name"].isnull().sum() == 0
    assert X_clean.loc[20, "Name"] == "Unknown"


def test_datacleaner_sex_imputation(fitted_cleaner, train_X):
    """ Verify missing value imputation in SexuponOutcome using the learned mode.

    GIVEN: a DataFrame with a missing value in SexuponOutcome where "Neutered
    Male" is the mode
    WHEN: transform is executed
    THEN: the missing value is replaced by the mode "Neutered Male", preserving
    the index
    """
    X_clean = fitted_cleaner.transform(train_X)

    assert X_clean.loc[40, "SexuponOutcome"] == "Neutered Male"
   

def test_datacleaner_age_imputation(fitted_cleaner, train_X):
    """ Verify missing age imputation with the fitted median followed by log1p transformation.

    GIVEN: a DataFrame with AgeuponOutcome values whose valid median in days is 730.0
    WHEN: transform is executed
    THEN: AgeuponOutcome is converted, dropped, its NaNs are filled with the median (730.0),
          and the resulting column is log-transformed, preserving the index
    """
    expected_days = np.array([365.0, 730.0, 1095.0, 730.0])
    expected = pd.Series(
        np.log1p(expected_days), index=[10, 20, 30, 40], name="log_age_in_days"
    )

    X_clean = fitted_cleaner.transform(train_X)

    pd.testing.assert_series_equal(
        X_clean["log_age_in_days"], expected, check_exact=False, atol=1e-7
    )

def test_datacleaner_age_log_transformation():
    """ Verify log1p transformation accuracy on complete, non-null age inputs.

    GIVEN: a DataFrame with valid AgeuponOutcome entries ("0 days", "7 days") 
           that do not need imputation
    WHEN: fit and transform are executed
    THEN: the resulting log_age_in_days column contains mathematically correct 
          log1p values, and the original column is dropped, preserving index
    """
    
    X_mock = pd.DataFrame({"AgeuponOutcome": ["0 days", "7 days"]}, index=[10, 20])
    
    
    expected = pd.Series([np.log1p(0.0), np.log1p(7.0)], index=[10, 20], name="log_age_in_days")

    X_clean = DataCleaner().fit(X_mock).transform(X_mock)

    pd.testing.assert_series_equal(X_clean["log_age_in_days"], expected, check_exact=False, atol=1e-7)


def test_datacleaner_breed_color_preventive_imputation(fitted_cleaner, train_X):
    """ Verify preventive imputation of missing Breed and Color values with 'Unknown'.

    GIVEN: a DataFrame containing missing values (NaN) in both Breed and Color columns,
           with custom non-default indices
    WHEN: transform is executed
    THEN: all missing values (NaN) in both columns are successfully replaced with "Unknown",
          preserving the original index
    """
    X_clean = fitted_cleaner.transform(train_X)

    assert X_clean.loc[20, "Breed"] == "Unknown"
    assert X_clean.loc[10, "Color"] == "Unknown"



def test_datacleaner_fit_without_sex_and_age_columns():
    """ Verify fallback statistics learned when SexuponOutcome and AgeuponOutcome are absent during fit.

    GIVEN: a DataFrame without SexuponOutcome and AgeuponOutcome 
    WHEN: fit is executed
    THEN: safe fallbacks are learned ('Unknown', 0.0) (covers the
          missing-column branches of fit)
    """
    X_mock = pd.DataFrame({"Breed": ["Beagle"], "Color": ["Black"]})

    cleaner = DataCleaner().fit(X_mock)

    assert cleaner.sex_mode_ == "Unknown"
    assert cleaner.age_median_ == 0.0


def test_datacleaner_transform_without_sex_and_age_columns(fitted_cleaner):
    """ Verify safe transform execution when SexuponOutcome and AgeuponOutcome are absent.

    GIVEN: a fitted cleaner and a DataFrame without SexuponOutcome and AgeuponOutcome
    WHEN: transform is executed
    THEN: log_age_in_days column isn't created and no exception raises
    """
    X_mock = pd.DataFrame({"Name": ["Bella", np.nan]}, index=[1, 2])

    X_clean = fitted_cleaner.transform(X_mock)

    assert "log_age_in_days" not in X_clean.columns
    assert X_clean.loc[2, "Name"] == "Unknown"


def test_datacleaner_all_nan():
    """ Verify robust handling of a DataFrame composed entirely of missing entries.

    GIVEN: a DataFrame where all elements are missing
    WHEN: fit and transform are executed
    THEN: Name and SexuponOutcome fall back to "Unknown", and log_age_in_days
          is safely imputed with 0.0
    """
    X_mock = pd.DataFrame(
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

    X_clean = DataCleaner().fit(X_mock).transform(X_mock)

    pd.testing.assert_frame_equal(X_clean, expected)


def test_datacleaner_empty_dataframe():
    """ Verify robust processing when fitting and transforming an empty DataFrame.

    GIVEN: an empty DataFrame with valid column headers
    WHEN: fit and transform are executed
    THEN: the output DataFrame is empty, schema is preserved, and default fallback states are learned
    """
    X_mock = pd.DataFrame(columns=["Name", "SexuponOutcome", "AgeuponOutcome"])
    
    cleaner = DataCleaner().fit(X_mock)
    X_clean=cleaner.transform(X_mock)

    assert cleaner.sex_mode_ == "Unknown"
    assert cleaner.age_median_ == 0.0    
    assert X_clean.empty
    assert "log_age_in_days" in X_clean.columns
    assert "AgeuponOutcome" not in X_clean.columns

def test_datacleaner_does_not_mutate_input(fitted_cleaner, train_X):
    """ Verify that transform does not mutate the input DataFrame.

    GIVEN: a fitted cleaner and an input DataFrame
    WHEN: transform is executed
    THEN: the input frame is unchanged 
    """
    original = train_X.copy(deep=True)

    fitted_cleaner.transform(train_X)

    pd.testing.assert_frame_equal(train_X, original)


def test_datacleaner_logs_fit(train_X, caplog):
    """ Verify logging output during the fit execution.

    GIVEN: a DataCleaner instance and a training DataFrame
    WHEN: fit is executed with INFO logging captured
    THEN: the fitting statistics (sex_mode_ and age_median_) are logged
    """
    cleaner = DataCleaner()

    with caplog.at_level(logging.INFO):
        cleaner.fit(train_X)

    assert any("Fitted DataCleaner" in record.message for record in caplog.records)

def test_datacleaner_logs_imputation_details(fitted_cleaner, train_X, caplog):
    """ Verify specific logging output for both SexuponOutcome and AgeuponOutcome imputations.

    GIVEN: a fitted cleaner and a DataFrame with missing values in both Sex and Age
    WHEN: transform is executed with INFO logging captured
    THEN: distinct imputation events are logged for both SexuponOutcome and AgeuponOutcome
    """
    with caplog.at_level(logging.INFO):
        fitted_cleaner.transform(train_X)

    messages = [record.message for record in caplog.records]

    assert any("Imputed" in msg and "SexuponOutcome" in msg for msg in messages)

    assert any("Imputed" in msg and "AgeuponOutcome" in msg for msg in messages)