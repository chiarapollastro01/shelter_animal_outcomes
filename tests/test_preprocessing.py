"""
Unit tests for the preprocessing module.
"""
import logging
import numpy as np
import pytest
import pandas as pd
from src import config
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
            config.ID_COL: ["A1", "A2", "A3", "A4"],
            config.DATETIME_COL: ["2026-01-01 10:00:00", "2026-01-02 11:00:00", "2026-01-03 12:00:00", "2026-01-04 13:00:00"],
            config.NAME_COL: ["Bella", np.nan, "Luna", "Max"],
            config.BREED_COL: ["Chihuahua Mix", np.nan, "Beagle", "Beagle"],
            config.COLOR_COL: [np.nan, "Black", "White", "Black"],
            config.SEX_COL: ["Neutered Male", "Neutered Male", "Intact Female", np.nan],
            config.AGE_COL: ["1 year", "2 years", "3 years", np.nan],
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

def test_extract_age_word_boundaries():
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

def test_extract_age_handles_null_values():
    """Verify parsing when a Series contains a mix of valid age strings and missing values.

    GIVEN: a Series containing a valid age string ('2 years') and a missing value (NaN)
    WHEN: extract_age_in_days is executed
    THEN: the valid string is correctly converted to float days (730.0) and the NaN remains NaN, preserving index
    """
    age_series = pd.Series(["2 years", np.nan], index=[10, 20])
    expected = pd.Series([730.0, np.nan], index=[10, 20])

    result = extract_age_in_days(age_series)

    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_extract_age_invalid_inputs():

    """ Verify safe fallback to NaN when encountering invalid cases.

    GIVEN: a Series containing an unparseable string ("Unknown") and a number with an invalid unit
    WHEN: the extract_age_in_days function is executed
    THEN: all inputs safely resolve to NaN without breaking the execution, preserving the index
    """
    age_series = pd.Series(["Unknown", "5 units"], index=[8,9])
    expected = pd.Series([np.nan, np.nan], index=[8,9])
    
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
    
    pd.testing.assert_series_equal(result, expected, check_names=False)

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
    pd.testing.assert_series_equal(result, expected, check_names=False)


# =====================================================================
#                       DATA CLEANER TESTS
# =====================================================================
# -----------------------TEST FIT------------------------------------------------------------------
class TestDataCleanerFit:
  
  def test_datacleaner_returns_self(self, train_X):
    """ Verify that DataCleaner.fit returns the fitted transformer instance.
    
    GIVEN: an unfitted DataCleaner
    WHEN: fit is executed
    THEN: the same instance is returned
    """
    cleaner = DataCleaner()
    assert cleaner.fit(train_X) is cleaner

  def test_datacleaner_learns_correct_statistics(self, train_X):
       """Verify that fit learns exact mode for sex ('Neutered Male') and median age (730.0 days).

        GIVEN: a training DataFrame with known sex mode ('Neutered Male') and median age (730.0 days)
        WHEN: fit is executed
        THEN: sex_mode_ is set to 'Neutered Male' and age_median_ is set to 730.0
        """
       cleaner = DataCleaner().fit(train_X)
       assert cleaner.sex_mode_ == "Neutered Male"
       assert cleaner.age_median_ == 730.0

  def test_datacleaner_handles_all_nan_columns(self):
     """Verify fallback statistics learned when columns contain only NaN values.

      GIVEN: a DataFrame where sex and age columns contain only missing values (NaN)
      WHEN: fit is executed
      THEN: safe default fallbacks are learned (sex_mode_='Unknown', age_median_=0.0)
     """
     X_mock = pd.DataFrame({config.SEX_COL: [np.nan, np.nan], config.AGE_COL: [np.nan, np.nan]})
     cleaner = DataCleaner().fit(X_mock)

     assert cleaner.sex_mode_ == "Unknown"
     assert cleaner.age_median_ == 0.0

  def test_datacleaner_without_sex_and_age_columns(self):
    """ Verify fallback statistics learned when sex and age columns are absent during fit.

    GIVEN: a DataFrame without sex and age columns
    WHEN: fit is executed
    THEN: safe fallbacks are learned ('Unknown', 0.0) 
    """
    X_mock = pd.DataFrame({config.BREED_COL: ["Beagle"], config.COLOR_COL: ["Black"]})

    cleaner = DataCleaner().fit(X_mock)

    assert cleaner.sex_mode_ == "Unknown"
    assert cleaner.age_median_ == 0.0


#-------------------------------TEST TRANSFORM--------------------------------------------

class TestDataCleanerTransform:
     
  def test_datacleaner_transform_raises_runtime_error_if_unfitted(self):
    """ Verify that transform raises a RuntimeError if executed before fitting.

    GIVEN: a DataCleaner instance that has not been fitted
    WHEN: transform is executed
    THEN: a RuntimeError is raised with an informative error message
    """
    cleaner = DataCleaner()

    with pytest.raises(RuntimeError, match="not fitted"):
        cleaner.transform(pd.DataFrame({config.NAME_COL: ["Bella"]}))


  def test_datacleaner_column_dropping(self, fitted_cleaner, train_X):
     """Verify automatic removal of identifier and raw age columns.

    GIVEN: a fitted cleaner and a frame containing identifier and age columns
    WHEN: transform is executed
    THEN: identifier and age columns are removed, log_age_in_days is added
     """
     X_clean = fitted_cleaner.transform(train_X)

     assert config.ID_COL not in X_clean.columns
     assert config.AGE_COL not in X_clean.columns
     assert "log_age_in_days" in X_clean.columns


  def test_datacleaner_name_imputation(self, fitted_cleaner, train_X):
    """ Verify missing value imputation in the name column.

    GIVEN: a DataFrame with some null values (NaN) in the name column
    WHEN: transform is executed
    THEN: all missing values in the name column are replaced with "Unknown",
    preserving the original index
    """
    X_clean = fitted_cleaner.transform(train_X)

    assert X_clean[config.NAME_COL].isnull().sum() == 0
    assert X_clean.loc[20, config.NAME_COL] == "Unknown"

  def test_datacleaner_sex_imputation(self, fitted_cleaner, train_X):
    """ Verify missing value imputation in sex column using the learned mode.

    GIVEN: a DataFrame with a missing value in sex column where "Neutered
    Male" is the mode
    WHEN: transform is executed
    THEN: the missing value is replaced by the mode "Neutered Male", preserving
    the index
    """
    X_clean = fitted_cleaner.transform(train_X)

    assert X_clean.loc[40, config.SEX_COL] == "Neutered Male"
   

  def test_datacleaner_age_imputation(self, fitted_cleaner, train_X):
    """ Verify missing age imputation with the fitted median followed by log1p transformation.

    GIVEN: a DataFrame with age column values whose valid median in days is 730.0
    WHEN: transform is executed
    THEN: age column is converted, dropped, its NaNs are filled with the median (730.0),
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

  def test_datacleaner_age_log_transformation(self, fitted_cleaner):
    """Verify log1p transformation accuracy on complete, non-null age inputs using fitted_cleaner.

        GIVEN: a fitted cleaner and a DataFrame with valid age entries ('0 days', '7 days')
        WHEN: transform is executed
        THEN: log_age_in_days contains mathematically correct log1p values and age column is dropped
        """
    X_mock = pd.DataFrame({config.AGE_COL: ["0 days", "7 days"]}, index=[10, 20])

    expected = pd.Series(
        [np.log1p(0.0), np.log1p(7.0)], index=[10, 20], name="log_age_in_days"
    )

    X_clean = fitted_cleaner.transform(X_mock)

    pd.testing.assert_series_equal(
        X_clean["log_age_in_days"], expected, check_exact=False, atol=1e-7
    )


  def test_datacleaner_age_unparseable_text(self):
    """Verify that unparseable age strings are safely imputed using a set median.

        GIVEN: a fitted cleaner and a DataFrame with unparseable age text (e.g. 'invalid_age')
        WHEN: transform is executed
        THEN: unparseable text is treated as NaN, imputed with fitted median, and log1p transformed
    """
    X_train = pd.DataFrame({config.AGE_COL: ["1 year"]})
    cleaner = DataCleaner().fit(X_train)

    X_test = pd.DataFrame({config.AGE_COL: ["invalid_age", "2 years"]})
    X_clean = cleaner.transform(X_test)

    expected = pd.Series([np.log1p(365.0), np.log1p(730.0)], name="log_age_in_days")
    pd.testing.assert_series_equal(X_clean["log_age_in_days"], expected, check_exact=False, atol=1e-7)

  def test_datacleaner_breed_color_preventive_imputation(self, fitted_cleaner, train_X):
    """ Verify preventive imputation of missing Breed and Color values with 'Unknown'.

    GIVEN: a DataFrame containing missing values (NaN) in both breed and color columns,
           with custom non-default indices
    WHEN: transform is executed
    THEN: all missing values (NaN) in both columns are successfully replaced with "Unknown",
          preserving the original index
    """
    X_clean = fitted_cleaner.transform(train_X)

    assert X_clean.loc[20, config.BREED_COL] == "Unknown"
    assert X_clean.loc[10, config.COLOR_COL] == "Unknown"


  def test_datacleaner_transform_without_sex_and_age_columns(self, fitted_cleaner):
    """ Verify safe transform execution when sex and age columns are absent.

    GIVEN: a fitted cleaner and a DataFrame without sex and age columns
    WHEN: transform is executed
    THEN: log_age_in_days column isn't created and no exception raises
    """
    X_mock = pd.DataFrame({config.NAME_COL: ["Bella", np.nan]}, index=[1, 2])

    X_clean = fitted_cleaner.transform(X_mock)

    assert "log_age_in_days" not in X_clean.columns
    assert X_clean.loc[2, config.NAME_COL] == "Unknown"


  def test_datacleaner_does_not_mutate_input(self, fitted_cleaner, train_X):
    """ Verify that transform does not mutate the input DataFrame.

    GIVEN: a fitted cleaner and an input DataFrame
    WHEN: transform is executed
    THEN: the input frame is unchanged 
    """
    original = train_X.copy(deep=True)

    fitted_cleaner.transform(train_X)

    pd.testing.assert_frame_equal(train_X, original)

  def test_datacleaner_logs_imputation_details(self, fitted_cleaner, train_X, caplog):
    """ Verify specific logging output for both sex and age columns imputations.

    GIVEN: a fitted cleaner and a DataFrame with missing values in both sex and age columns
    WHEN: transform is executed with INFO logging captured
    THEN: distinct imputation events are logged for both sex and age columns
    """
    with caplog.at_level(logging.INFO):
        fitted_cleaner.transform(train_X)

    assert config.SEX_COL in caplog.text
    assert config.AGE_COL in caplog.text

#----------------------------------------END TO END, CUSTOM & EDGE CASES------------------------------------------------------
class TestDataCleanerCustomAndE2E:
  def test_datacleaner_all_nan(self):
    """ Verify robust handling of a DataFrame composed entirely of missing entries.

    GIVEN: a DataFrame where all elements are missing
    WHEN: fit and transform are executed
    THEN: name and sex columns fall back to "Unknown", and log_age_in_days
          is safely imputed with 0.0
    """
    X_mock = pd.DataFrame(
        {
            config.NAME_COL: [np.nan, np.nan],
            config.SEX_COL: [np.nan, np.nan],
            config.AGE_COL: [np.nan, np.nan],
        },
        index=[100, 200],
    )
    expected = pd.DataFrame(
        {
            config.NAME_COL: ["Unknown", "Unknown"],
            config.SEX_COL: ["Unknown", "Unknown"],
            "log_age_in_days": [0.0, 0.0], 
        },
        index=[100, 200],
    )

    X_clean = DataCleaner().fit(X_mock).transform(X_mock)

    pd.testing.assert_frame_equal(X_clean, expected)

  def test_datacleaner_empty_dataframe(self):
    """ Verify robust processing when fitting and transforming an empty DataFrame.

    GIVEN: an empty DataFrame with valid column headers
    WHEN: fit and transform are executed
    THEN: the output DataFrame is empty, schema is preserved, and default fallback states are learned
    """
    X_mock = pd.DataFrame(columns=(config.NAME_COL, config.SEX_COL, config.AGE_COL))
    
    cleaner = DataCleaner().fit(X_mock)
    X_clean=cleaner.transform(X_mock)

    assert cleaner.sex_mode_ == "Unknown"
    assert cleaner.age_median_ == 0.0    
    assert X_clean.empty
    assert "log_age_in_days" in X_clean.columns
    assert config.AGE_COL not in X_clean.columns


  def test_datacleaner_prevents_data_leakage(self, train_X):
    """ Verify that test set imputation uses learned statistics from training data.

    GIVEN: a training DataFrame with specific median age (2 years = 730 days) and sex mode (Neutered Male)
           and a test DataFrame with missing values
    WHEN: fit is called on train, and transform is called on test
    THEN: the test DataFrame is imputed using train statistics, completely preventing data leakage
    """

    X_test = pd.DataFrame(
        {config.SEX_COL: [np.nan], config.AGE_COL: [np.nan]}, index=[99]
    )

    cleaner = DataCleaner().fit(train_X)
    X_test_clean = cleaner.transform(X_test)

    assert X_test_clean[config.SEX_COL].iloc[0] == "Neutered Male"
    assert np.isclose(
        X_test_clean["log_age_in_days"].iloc[0], np.log1p(730.0), atol=1e-7
    )

  def test_datacleaner_custom_columns_to_remove(self, train_X):
    """ Verify column dropping logic in DataCleaner.

    GIVEN: a DataCleaner initialised with a custom list
    WHEN: fit and transform are executed
    THEN: only the requested columns are dropped 
    """
    cleaner = DataCleaner(columns_to_remove=(config.NAME_COL,))
    X_clean = cleaner.fit_transform(train_X)

    assert config.NAME_COL not in X_clean.columns
    assert config.ID_COL in X_clean.columns 


  def test_datacleaner_custom_fill_targets(self):
    """Verify categorical missing value imputation with custom fill_targets.

    GIVEN: a DataFrame with missing values in custom categorical columns
    WHEN: DataCleaner is initialized with custom fill_targets tuple
    THEN: missing values in specified columns are filled with 'Unknown', 
          while untouched columns retain their original NaNs
    """
    X_mock = pd.DataFrame(
        {
            "custom_breed": ["Labrador", np.nan],
            "custom_color": [np.nan, "Black"],
            "other_col": [np.nan, "Keep_NaN"],
        }
    )

    cleaner = DataCleaner(fill_targets=("custom_breed", "custom_color"))
    X_clean = cleaner.fit_transform(X_mock)

    assert X_clean["custom_breed"].isnull().sum() == 0
    assert X_clean["custom_color"].isnull().sum() == 0
    assert X_clean["custom_breed"].iloc[1] == "Unknown"
    assert X_clean["custom_color"].iloc[0] == "Unknown"

    assert X_clean["other_col"].isnull().sum() == 1


  def test_datacleaner_custom_column_names(self):
    """Verify processing when using custom sex and age column names.

    GIVEN: a DataFrame with custom column names ('animal_sex', 'animal_age')
    WHEN: DataCleaner initialized with sex_col='animal_sex' and age_col='animal_age' executes fit and transform
    THEN: custom columns are processed correctly and log_age_in_days is generated
    """
    X_mock = pd.DataFrame(
        {"animal_sex": ["Neutered Male", np.nan], "animal_age": ["1 year", "2 years"]},
        index=[1, 2],
    )

    cleaner = DataCleaner(sex_col="animal_sex", age_col="animal_age")
    X_clean = cleaner.fit_transform(X_mock)

    assert "log_age_in_days" in X_clean.columns
    assert "animal_age" not in X_clean.columns
    assert X_clean["animal_sex"].isnull().sum() == 0