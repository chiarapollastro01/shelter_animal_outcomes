"""
Unit tests for the preprocessing module.

"""
import numpy as np
import pandas as pd
from src.preprocessing import extract_age_in_days, DataCleaner
from src.preprocessing import TemporalFeaturesExtractor

# =====================================================================
#                      AGE EXTRACTION TESTS
# =====================================================================

def test_extract_age_typical_cases():
    """
    Test the conversion of typical age strings with standard units.

    GIVEN: a pandas Series with well-formatted, lowercase age strings (years, months, weeks, days)
    WHEN: the extract_age_in_days function is executed
    THEN: the numeric values are correctly converted to float days
    """
    age_series = pd.Series(["1 year", "2 months", "3 weeks", "5 days"])
    
    expected = pd.Series([365.0, 60.0, 21.0, 5.0])
    
    result = extract_age_in_days(age_series)
    # check_names=False is used to ignore the name of the Series during comparison, as the function does not set a name for the output Series.
    # We are indeed only interested in the values and their order, not the name of the Series.
    pd.testing.assert_series_equal(result, expected, check_names=False)
  

def test_extract_age_formatting_variations():
    """
    Test that the function can handle case variations and spacing around text.

    GIVEN: a Series with uppercase units, singular terms, and leading/trailing spaces
    WHEN: the extract_age_in_days function is executed
    THEN: values are parsed correctly, ignoring case and surrounding spacing
    """
    age_series = pd.Series(["1 YEAR", "  2 months  ", "1 day", "10 DAYS"])

    expected = pd.Series([365.0, 60.0, 1.0, 10.0])
    
    result = extract_age_in_days(age_series)
    
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_extract_age_invalid_and_null_inputs():
    """
    Test how the function handles unparseable text and singular null values.

    GIVEN: a Series containing a missing value (NaN) and an unparseable string ("Unknown")
    WHEN: the extract_age_in_days function is executed
    THEN: both the NaN and the unparseable input return NaN
    """
    age_series = pd.Series([np.nan, "Unknown"])
    expected = pd.Series([np.nan, np.nan])
    
    result = extract_age_in_days(age_series)
    
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_extract_age_all_null_series():
    """
    Test the specific branch optimization where the entire input Series is null.

    GIVEN: a Series where all elements are NaN
    WHEN: the extract_age_in_days function is executed
    THEN: it returns a Series of the same length containing NaN with a float dtype
    """
    # Indexes are intentionally non-sequential to ensure the function preserves the original index.
    age_series = pd.Series([np.nan, np.nan], index=[10, 20])
    expected = pd.Series([np.nan, np.nan], index=[10, 20], dtype=float)
    
    result = extract_age_in_days(age_series)
    
    pd.testing.assert_series_equal(result, expected)


# =====================================================================
#                       DATA CLEANER TESTS
# =====================================================================


def test_datacleaner_column_dropping():
    """Verify that irrelevant columns are removed and AgeuponOutcome is converted.

    GIVEN: a DataFrame containing columns to drop (AnimalID, OutcomeSubtype),
    AgeuponOutcome, and Name/SexuponOutcome WHEN: clean_data is executed THEN:
    AnimalID, OutcomeSubtype, and AgeuponOutcome are removed, and log_age_in_days is
    added
    """

    df_mock = pd.DataFrame(
        {
            "AnimalID": ["A1", "A2"],
            "OutcomeSubtype": ["Partner", "Foster"],
            "Name": ["Bella", "Max"],
            "SexuponOutcome": ["Neutered Male", "Intact Female"],
            "AgeuponOutcome": ["1 year", "2 years"],
        },
        index=[10, 20],
    )
    expected_columns = {"Name", "SexuponOutcome", "log_age_in_days"}

    df_clean = DataCleaner().clean_data(df_mock)

    assert set(df_clean.columns) == expected_columns


def test_datacleaner_name_imputation():
    """Verify that missing values in the Name column are filled with "Unknown".

    GIVEN: a DataFrame with some null values (NaN) in the Name column
    WHEN: clean_data is executed
    THEN: all missing values in the Name column are replaced with "Unknown",
    preserving the original index
    """
    df_mock = pd.DataFrame({"Name": ["Bella", np.nan, "Luna"]}, index=[10, 20, 30])
    expected = pd.Series(
        ["Bella", "Unknown", "Luna"], index=[10, 20, 30], name="Name"
    )

    df_clean = DataCleaner().clean_data(df_mock)

    pd.testing.assert_series_equal(df_clean["Name"], expected)


def test_datacleaner_sex_imputation():
    """Verify that missing values in SexuponOutcome are imputed using the
    column's mode.

    GIVEN: a DataFrame with a missing value in SexuponOutcome where "Neutered
    Male" is the mode
    WHEN: clean_data is executed
    THEN: the missing value is replaced by the mode "Neutered Male", preserving
    the index
    """
    df_mock = pd.DataFrame(
        {
            "SexuponOutcome": [
                "Neutered Male",
                "Neutered Male",
                "Intact Female",
                np.nan,
                "Neutered Male",
            ]
        },
        index=[10, 20, 30, 40, 50],
    )
    expected = pd.Series(
        [
            "Neutered Male",
            "Neutered Male",
            "Intact Female",
            "Neutered Male",
            "Neutered Male",
        ],
        index=[10, 20, 30, 40, 50],
        name="SexuponOutcome",
    )

    df_clean = DataCleaner().clean_data(df_mock)

    pd.testing.assert_series_equal(df_clean["SexuponOutcome"], expected)

def test_datacleaner_age_imputation():
    """Verify that AgeuponOutcome is converted, imputed with median, and log-transformed.

    GIVEN: a DataFrame with AgeuponOutcome values whose valid median in days is 730.0
    WHEN: clean_data is executed
    THEN: AgeuponOutcome is converted, dropped, its NaNs are filled with the median (730.0),
          and the resulting column is log-transformed, preserving the index
    """
    df_mock = pd.DataFrame(
        {"AgeuponOutcome": ["1 year", "2 years", "3 years", np.nan, "2 years"]},
        index=[10, 20, 30, 40, 50],
    )
    expected_days = np.array([365.0, 730.0, 1095.0, 730.0, 730.0])
    expected = pd.Series(
        np.log1p(expected_days),
        index=[10, 20, 30, 40, 50],
        name="log_age_in_days",
    )

    df_clean = DataCleaner().clean_data(df_mock)

    pd.testing.assert_series_equal(df_clean["log_age_in_days"], expected, check_exact=False, atol=1e-7)

def test_datacleaner_age_log_transformation():
    """
    Verify that age values are correctly transformed using the np.log1p formula.

    GIVEN: a DataFrame with valid AgeuponOutcome entries ("0 days", "7 days") 
           that do not trigger median calculation (no missing values)
    WHEN: clean_data is executed
    THEN: the resulting log_age_in_days column contains mathematically correct 
          log1p values, and the original column is dropped, preserving index
    """
    
    df_mock = pd.DataFrame({"AgeuponOutcome": ["0 days", "7 days"]}, index=[10, 20])
    
    
    expected = pd.Series([np.log1p(0.0), np.log1p(7.0)], index=[10, 20], name="log_age_in_days")

    df_clean = DataCleaner().clean_data(df_mock)

    pd.testing.assert_series_equal(df_clean["log_age_in_days"], expected, check_exact=False, atol=1e-7)


def test_datacleaner_all_nan():
    """Verify DataCleaner behavior under extreme cases where all variables are NaN.

    GIVEN: a DataFrame where all elements are missing
    WHEN: clean_data is executed
    THEN: Name and SexuponOutcome fall back to "Unknown", and log_age_in_days
          remains NaN, preserving the original index and data types
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
            "log_age_in_days": [np.nan, np.nan], 
        },
        index=[100, 200],
    )

    df_clean = DataCleaner().clean_data(df_mock)

    pd.testing.assert_frame_equal(df_clean, expected)

# =====================================================================
#                 TEMPORAL FEATURES EXTRACTOR TESTS
# =====================================================================

def test_temporal_base_typical_cases():
    """Verify that the base temporal extractor correctly parses DateTime and
       extracts Hour and Weekday.

    GIVEN: a DataFrame with raw string dates and custom, non-default indices
    WHEN: the transform method of TemporalFeaturesExtractor is executed
    THEN: the DateTime column is converted to datetime64, and the correct Hour
    and Weekday
    integers are appended as new columns, preserving the original index
    """
    df_mock = pd.DataFrame(
        {"DateTime": ["2026-07-06 12:00:00", "2026-07-10 23:00:00"]},
        index=[101, 102],
    )
    expected_hours = pd.Series([12, 23], index=[101, 102], name="Hour")
    expected_weekdays = pd.Series([0, 4], index=[101, 102], name="Weekday")

    df_transformed = TemporalFeaturesExtractor().transform(df_mock)

    assert df_transformed["DateTime"].dtype == "datetime64[ns]"

    pd.testing.assert_series_equal(
        df_transformed["Hour"], expected_hours, check_dtype=False
    )
    pd.testing.assert_series_equal(
        df_transformed["Weekday"], expected_weekdays, check_dtype=False
    )


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