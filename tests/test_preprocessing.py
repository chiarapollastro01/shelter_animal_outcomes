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
    AnimalID, OutcomeSubtype, and AgeuponOutcome are removed, and age_in_days is
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
    expected_columns = {"Name", "SexuponOutcome", "age_in_days"}

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
    """Verify that AgeuponOutcome is converted to age_in_days and NaNs filled

    with median.

    GIVEN: a DataFrame with AgeuponOutcome values whose valid median in days is
    730.0
    WHEN: clean_data is executed
    THEN: AgeuponOutcome is converted, dropped, and its NaNs are filled with the
    median (730.0), preserving the index
    """
    df_mock = pd.DataFrame(
        {"AgeuponOutcome": ["1 year", "2 years", "3 years", np.nan, "2 years"]},
        index=[10, 20, 30, 40, 50],
    )
    expected = pd.Series(
        [365.0, 730.0, 1095.0, 730.0, 730.0],
        index=[10, 20, 30, 40, 50],
        name="age_in_days",
    )

    df_clean = DataCleaner().clean_data(df_mock)

    pd.testing.assert_series_equal(df_clean["age_in_days"], expected)


def test_datacleaner_all_nan():
    """Verify DataCleaner behavior under extreme cases where all variables are

    NaN.

    GIVEN: a DataFrame where all elements are missing
    WHEN: clean_data is executed
    THEN: Name and SexuponOutcome fall back to "Unknown", and age_in_days
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
            "age_in_days": [np.nan, np.nan],
        },
        index=[100, 200],
    )

    df_clean = DataCleaner().clean_data(df_mock)

    pd.testing.assert_frame_equal(df_clean, expected)


# =====================================================================
#                 TEMPORAL FEATURES EXTRACTOR TESTS
# =====================================================================
def test_temporal_basic_integer_extraction():
    """Verify that basic integer temporal units (Hour and Weekday) are correctly
      extracted.

    GIVEN: a DataFrame with a DateTime column and custom non-default indices
    WHEN: the transform method of TemporalFeaturesExtractor is executed
    THEN: the integer Hour and Weekday columns are correctly extracted,
    preserving the original index
    """
    df_mock = pd.DataFrame(
        {"DateTime": ["2026-01-01 12:00:00", "2026-01-03 23:00:00"]},
        index=[101, 102],
    )
    # name attributes are set to match the expected output of the transform method, which assigns names to the new columns based on their content.
    expected_hours = pd.Series([12, 23], index=[101, 102], name="Hour")
    expected_weekdays = pd.Series(
        [3, 5], index=[101, 102], name="Weekday"
    ) 

    df_transformed = TemporalFeaturesExtractor().transform(df_mock)

    pd.testing.assert_series_equal(df_transformed["Hour"], expected_hours, check_dtype=False)
    pd.testing.assert_series_equal(df_transformed["Weekday"], expected_weekdays, check_dtype=False)



def test_temporal_cyclic_hours():
    """Verify the mathematical correctness of sine and cosine transformations

    for hours.

    GIVEN: a DataFrame with specific hours representing midnight, 6 AM, and noon
    WHEN: the transform method of TemporalFeaturesExtractor is executed
    THEN: Hour_sin and Hour_cos calculate accurate circular representations,
    preserving the index
    """
    df_mock = pd.DataFrame(
        {
            "DateTime": [
                "2026-01-01 00:00:00",
                "2026-01-01 06:00:00",
                "2026-01-01 12:00:00",
            ]
        },
        index=[10, 20, 30],
    )
 
    expected_sin = pd.Series([0.0, 1.0, 0.0], index=[10, 20, 30], name="Hour_sin")
    expected_cos = pd.Series(
        [1.0, 0.0, -1.0], index=[10, 20, 30], name="Hour_cos"
    )

    df_transformed = TemporalFeaturesExtractor().transform(df_mock)

    # assert_series_equal is used to compare the transformed DataFrame's Hour_sin and Hour_cos columns against the expected values, allowing for a small 
    # numerical tolerance due to potential floating-point precision issues in trigonometric calculations.
    pd.testing.assert_series_equal(
        df_transformed["Hour_sin"], expected_sin, check_exact=False, atol=1e-7
    )
 
    pd.testing.assert_series_equal(
        df_transformed["Hour_cos"], expected_cos, check_exact=False, atol=1e-7
    )

def test_temporal_cyclic_weekdays():
    """Verify the mathematical correctness of sine and cosine transformations

    for weekdays.

    GIVEN: a DataFrame containing a Monday (day 0) and a Sunday (day 6)
    WHEN: the transform method of TemporalFeaturesExtractor is executed
    THEN: Wday_sin and Wday_cos calculate accurate cyclic values, preserving
    the index
    """
    df_mock = pd.DataFrame(
        {"DateTime": ["2026-07-06", "2026-07-12"]}, index=[15, 25]
    )  # Lunedì, Domenica
    expected_sin = pd.Series(
        np.sin(2 * np.pi * np.array([0, 6]) / 7),
        index=[15, 25],
        name="Wday_sin",
    )
    expected_cos = pd.Series(
        np.cos(2 * np.pi * np.array([0, 6]) / 7),
        index=[15, 25],
        name="Wday_cos",
    )

    df_transformed = TemporalFeaturesExtractor().transform(df_mock)

    pd.testing.assert_series_equal(
        df_transformed["Wday_sin"], expected_sin, check_exact=False, atol=1e-7
    )
    pd.testing.assert_series_equal(
        df_transformed["Wday_cos"], expected_cos, check_exact=False, atol=1e-7
    )


def test_temporal_cyclic_day_of_year():
    """Verify the mathematical correctness of sine and cosine transformations

    for the day of the year.

    GIVEN: a DataFrame containing dates representing Day of Year 1 (Jan 1) and
    100 (Apr 10) in a non-leap year
    WHEN: the transform method of TemporalFeaturesExtractor is executed
    THEN: DoY_sin and DoY_cos columns are computed accurately, preserving the
    index
    """
    df_mock = pd.DataFrame(
        {"DateTime": ["2026-01-01", "2026-04-10"]}, index=[11, 22]
    )
    expected_sin = pd.Series(
        np.sin(2 * np.pi * np.array([1, 100]) / 365.25),
        index=[11, 22],
        name="DoY_sin",
    )
    expected_cos = pd.Series(
        np.cos(2 * np.pi * np.array([1, 100]) / 365.25),
        index=[11, 22],
        name="DoY_cos",
    )

    df_transformed = TemporalFeaturesExtractor().transform(df_mock)

    pd.testing.assert_series_equal(
        df_transformed["DoY_sin"], expected_sin, check_exact=False, atol=1e-7
    )
    pd.testing.assert_series_equal(
        df_transformed["DoY_cos"], expected_cos, check_exact=False, atol=1e-7
    )


def test_temporal_missing_datetime_column():
    """Verify that the transformer returns the DataFrame unchanged when the

    specified datetime column is missing.

    GIVEN: a DataFrame that does not contain the target DateTime column
    WHEN: the transform method is executed
    THEN: the input DataFrame is returned completely unmodified, preserving
    indices and values
    """
    df_mock = pd.DataFrame({"Name": ["Bella", "Max"]}, index=[100, 200])

    df_transformed = TemporalFeaturesExtractor(datetime_col="DateTime").transform(
        df_mock
    )

    pd.testing.assert_frame_equal(df_transformed, df_mock)

