"""
Data preprocessing module.

"""
import pandas as pd
import numpy as np

def extract_age_in_days(age_series: pd.Series) -> pd.Series:
    """
    Converse a textual Series of age (e.g., '2 years') into a numeric Series of days.
    Utilizes Pandas vectorization to maximize performance (no for-loops).
    
    Parameters
    ----------
    age_series : pd.Series
        The column containing the age strings.
        
    Returns
    -------
    pd.Series
        A new column with the values converted into numeric days (float).
    """

    # 1. Handle the case where the entire Series is NaN
    if age_series.isnull().all():
        return pd.Series(np.nan, index=age_series.index, dtype=float)
    
    # 2. Extract the numeric part of the age string:
    #.str.extract(r'(\d+)') finds the first occurrence of one or more digits in the string and returns it as a new Series.
    numeric_values = age_series.str.extract(r'(\d+)')[0].astype(float)
    
    # 3. To ensure case insensitivity, we convert the text to lowercase.
    text = age_series.str.lower()
    
    # 4. Create a multiplier based on the time unit found in the text.
    multipliers = np.where(text.str.contains('year', na=False), 365.0,
                     np.where(text.str.contains('month', na=False), 30.0,
                     np.where(text.str.contains('week', na=False), 7.0,
                     np.where(text.str.contains('day', na=False), 1.0, 
                     np.nan)))) # Se non trova nulla o è NaN, restituisce NaN
    

    
    # 5. Multiply the numeric values by their corresponding multipliers to get the age in days. 
    # The NaN values will remain NaN because a number multiplied by NaN is NaN.
    return numeric_values * multipliers
 

class DataCleaner:
    """
    Initial class for cleaning the Shelter Animal Outcomes dataset.
    
    This class handles the basic preprocessing steps, including the removal
    of irrelevant or data-leaking columns and the imputation of
    missing values.

    """
    
    def __init__(self):
        """
        Initialize the DataCleaner with a list of columns to remove.

        """
        self.columns_to_remove = ["AnimalID", "OutcomeSubtype"]

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input DataFrame by removing specified columns and imputing missing values.

        """
        df_clean = df.copy()

        # 1. Columns that are not relevant or have too many missing values are removed
        df_clean = df_clean.drop(
            columns=self.columns_to_remove, errors="ignore"
        )

        # 2. Name imputation: if the 'Name' column is present, we fill missing values with "Unknown".
        if "Name" in df_clean.columns:
            df_clean["Name"] = df_clean["Name"].fillna("Unknown")
        
        # 3. Sex imputation: categorical variable with only 1 missing value, so it's best to use the mode for imputation
        if "SexuponOutcome" in df_clean.columns:
                modes = df_clean["SexuponOutcome"].mode()
                mode_sex = modes[0] if not modes.empty else "Unknown"
                df_clean["SexuponOutcome"] = df_clean["SexuponOutcome"].fillna(
                    mode_sex
                )

       # 4. Pipeline for AgeuponOutcome: we extract the age in days, impute missing values with the median, and apply a log1p transformation to reduce skewness.
        if "AgeuponOutcome" in df_clean.columns:
            age_days = extract_age_in_days(df_clean["AgeuponOutcome"])

            # Missing value imputation: we fill NaN values with the median of the non-NaN values. If all values are NaN, we leave them as NaN.
            valid_ages = age_days.dropna()
            if not valid_ages.empty:
                age_days = age_days.fillna(valid_ages.median())

            # Skewness reduction: we apply a log1p transformation to the age in days. This is particularly useful for machine learning models.
            # log1p is used instead of log to handle the case where age_days might be 0.
            df_clean["log_age_in_days"] = np.log1p(age_days)
            df_clean = df_clean.drop(columns=["AgeuponOutcome"])

        return df_clean


class TemporalFeaturesExtractor:
    """Extractor class for temporal and cyclic features from the DateTime column.

    This class converts the 'DateTime' column into a datetime object, generates
    cyclic representations (sine/cosine) for hours, weekdays, and days of the
    year.
    """

    def __init__(self, datetime_col: str = "DateTime"):
        """Initialize the extractor with the name of the datetime column."""
        self.datetime_col = datetime_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal and cyclic features from the specified datetime
          column.
        """
        df_out = df.copy()

        # If the specified datetime column is not present in the DataFrame, return the DataFrame unchanged.
        if self.datetime_col not in df_out.columns:
            return df_out

        # 1. Converting to datetime format to ensure proper extraction of features.
        df_out[self.datetime_col] = pd.to_datetime(df_out[self.datetime_col])
        dt_series = df_out[self.datetime_col]

        # 2. In order to capture the cyclic nature of hours in a day, we use sine and cosine transformations. 
        # This allows machine learning models to understand that 23:00 and 00:00 are close in time, despite their numerical difference.
        hours = dt_series.dt.hour
        df_out["Hour"] = hours
        df_out["Hour_sin"] = np.sin(2 * np.pi * hours / 24)
        df_out["Hour_cos"] = np.cos(2 * np.pi * hours / 24)

        # 3. To capture the cyclic nature of weekdays, we again use sine and cosine transformations. This ensures that models understand the proximity of days in a week, e.g., Sunday and Monday.
        weekday = dt_series.dt.dayofweek  # Monday=0, Sunday=6
        df_out["Weekday"] = weekday
        df_out["Wday_sin"] = np.sin(2 * np.pi * weekday / 7)
        df_out["Wday_cos"] = np.cos(2 * np.pi * weekday / 7)

        # 4. To capture the cyclic nature of days in a year, we use sine and cosine transformations based on the day of the year. This is particularly useful for capturing seasonal patterns in the data.
        doy = dt_series.dt.dayofyear
        df_out["DoY_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df_out["DoY_cos"] = np.cos(2 * np.pi * doy / 365.25)


        return df_out
    




    #TODO: outliers for age + breed + color. visual representation of daytime feature + the ones that are missing like sex (maybe after/during feature eng.). 