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
        for col in self.columns_to_remove:
            if col in df_clean.columns:
                df_clean = df_clean.drop(columns=[col])

        # 2. Name imputation: if the 'Name' column is present, we fill missing values with "Unknown".
        if "Name" in df_clean.columns:
            df_clean["Name"] = df_clean["Name"].fillna("Unknown")
        
        # 3. Sex imputation: categorical variable with only 1 missing value, so it's best to use the mode for imputation
        if "SexuponOutcome" in df_clean.columns:
            missing_sex = df_clean["SexuponOutcome"].isnull().sum()
            if missing_sex > 0:
                modes = df_clean["SexuponOutcome"].mode()
                mode_sex = modes[0] if not modes.empty else "Unknown"
                df_clean["SexuponOutcome"] = df_clean["SexuponOutcome"].fillna(
                    mode_sex
                )

       # 4. In order to impute the age, we first need to convert the textual representation of age into a numeric format (in days). This is done using the extract_age_in_days function defined above.
        if "AgeuponOutcome" in df_clean.columns:
            df_clean["age_in_days"] = extract_age_in_days(
                df_clean["AgeuponOutcome"]
                )
            df_clean = df_clean.drop(columns=["AgeuponOutcome"])

       # 5. Age imputation: we use the median to fill in missing values, as it is less sensitive to outliers than the mean.

        if "age_in_days" in df_clean.columns:
            missing_age = df_clean["age_in_days"].isnull().sum()
            if missing_age > 0:
                median_age = df_clean["age_in_days"].median()
                df_clean["age_in_days"] = df_clean["age_in_days"].fillna(median_age)
      
        return df_clean


