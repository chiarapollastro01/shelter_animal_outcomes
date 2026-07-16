"""
Preprocessing and Cleaning Module for the Shelter Animal Outcomes Dataset.

This module provides the primary data cleaning pipeline and custom transformers 
required to prepare raw data for machine learning algorithms. 

Exported Classes
----------------
DataCleaner
    A class that orchestrates column dropping,
    imputations and mathematical formatting.

Exported Functions
------------------
extract_age_in_days(age_series: pd.Series) -> pd.Series
    Function that parses textual age strings 
    (e.g., '2 years', '3 weeks') into equivalent numeric float days.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from sklearn.base import TransformerMixin

logger = logging.getLogger(__name__)

def extract_age_in_days(age_series: pd.Series) -> pd.Series:
    """
    Convert a textual Series of age (e.g., '2 years') 
    into a numeric Series of days (float).
    
    Parameters
    ----------
    age_series : pd.Series
        The column containing the age strings.
        
    Returns
    -------
    pd.Series
        A new column with the values converted into numeric days (float).
        Non-parsable entries are NaN.
    """
    if age_series.isnull().all():
        return pd.Series(np.nan, index=age_series.index, dtype=float)
    
    numeric_values = age_series.str.extract(r'(\d+)')[0].astype(float)

    text = age_series.str.lower()
    
    conds = [
        text.str.contains('year', na=False),
        text.str.contains('month', na=False),
        text.str.contains('week', na=False),
        text.str.contains('day', na=False)
    ]
    choices = [365.0, 30.0, 7.0, 1.0]
    
    multipliers = np.select(conds, choices, default=np.nan)

    return numeric_values * multipliers
 
# May need BaseEstimator other than TransformerMixin ... in case of future GridSearch (?)
@dataclass
class DataCleaner(TransformerMixin):
    """
    Initial class for cleaning the Shelter Animal Outcomes dataset.
    Responsibilities
    ----------------
    - Drop leaky or irrelevant columns.
    - Impute missing values for key columns.
    - Convert ``AgeuponOutcome`` to log age in days.

    """

    columns_to_remove: list[str] = field(
    default_factory=lambda: ["AnimalID", "OutcomeSubtype"])

    sex_mode_: str | None = field(default=None, init=False, repr=False)
    age_median_: float | None = field(default=None, init=False, repr=False)

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
       """Learn imputation statistics (mode for sex, median age in days).
        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.
        Returns
        -------
        DataCleaner
            Fitted instance.
        """
       if "SexuponOutcome" in df.columns:
            modes = df["SexuponOutcome"].mode()
            self.sex_mode_ = modes.iloc[0] if not modes.empty else "Unknown"
       else:
            self.sex_mode_ = "Unknown"

       if "AgeuponOutcome" in df.columns:
            age_days = extract_age_in_days(df["AgeuponOutcome"])
            valid_ages = age_days.dropna()
            self.age_median_ = float(valid_ages.median()) if not valid_ages.empty else 0.0
       else:
            self.age_median_ = 0.0
       logger.info(
            "Fitted DataCleaner: sex_mode_='%s', age_median_=%.1f days",
            self.sex_mode_,
            self.age_median_,
        )
       return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned statistics to clean and impute the dataset.
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to clean (train, validation or test).
        Returns
        -------
        pd.DataFrame
            Cleaned copy of the input DataFrame.
        """
        if self.sex_mode_ is None or self.age_median_ is None:
            raise RuntimeError(
                "DataCleaner instance is not fitted. Call 'fit' before 'transform'."
            )
        df_clean = df.copy()
        df_clean = df_clean.drop(columns=self.columns_to_remove, errors="ignore")

        fill_targets = ("Name", "Breed", "Color")
        fill_values = {col: "Unknown" for col in fill_targets if col in df_clean.columns}
        df_clean = df_clean.fillna(value=fill_values)

        if "SexuponOutcome" in df_clean.columns:
            n_missing_sex = df_clean["SexuponOutcome"].isna().sum()
            df_clean["SexuponOutcome"] = df_clean["SexuponOutcome"].fillna(self.sex_mode_)
            if n_missing_sex:
                logger.info(
                    "Imputed %d missing SexuponOutcome -> mode '%s'",
                    n_missing_sex,
                    self.sex_mode_,
                )

        if "AgeuponOutcome" in df_clean.columns:
            age_days = extract_age_in_days(df_clean["AgeuponOutcome"])
            n_missing_age = age_days.isna().sum()
            age_days = age_days.fillna(self.age_median_)
            if n_missing_age:
                logger.info(
                    "Imputed %d missing AgeuponOutcome -> median %.1f days",
                    n_missing_age,
                    self.age_median_,
                )
            df_clean["log_age_in_days"] = np.log1p(age_days)
            df_clean = df_clean.drop(columns=["AgeuponOutcome"])
        return df_clean