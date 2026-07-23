"""
Preprocessing module for the Shelter Animal Outcomes Dataset.

This module provides the primary data cleaning pipeline and custom transformers 
required to prepare raw data for machine learning algorithms. 

Exported Classes
----------------
DataCleaner
    A scikit-learn compatible transformer that orchestrates column dropping,
    imputations and log-transformation of age in days..

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
from sklearn.base import TransformerMixin, BaseEstimator

logger = logging.getLogger(__name__)

def extract_age_in_days(age_series: pd.Series) -> pd.Series:
    """
    Convert a textual Series of age (e.g., '2 years') 
    into a numeric Series of days (float).
    
    Parameters
    ----------
    age_series : pd.Series
        Pandas Series containing textual age representations (e.g., '2 years').
        
    Returns
    -------
    pd.Series
        Numeric Series with age values converted to floating-point days.
        Unparseable, empty, or NaN entries are returned as NaN.

    Examples
    --------
    >>> import pandas as pd
    >>> ages = pd.Series(["2 years", "1 month", "3 weeks", "4 days", None])
    >>> extract_age_in_days(ages)
    0    730.0
    1     30.0
    2     21.0
    3      4.0
    4      NaN
    dtype: float64
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
 
@dataclass
class DataCleaner(TransformerMixin, BaseEstimator):
    """
    Clean and impute raw shelter animal data for machine learning.

    Orchestrates the initial cleaning phase of the pipeline by removing identifier
    column, imputing categorical features with fixed labels
    or learned modes, and transforming textual ages into log-scaled numeric days.

    Parameters
    ----------
    columns_to_remove : list[str], default=["AnimalID"]
        List of column names to drop from input DataFrames to prevent noise.

    Attributes
    ----------
    sex_mode_ : str | None
        The most frequent value learned from 'SexuponOutcome' during fitting.
    age_median_ : float | None
        The median age in days learned from 'AgeuponOutcome' during fitting.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "AnimalID": ["A1", "A2"],
    ...     "SexuponOutcome": ["Neutered Male", None],
    ...     "AgeuponOutcome": ["2 years", None]
    ... })
    >>> cleaner = DataCleaner()
    >>> cleaner.fit_transform(df)
      SexuponOutcome  log_age_in_days
    0  Neutered Male         6.594413
    1  Neutered Male         6.594413

    """

    columns_to_remove: list[str] = field(
    default_factory=lambda: ["AnimalID"])

    sex_mode_: str | None = field(default=None, init=False, repr=False)
    age_median_: float | None = field(default=None, init=False, repr=False)

    def fit(self, df: pd.DataFrame, y=None) -> "DataCleaner":
       """Learn imputation statistics (mode for sex, median age in days).

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.

        y : None, optional
            Ignored. Included for scikit-learn compatibility.

        Returns
        -------
        DataCleaner
            Fitted instance of the transformer.
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

        Drops specified columns, fills missing categorical values ('Name', 'Breed',
        'Color', 'SexuponOutcome'), converts textual ages to days, imputes missing ages
        with the fitted median, and applies a log(1 + x) transformation.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to clean (train, validation or test).

        Returns
        -------
        pd.DataFrame
            Cleaned copy of the input DataFrame.
        
        Raises
        ------
        RuntimeError
            If transform() is called before the transformer is fitted.
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