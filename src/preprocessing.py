"""
Preprocessing module for the Shelter Animal Outcomes Dataset.
This module provides the data cleaning steps that prepare raw shelter data for machine learning.

Exported Classes
----------------
DataCleaner
    A scikit-learn compatible transformer that orchestrates column dropping,
    imputations and log-transformation of the age feature.

Exported Functions
------------------
extract_age_in_days(age_series) -> pd.Series
    Function that parses textual age strings
    (e.g., '2 years', '3 weeks') into equivalent numeric float days.

drop_rows_missing_required(X, y, row_required_cols) -> tuple[pd.DataFrame, pd.Series]
    Function that removes the rows whose required columns are missing or unusable,

Note on the module boundary
---------------------------
Creating log_age_in_days is, strictly speaking, feature engineering: it
derives a new column rather than repairing an existing one. It lives here
because the age column is treated as a whole: the raw text is parsed into
days, the gaps are filled with the median learned during the fit, and the
logarithm is applied to that same scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from src import config

logger = logging.getLogger(__name__)

# Here 365 is used instead of 365.25, considering that the dataset already
# contains the writer's own rounding, so half a day of extra precision
# would be invented, not measured.
DAYS_PER_UNIT: dict[str, float] = {
    "year": 365.0,
    "month": 30.0,
    "week": 7.0,
    "day": 1.0,
}

_NUMBER_PATTERN = r"(\d+(?:\.\d+)?)"
_UNIT_PATTERN = r"\b(year|month|week|day)s?\b"

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

    text = age_series.astype(str).str.lower()

    numeric_values = pd.to_numeric(
        text.str.extract(_NUMBER_PATTERN, expand=False),
        errors="coerce"
    )

    # The unit is the first one appearing in the string. Compound values like
    # "1 year 6 months" do not occur in this dataset, where every entry is a
    # single number followed by a single unit.
    units = text.str.extract(_UNIT_PATTERN, expand=False)

    return (numeric_values * units.map(DAYS_PER_UNIT)).astype(float)


def drop_rows_missing_required(
    X: pd.DataFrame,
    y: pd.Series,
    row_required_cols: tuple[str, ...] = config.ROW_REQUIRED_COLS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Function that removes the rows whose required columns are missing or unusable.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target values, aligned with 'X'.
    row_required_cols : tuple[str, ...], default=config.ROW_REQUIRED_COLS
        Columns without which a row cannot be used at all.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        The surviving rows, both re-indexed from zero.

    Raises
    ------
    KeyError
        If any of row_required_cols is absent from X
        
    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ...     "DateTime": ["2015-01-01 10:00:00", "not a date", "2015-01-03 12:00:00"],
    ...     "AnimalType": ["Dog", "Cat", None],
    ... })
    >>> y = pd.Series(["Adoption", "Transfer", "Adoption"])
    >>> X_clean, y_clean = drop_rows_missing_required(X, y)
    >>> X_clean
                  DateTime AnimalType
    0  2015-01-01 10:00:00        Dog
    >>> list(y_clean)
    ['Adoption']
    """
    mask = X[list(row_required_cols)].notna().all(axis=1)

    # notna() above only catches empty cells: a string like "not a date" is not
    # empty, so it survives. Therefore:
    if config.DATETIME_COL in X.columns:
        mask &= pd.to_datetime(X[config.DATETIME_COL], errors="coerce").notna()

    n_dropped = int((~mask).sum())
    if n_dropped:
        logger.info(
            "Dropped %d rows with missing or unparsable columns %s",
            n_dropped, row_required_cols,
        )

    return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)

@dataclass
class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Clean and impute raw shelter animal data for machine learning.

    Orchestrates the initial cleaning phase of the pipeline by removing identifier
    column, imputing categorical features with fixed labels
    or learned modes, and transforming ages into log-scaled numeric days.

    Parameters
    ----------
    sex_col : str, default=config.SEX_COL
            Name of the column containing sex and reproductive outcome data.
    age_col : str, default=config.AGE_COL
            Name of the column containing textual age values.
    fill_targets : tuple[str, ...], default=config.FILL_TARGETS
            Tuple of categorical column names to preventively fill missing values with 'Unknown'.
    columns_to_remove : tuple[str, ...], default=config.COLUMNS_TO_REMOVE
           Tuple of identifier or noisy column names to drop from input DataFrames

    Attributes
    ----------
    sex_mode_ : str | None
        The most frequent value learned from sex column during fitting.
    age_median_ : float | None
        The median age in days learned from age column during fitting.

    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ...     "AnimalID": ["A1", "A2"],
    ...     "SexuponOutcome": ["Neutered Male", None],
    ...     "AgeuponOutcome": ["2 years", None]
    ... })
    >>> cleaner = DataCleaner()
    >>> cleaner.fit_transform(X)
      SexuponOutcome  log_age_in_days
    0  Neutered Male         6.594413
    1  Neutered Male         6.594413

    Notes
    -----
    The column names read from the input are configurable, but the name of the
    column written out is not: it is fixed to config.LOG_AGE_COL, which is
    the name the ColumnTransformer downstream looks for when scaling.

    """
    sex_col: str = config.SEX_COL
    age_col: str = config.AGE_COL
    fill_targets: tuple[str, ...] = config.FILL_TARGETS
    columns_to_remove: tuple[str, ...] = config.COLUMNS_TO_REMOVE

    sex_mode_: str | None = field(default=None, init=False, repr=False)
    age_median_: float | None = field(default=None, init=False, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DataCleaner":
        """Learn imputation statistics (mode for sex, median age in days).

        Parameters
        ----------
        X : pd.DataFrame
            Training DataFrame.

        y : pd.Series | None, optional
            Ignored. Included for scikit-learn compatibility.

        Returns
        -------
        DataCleaner
            Fitted instance of the transformer.
        """
        if self.sex_col in X.columns:
            modes = X[self.sex_col].mode()
            self.sex_mode_ = modes.iloc[0] if not modes.empty else "Unknown"
        else:
            self.sex_mode_ = "Unknown"

        if self.age_col in X.columns:
            valid_ages = extract_age_in_days(X[self.age_col]).dropna()
            self.age_median_ = float(valid_ages.median()) if not valid_ages.empty else 0.0
        else:
            self.age_median_ = 0.0
        logger.info(
            "Fitted DataCleaner: sex_mode_='%s', age_median_=%.1f days",
            self.sex_mode_,
            self.age_median_,
         )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned statistics to clean and impute the dataset.

        Drops specified columns, fills missing categorical values (name, breed,
        color and sex), converts textual ages to days, imputes missing ages
        with the fitted median, and applies a log(1 + x) transformation.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to clean (train, validation or test).

        Returns
        -------
        pd.DataFrame
            Cleaned copy of the input DataFrame.

        Raises
        ------
        NotFittedError
            If transform() is called before the transformer is fitted.
        """
        if self.sex_mode_ is None or self.age_median_ is None:
            raise NotFittedError(
                "DataCleaner instance is not fitted. Call 'fit' before 'transform'."
            )
        X_clean = X.copy()
        X_clean = X_clean.drop(columns=list(self.columns_to_remove), errors="ignore")

        # Runs before the mode imputation below so a column listed in fill_targets never reaches it:
        # config keeps sex out of that tuple (test_the_sex_column_is_not_filled_with_a_literal).
        fill_values = {col: "Unknown" for col in self.fill_targets if col in X_clean.columns}
        X_clean = X_clean.fillna(value=fill_values)

        if self.sex_col in X_clean.columns:
            n_missing_sex = X_clean[self.sex_col].isna().sum()
            X_clean[self.sex_col] = X_clean[self.sex_col].fillna(self.sex_mode_)
            if n_missing_sex:
                logger.info(
                    "Imputed %d missing %s -> mode '%s'",
                    n_missing_sex,
                    self.sex_col,
                    self.sex_mode_,
                )

        if self.age_col in X_clean.columns:
            age_days = extract_age_in_days(X_clean[self.age_col])
            n_missing_age = age_days.isna().sum()
            age_days = age_days.fillna(self.age_median_)
            if n_missing_age:
                logger.info(
                    "Imputed %d missing %s -> median %.1f days",
                    n_missing_age,
                    self.age_col,
                    self.age_median_,
                )

            X_clean[config.LOG_AGE_COL] = np.log1p(age_days)
            X_clean = X_clean.drop(columns=[self.age_col])

        return X_clean
