"""
Feature Engineering Module for the Shelter Animal Outcomes Dataset.

This module provides Scikit-Learn compatible transformers designed to extract,
engineer, and encode specific features from cleaned shelter data.

Exported Classes
----------------
TemporalFeaturesExtractor
    A class that transforms raw DateTime strings into cyclic sine/cosine
    coordinates and structural weekend flags.

SexFeaturesExtractor
    A class that simplifies raw sex strings into predictive
    reproductive status categories.

NameFeaturesExtractor
    A transformer that converts textual animal name data into a binary
    indicator representing name presence.

RareCategoriesGrouper
    A stateful transformer that dynamically bins low-frequency categorical
    levels into an 'Other' category based on an information retention threshold.

CategoricalFeaturesEngineer
    An orchestrator that handles primary text extraction, breed mix detection,
    and rare category grouping for high-cardinality columns.

Exported Functions
------------------
extract_primary_color(color_series) -> pd.Series
    Function that isolates the first listed color component from strings
    delimited by a forward slash.

extract_primary_breed(breed_series) -> pd.Series
    Function that extracts the primary breed component and strips the trailing
    'Mix' keyword from string data.

require_columns(X, columns, reason) -> None
    Utility function that raises a ValueError if any specified columns are
    missing from the input DataFrame.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from src import config

logger = logging.getLogger(__name__)

# Overridden by the search grid during training: this default only matters
# when a transformer is built outside the tournament.
DEFAULT_MAX_OTHER_RATIO: float = 0.15
TWO_PI: float = 2 * np.pi
HOUR_FACTOR: float = TWO_PI / 24.0
WDAY_FACTOR: float = TWO_PI / 7.0
# 365.25 averages in the leap day, so the same calendar date lands at
# the same point of the circle in every year
DOY_FACTOR: float = TWO_PI / 365.25

def require_columns(X: pd.DataFrame, columns: tuple[str, ...], reason: str) -> None:
    """Raise unless every named column is present in the DataFrame.


    Parameters
    ----------
    X : pd.DataFrame
        DataFrame whose columns are checked.
    columns : tuple[str, ...]
        Names that must all be present.
    reason : str
        Clause naming what cannot be done without the columns.

    Raises
    ------
    ValueError
        If any of 'columns' is missing, naming all the missing ones.

    Examples
    --------
    >>> import pandas as pd
    >>> require_columns(pd.DataFrame({"a": [1]}), ("a",), "nothing works")
    >>> require_columns(pd.DataFrame({"a": [1]}), ("b",), "nothing works")
    Traceback (most recent call last):
    ValueError: Missing column(s) ('b',): nothing works.
    """
    missing = tuple(col for col in columns if col not in X.columns)
    if missing:
        raise ValueError(f"Missing column(s) {missing}: {reason}.")

@dataclass
class TemporalFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Extract cyclic trigonometric temporal features from datetime column.

    Converts raw datetime strings or pandas Timestamp objects into periodic sine/cosine
    coordinates (Hour, Day of Week, Day of Year) to preserve cyclical continuity,
    while deriving a binary 'IsWeekend' indicator.

    Parameters
    ----------
    datetime_col : str, default= config.DATETIME_COL
        Name of the target raw datetime column to process and drop.

    Attributes
    ----------
    None
        Stateless transformer; no parameters learned during fit.


    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({"DateTime": ["2026-07-06 12:00:00"]})
    >>> extractor = TemporalFeaturesExtractor()
    >>> X_trans = extractor.fit_transform(X)
    >>> "Hour_sin" in X_trans.columns and "IsWeekend" in X_trans.columns
    True
    >>> "DateTime" in X_trans.columns
    False
    """
    datetime_col: str = config.DATETIME_COL

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TemporalFeaturesExtractor":
        """Stateless transformer: returns self without learning parameters."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform raw datetime strings into cyclic sine/cosine coordinates.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the raw target datetime column.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with cyclic features and original datetime column dropped.

        Raises
        ------
        ValueError
            If the specified datetime column is missing from the input DataFrame.
        """
        require_columns(X, (self.datetime_col,), "none of the cyclical features can be derived")
        X_out = X.copy()
        X_out[self.datetime_col] = pd.to_datetime(X_out[self.datetime_col], errors="coerce")
        dt_series = X_out[self.datetime_col]

        hours = dt_series.dt.hour
        X_out[config.HOUR_SIN_COL] = np.sin(hours * HOUR_FACTOR)
        X_out[config.HOUR_COS_COL] = np.cos(hours * HOUR_FACTOR)

        weekday = dt_series.dt.dayofweek
        X_out[config.WDAY_SIN_COL] = np.sin(weekday * WDAY_FACTOR)
        X_out[config.WDAY_COS_COL] = np.cos(weekday * WDAY_FACTOR)

        # float, not int: int cannot hold NaN
        X_out[config.IS_WEEKEND_COL] = np.where(
            weekday.isna(),
            np.nan,
            (weekday >= 5).astype(float)
        )

        doy = dt_series.dt.dayofyear
        X_out[config.DOY_SIN_COL] = np.sin(doy * DOY_FACTOR)
        X_out[config.DOY_COS_COL] = np.cos(doy * DOY_FACTOR)


        return X_out.drop(columns=[self.datetime_col])

@dataclass
class SexFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Extract simplified reproductive status categories from raw animal sex descriptions.

    Parses raw outcome text (e.g., 'Neutered Male', 'Spayed Female', 'Intact Female')
    and maps them into three clean categories: 'Neutered/Spayed', 'Intact', or 'Unknown'.

    Parameters
    ----------
    sex_col : str, default= config.SEX_COL
        Name of the target column containing animal sex information.

    Attributes
    ----------
    None
        Stateless transformer; no parameters learned during fit.


    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({"SexuponOutcome": ["Neutered Male", "Intact Female", None]})
    >>> extractor = SexFeaturesExtractor()
    >>> extractor.fit_transform(X)[config.REPRODUCTIVE_STATUS_COL].tolist()
    ['Neutered/Spayed', 'Intact', 'Unknown']
    """
    sex_col: str = config.SEX_COL

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SexFeaturesExtractor":
        """Stateless transformer: returns self without learning parameters."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Map raw sex strings into config.REPRODUCTIVE_STATUS_COL and drop the source column.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the raw target sex column.

        Returns
        -------
        pd.DataFrame
            Transformed frame with config.REPRODUCTIVE_STATUS_COL, and original sex column dropped.

        Raises
        ------
        ValueError
            If the specified sex column is missing from the input DataFrame.
        """
        require_columns(X, (self.sex_col,), "the reproductive status cannot be derived")
        X_out = X.copy()
        sex_series = X_out[self.sex_col].astype(str)
        is_neutered = sex_series.str.contains(
            "Neutered|Spayed", na=False, case=False
        )
        is_intact = sex_series.str.contains(
            "Intact", na=False, case=False
        )

        X_out[config.REPRODUCTIVE_STATUS_COL] = np.where(
            is_neutered,
            "Neutered/Spayed",
            np.where(is_intact, "Intact", "Unknown"),
        )

        X_out = X_out.drop(columns=[self.sex_col])

        return X_out

@dataclass
class NameFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Convert high-cardinality animal name strings into a binary indicator 'has_name'.

    Reduces high-cardinality noise by determining whether an animal has a valid name
    (non-empty, non-whitespace, and not equal to 'Unknown', case-insensitive). Drops the
    original raw column.

    Parameters
    ----------
    name_col : str, default=config.NAME_COL
        Name of the target text column containing animal names.

    Attributes
    ----------
    None
        Stateless transformer; no parameters learned during fit.

    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({"Name": ["Bella", "   ", "Unknown"]})
    >>> extractor = NameFeaturesExtractor()
    >>> extractor.fit_transform(X)["has_name"].tolist()
    [1, 0, 0]
    """
    name_col: str = config.NAME_COL
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NameFeaturesExtractor":
        """Stateless transformer: returns self without learning parameters."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform name column into binary 'has_name' indicator and drop the original column.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the target name text column.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with binary 'has_name' feature and original name column dropped.

        Raises
        --------
        ValueError
            If the specified name column is missing from the input DataFrame.
        """
        require_columns(X, (self.name_col,), "the name indicator cannot be derived")

        X_out = X.copy()
        clean_name = (
            X_out[self.name_col].fillna("").astype(str).str.strip()
        )

        X_out[config.HAS_NAME_COL] = (
            (clean_name.str.len() > 0)
            & (clean_name.str.lower() != "unknown")
        ).astype(int)

        X_out = X_out.drop(columns=[self.name_col])

        return X_out


def extract_primary_color(color_series: pd.Series) -> pd.Series:
    """Extract the primary color component from a pandas Series.

    Splits slash-separated color strings (e.g., 'Black/White') and retains only
    the primary (first) color, removing surrounding whitespace.

    Parameters
    ----------
    color_series : pd.Series
        Pandas Series containing raw color descriptions.

    Returns
    -------
    pd.Series
        Series with extracted primary colors, preserving missing values and original index.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series(["Black / White", "Blue"])
    >>> extract_primary_color(s).tolist()
    ['Black', 'Blue']
    """
    if color_series.isna().all():
        return color_series
    return color_series.str.split("/").str[0].str.strip()


def extract_primary_breed(breed_series: pd.Series) -> pd.Series:
    """Extract the primary breed component from a pandas Series.

    Splits slash-separated crossbreed descriptions (e.g., 'Labrador/Poodle') and strips
    trailing 'Mix' keywords regardless of casing, leaving clean primary breed names.

    Parameters
    ----------
    breed_series : pd.Series
        Pandas Series containing raw breed descriptions.

    Returns
    -------
    pd.Series
        Series with clean primary breed names, preserving missing values and original index.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series(["Labrador Retriever/German Shepherd", "Chihuahua Mix"])
    >>> extract_primary_breed(s).tolist()
    ['Labrador Retriever', 'Chihuahua']
    """
    if breed_series.isna().all():
        return breed_series
    primary_breed = breed_series.str.split("/").str[0]
    primary_breed = primary_breed.str.replace(
        r"\s+Mix$", "", regex=True, case=False
    )
    return primary_breed.str.strip()

# Column-specific cleaning applied before binning. A column with no entry here
# is binned on its raw values, which is what a caller passing a custom column
# to CategoricalFeaturesEngineer gets.
PRIMARY_EXTRACTORS: dict[str, Callable[[pd.Series], pd.Series]] = {
    config.BREED_COL: extract_primary_breed,
    config.COLOR_COL: extract_primary_color,
}

@dataclass
class RareCategoriesGrouper(BaseEstimator, TransformerMixin):
    """Group low-frequency categorical labels into an 'Other' placeholder.

    Dynamically identifies the minimal set of frequent categories required to cover at
    least (1 - max_other_ratio) of the dataset during training, replacing rare categories
    with 'Other' during transformation to mitigate high cardinality and overfitting.

    Parameters
    ----------
    columns : tuple[str, ...]
        Tuple of target categorical column names to process.
    max_other_ratio : float, default=DEFAULT_MAX_OTHER_RATIO
        Maximum acceptable proportion of dataset instances allowed to be grouped into 'Other'.

    Attributes
    ----------
    frequent_categories_ : dict[str, tuple[str,...]] | None
        Dictionary mapping column names to their tuple of frequent categories learned during fit.

    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({"Breed": ["Labrador"] * 8 + ["Poodle"] * 2})
    >>> grouper = RareCategoriesGrouper(columns=("Breed",), max_other_ratio=0.3)
    >>> grouper = grouper.fit(X)
    >>> grouper.frequent_categories_["Breed"]
    ('Labrador',)
    >>> grouper.transform(X)["Breed"].value_counts().to_dict()
    {'Labrador': 8, 'Other': 2}
    """

    columns: tuple[str, ...]
    max_other_ratio: float = DEFAULT_MAX_OTHER_RATIO
    frequent_categories_: dict[str, tuple[str, ...]] | None = field(
        default=None, init=False, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RareCategoriesGrouper":
        """Identify frequent categories per column to satisfy the information retention constraint.

        Calculates cumulative frequency distributions for specified columns and identifies
        the minimal subset of categories required to cover at least (1 - max_other_ratio)
        of the training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training DataFrame containing target categorical columns.
        y : None, optional
            Ignored. Included for scikit-learn API compatibility.

        Returns
        -------
        RareCategoriesGrouper
            Fitted instance with learned frequent_categories_ dictionary.

        Raises
        ------
        ValueError
            If any required column specified in self.columns is missing from X.
        """
        require_columns(X, self.columns, "the frequent categories cannot be learned")

        self.frequent_categories_ = {}

        for col in self.columns:
            freqs = X[col].value_counts(normalize=True)
            if freqs.empty:
                self.frequent_categories_[col] = ()
                logger.warning("Column '%s' is empty or contains only NaNs during fit.", col)
                continue
            cum_sum = freqs.cumsum()

            target_ratio = 1.0 - self.max_other_ratio

            frequent = tuple(
                cum_sum[cum_sum.shift(fill_value=0) < target_ratio].index
            )

            self.frequent_categories_[col] = frequent
            logger.info(
                "Fitted RareCategoriesGrouper for '%s': kept %d categories",
                col, len(frequent)
            )
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace rare categories in specified columns with the 'Other' placeholder.

        Maps values not present in frequent_categories_ to 'Other' while preserving
        NaN entries and original indices. Emits a DEBUG log if the resulting 'Other'
        proportion exceeds max_other_ratio due to data drift.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to transform (train, validation, or test).

        Returns
        -------
        pd.DataFrame
            Transformed copy of the input DataFrame with rare labels binned.

        Raises
        ------
        NotFittedError
            If transform is called before the transformer is fitted.

        ValueError
            If a column that was fitted is missing from X, since skipping it
        would let unbinned categories reach the encoder unnoticed.
        """
        if self.frequent_categories_ is None:
            raise NotFittedError(
                "RareCategoriesGrouper instance is not fitted. Call 'fit' before 'transform'."
            )

        require_columns(X, self.columns, "the categories fitted on them cannot be binned")

        X_out = X.copy()
        if X_out.empty:
            return X_out

        for col in self.columns:
            frequent = self.frequent_categories_[col]

            mask = X_out[col].isin(frequent) | X_out[col].isna()
            X_out[col] = X_out[col].where(mask, "Other")

            other_ratio = (X_out[col] == "Other").mean()
            if other_ratio > self.max_other_ratio:
                logger.debug(
                        "Column '%s' has an 'Other' ratio of %.3f, which "
                        "exceeds the configured max_other_ratio of %.3f.",
                        col,
                        other_ratio,
                        self.max_other_ratio,
                    )

        return X_out

@dataclass
class CategoricalFeaturesEngineer(BaseEstimator, TransformerMixin):
    """Feature engineer for high-cardinality categorical variables (breed, color).

    Orchestrates the derivation of a binary config.IS_MIX_COL indicator, extracts clean
    primary breed/color values, and leverages an internal RareCategoriesGrouper instance
    to group rare labels.

    Parameters
    ----------

    columns : tuple[str,...], default= config.CATEGORICAL_COLS
        Target high-cardinality categorical columns to process.
    max_other_ratio : float, default=DEFAULT_MAX_OTHER_RATIO
        Maximum proportion threshold passed to the underlying RareCategoriesGrouper.

    Attributes
    ----------
    grouper_ : RareCategoriesGrouper | None
        Fitted internal RareCategoriesGrouper instance.

    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({"Breed": ["Labrador Mix", "Poodle"], "Color": ["Black/White", "Red"]})
    >>> engineer = CategoricalFeaturesEngineer()
    >>> X_trans = engineer.fit_transform(X)
    >>> config.IS_MIX_COL in X_trans.columns
    True
    """
    columns: tuple[str, ...] = config.CATEGORICAL_COLS
    max_other_ratio: float = DEFAULT_MAX_OTHER_RATIO
    grouper_: RareCategoriesGrouper | None = field(
        default=None, init=False, repr=False
        )

    def _extract_primary_forms(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy where each declared column is reduced to its primary form.

        A column with no rule in PRIMARY_EXTRACTORS is left as it is, which is
        what a caller declaring a column of its own gets.
        """
        X_clean = X.copy()
        for col in self.columns:
            cleaner = PRIMARY_EXTRACTORS.get(col)
            if cleaner is not None and col in X_clean.columns:
                X_clean[col] = cleaner(X_clean[col])
        return X_clean

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None)-> "CategoricalFeaturesEngineer":
        """Fit internal RareCategoriesGrouper on primary representations of target columns.

        Each declared column is first reduced to its primary form when a rule
        for it exists in PRIMARY_EXTRACTORS, then the internal
        RareCategoriesGrouper learns the frequent categories from the result.

        Parameters
        ----------
        X : pd.DataFrame
            Training DataFrame containing every column named in 'columns'.
        y : pd.Series | None, optional
            Ignored. Included for scikit-learn API compatibility.

        Returns
        -------
        CategoricalFeaturesEngineer
            Fitted instance with an initialized and fitted grouper_.

        Raises
        ------
        ValueError
            If a column named in columns is missing from X, raised by the
            internal grouper.
        """
        X_temp = self._extract_primary_forms(X)
        self.grouper_ = RareCategoriesGrouper(
            columns=self.columns,
            max_other_ratio=self.max_other_ratio
        )
        self.grouper_.fit(X_temp)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract features, simplify categorical values, and group rare categories.

        Adds a binary indicator telling whether the breed was recorded as a
        cross, reduces each declared column to its primary form, then delegates
        the binning of rare categories to the fitted internal grouper.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to transform (train, validation, or test).

        Returns
        -------
        pd.DataFrame
            Transformed copy of the input DataFrame with new indicators and binned categories.

        Raises
        ------
        NotFittedError
            If transform is called before the transformer is fitted.
        ValueError
            If a column that was fitted is missing from X, raised by the
            internal grouper.
        """
        if self.grouper_ is None:
            raise NotFittedError(
                "CategoricalFeaturesEngineer instance is not fitted. Call 'fit' before 'transform'."
            )

        X_out = X.copy()
        if config.BREED_COL in X_out.columns:
            breed_str = X_out[config.BREED_COL].fillna("").astype(str)
            is_mix_series = breed_str.str.contains(
                "Mix", na=False, case=False
            ) | breed_str.str.contains("/", na=False)
            X_out[config.IS_MIX_COL] = is_mix_series.astype(int)

        return self.grouper_.transform(self._extract_primary_forms(X_out))
