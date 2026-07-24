"""
Feature Engineering and Transformation Module for the Shelter Animal Outcomes Dataset.

This module provides Scikit-Learn compatible transformers designed to extract, 
engineer, and encode specific features from cleaned shelter data.

Exported Classes
----------------
TemporalFeaturesExtractor
    A class that transforms raw DateTime strings into cyclic sine/cosine 
    coordinates and structural weekend flags.

RareCategoriesGrouper
    A stateful transformer that dynamically bins low-frequency categorical 
    levels into an 'Other' category based on an information retention threshold.

CategoricalFeaturesEngineer
    An orchestrator that handles primary text extraction, breed mix detection, 
    and rare category grouping for high-cardinality columns.

SexFeaturesExtractor
    A class that simplifies raw sex-upon-outcome strings into predictive 
    reproductive status categories.

NameFeaturesExtractor
    A transformer that converts textual animal name data into a binary 
    indicator representing name presence.

Exported Functions
------------------
extract_primary_color(color_series: pd.Series) -> pd.Series
    Function that isolates the first listed color component from strings 
    delimited by a forward slash.

extract_primary_breed(breed_series: pd.Series) -> pd.Series
    Function that extracts the primary breed component and strips the trailing 
    'Mix' keyword from string data.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from dataclasses import dataclass, field
logger = logging.getLogger(__name__)

@dataclass
class TemporalFeaturesExtractor(TransformerMixin, BaseEstimator):
    """Extract cyclic trigonometric and high-level operational temporal features from DateTime.

    Converts raw datetime strings or pandas Timestamp objects into periodic sine/cosine
    coordinates (Hour, Day of Week, Day of Year) to preserve cyclical continuity for linear
    and tree-based models, while deriving a binary 'IsWeekend' indicator.

    Parameters
    ----------
    datetime_col : str, default="DateTime"
        Name of the target raw datetime column to process and drop.

    Attributes
    ----------
    None
        Stateless transformer; no parameters learned during fit.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"DateTime": ["2026-07-06 12:00:00"]})
    >>> extractor = TemporalFeaturesExtractor()
    >>> df_trans = extractor.fit_transform(df)
    >>> "Hour_sin" in df_trans.columns and "IsWeekend" in df_trans.columns
    True
    >>> "DateTime" in df_trans.columns
    False
    """
    datetime_col: str = "DateTime"

    def fit(self, X: pd.DataFrame, y=None) -> "TemporalFeaturesExtractor":
        """Stateless transformer: returns self without learning parameters."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform raw DateTime strings into cyclic sine/cosine coordinates.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the raw target datetime column.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with cyclic features and original datetime column dropped.
        """
        if self.datetime_col not in X.columns:
            return X
        X_out=X.copy()
        X_out[self.datetime_col] = pd.to_datetime(X_out[self.datetime_col])
        dt_series = X_out[self.datetime_col]
        two_pi = 2 * np.pi
        HOUR_FACTOR = two_pi / 24.0
        WDAY_FACTOR = two_pi / 7.0
        DOY_FACTOR = two_pi / 365.25

        hours = dt_series.dt.hour
        X_out["Hour_sin"] = np.sin(hours * HOUR_FACTOR)
        X_out["Hour_cos"] = np.cos(hours * HOUR_FACTOR)

        weekday = dt_series.dt.dayofweek
        X_out["Wday_sin"] = np.sin(weekday * WDAY_FACTOR)
        X_out["Wday_cos"] = np.cos(weekday * WDAY_FACTOR)


        X_out["IsWeekend"] = np.where(
            weekday.isna(),
            np.nan,
            (weekday >= 5).astype(float)
        )

        doy = dt_series.dt.dayofyear
        X_out["DoY_sin"] = np.sin(doy * DOY_FACTOR)
        X_out["DoY_cos"] = np.cos(doy * DOY_FACTOR)


        return X_out.drop(columns=[self.datetime_col])

@dataclass
class SexFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Extract simplified reproductive status categories from raw animal sex descriptions.

    Parses raw outcome text (e.g., 'Neutered Male', 'Spayed Female', 'Intact Female')
    and maps them into three clean categories: 'Neutered/Spayed', 'Intact', or 'Unknown'.

    Parameters
    ----------
    sex_col : str, default="SexuponOutcome"
        Name of the target column containing animal sex information.

    Attributes
    ----------
    None
        Stateless transformer; no parameters learned during fit.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"SexuponOutcome": ["Neutered Male", "Intact Female", None]})
    >>> extractor = SexFeaturesExtractor()
    >>> extractor.fit_transform(df)["Reproductive_Status"].tolist()
    ['Neutered/Spayed', 'Intact', 'Unknown']
    """
    sex_col: str = "SexuponOutcome"
    
    def fit(self, X: pd.DataFrame, y=None) -> "SexFeaturesExtractor":
        """Stateless transformer: returns self without learning parameters."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Map raw sex strings into 'Reproductive_Status' and drop the source column.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the target sex/reproductive status column.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with 'Reproductive_Status' derived and original sex column dropped.
        """
        if self.sex_col not in X.columns:
            return X
        X_out=X.copy()
        is_neutered = X_out[self.sex_col].str.contains("Neutered|Spayed", na=False, case=False)
        is_intact = X_out[self.sex_col].str.contains("Intact", na=False, case=False)
        
        X_out["Reproductive_Status"] = np.where(
            is_neutered, "Neutered/Spayed", 
            np.where(is_intact, "Intact", "Unknown")
        )

        X_out = X_out.drop(columns=[self.sex_col])

        return X_out
    
@dataclass
class NameFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Convert high-cardinality animal name strings into a binary indicator 'has_name'.

    Reduces high-cardinality noise by determining whether an animal has a valid name
    (non-empty, non-whitespace, and not equal to 'Unknown'). Drops the original raw column.

    Parameters
    ----------
    name_col : str, default="Name"
        Name of the target text column containing animal names.

    Attributes
    ----------
    None
        Stateless transformer; no parameters learned during fit.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Name": ["Bella", "   ", "Unknown"]})
    >>> extractor = NameFeaturesExtractor()
    >>> extractor.fit_transform(df)["has_name"].tolist()
    [1, 0, 0]
    """
    name_col: str = "Name"
    def fit(self, X: pd.DataFrame, y=None) -> "NameFeaturesExtractor":
        """Stateless transformer: returns self without learning parameters."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform 'Name' into binary 'has_name' indicator and drop the original column.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing the target name text column.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with binary 'has_name' feature and original name column dropped.
        """
        if self.name_col not in X.columns:
            return X
        
        X_out=X.copy()
        clean_name = (
            X_out[self.name_col].fillna("").astype(str).str.strip()
        )

        X_out["has_name"] = (
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
    if color_series.empty or color_series.isna().all():
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
    if breed_series.empty or breed_series.isna().all():
        return breed_series
    primary_breed = breed_series.str.split("/").str[0]
    primary_breed = primary_breed.str.replace(
        r"\s+Mix$", "", regex=True, case=False
    )
    return primary_breed.str.strip()


@dataclass
class RareCategoriesGrouper(BaseEstimator, TransformerMixin):
    """Group low-frequency categorical labels into an 'Other' placeholder.

    Dynamically identifies the minimal set of frequent categories required to cover at
    least (1 - max_other_ratio) of the dataset during training, replacing rare categories
    with 'Other' during transformation to mitigate high cardinality and overfitting.

    Parameters
    ----------
    columns : list[str]
        List of target categorical column names to process.
    max_other_ratio : float, default=0.15
        Maximum acceptable proportion of dataset instances allowed to be grouped into 'Other'.

    Attributes
    ----------
    frequent_categories_ : dict[str, list] | None
        Dictionary mapping column names to their list of frequent categories learned during fit.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Breed": ["Labrador"] * 8 + ["Poodle"] * 2})
    >>> grouper = RareCategoriesGrouper(columns=["Breed"], max_other_ratio=0.3)
    >>> _ = grouper.fit(df)
    >>> grouper.frequent_categories_["Breed"]
    ['Labrador']
    """

    columns: list[str]
    max_other_ratio: float = 0.15
    frequent_categories_: dict[str, list] | None = field(default=None, init=False, repr=False)

    def fit(self, X: pd.DataFrame, y=None) -> "RareCategoriesGrouper":
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
            Fitted instance with learned `frequent_categories_` dictionary.

        Raises
        ------
        ValueError
            If any required column specified in `self.columns` is missing from `X`.
         """
       self.frequent_categories_ = {}

       for col in self.columns:
            if col not in X.columns:
             raise ValueError(
            f"Required column '{col}' is missing from the training DataFrame during fit. "
            f"Please check your input features configuration."
        )

            freqs = X[col].value_counts(normalize=True)
            if freqs.empty:
                self.frequent_categories_[col] = []
                logger.warning(f"Column '{col}' is empty or contains only NaNs during fit.")
                continue
            cum_sum = freqs.cumsum()

            target_ratio = 1.0 - self.max_other_ratio

            frequent = cum_sum[
                    cum_sum.shift(fill_value=0) < target_ratio
                ].index.tolist()

            self.frequent_categories_[col] = frequent
            logger.info(
                "Fitted RareCategoriesGrouper for '%s': kept %d categories",
                col, len(frequent)
            )
       return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace rare categories in specified columns with the 'Other' placeholder.

        Maps values not present in `frequent_categories_` to 'Other' while preserving
        NaN entries and original indices. Emits a DEBUG log if the resulting 'Other'
        proportion exceeds `max_other_ratio` due to data drift.

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
        RuntimeError
            If transform is called before the transformer is fitted.
        """
        if self.frequent_categories_ is None:
            raise RuntimeError(
                "RareCategoriesGrouper instance is not fitted. Call 'fit' before 'transform'."
            )

        X_out=X.copy()
        if X_out.empty:
            return X_out
        
        for col in self.columns:
            if col in X_out.columns and col in self.frequent_categories_:
                frequent = self.frequent_categories_[col]
                X_out[col] = np.where(
                    X_out[col].isin(frequent) | X[col].isna(),
                    X_out[col],
                    "Other",
                )

                other_ratio = (X_out[col] == "Other").mean()
                if other_ratio > self.max_other_ratio:
                  msg = (
                        f"Column '{col}' has an 'Other' ratio of {other_ratio:.3f}, "
                        f"which exceeds the configured max_other_ratio of {self.max_other_ratio:.3f}."
                    )
                  logger.debug(msg)
                    
        return X_out
    
@dataclass
class CategoricalFeaturesEngineer(BaseEstimator, TransformerMixin):
    """Feature engineer for high-cardinality categorical variables ('Breed', 'Color').

    Orchestrates the derivation of a binary 'is_mix' indicator, extracts clean primary breed/color
    values, and leverages an internal `RareCategoriesGrouper` instance to group rare labels.

    Parameters
    ----------
    columns : list[str], default=["Breed", "Color"]
        Target high-cardinality categorical columns to process.
    max_other_ratio : float, default=0.15
        Maximum proportion threshold passed to the underlying RareCategoriesGrouper.

    Attributes
    ----------
    grouper_ : RareCategoriesGrouper | None
        Fitted internal RareCategoriesGrouper instance.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Breed": ["Labrador Mix", "Poodle"], "Color": ["Black/White", "Red"]})
    >>> engineer = CategoricalFeaturesEngineer()
    >>> df_trans = engineer.fit_transform(df)
    >>> "is_mix" in df_trans.columns
    True
    """
    columns: list[str] = field(default_factory=lambda: ["Breed", "Color"])
    max_other_ratio: float = 0.15
    grouper_: RareCategoriesGrouper | None = field(default=None, init=False, repr=False)

    def fit(self, X: pd.DataFrame, y=None)-> "CategoricalFeaturesEngineer":
        """Fit internal RareCategoriesGrouper on primary representations of target columns.

        Cleans 'Breed' and 'Color' entries down to their primary forms, then fits
        the internal `RareCategoriesGrouper` on all target columns present in `X`.

        Parameters
        ----------
        X : pd.DataFrame
            Training DataFrame containing 'Breed' and/or 'Color' columns.
        y : None, optional
            Ignored. Included for scikit-learn API compatibility.

        Returns
        -------
        CategoricalFeaturesEngineer
            Fitted instance with an initialized and fitted `grouper_`.
        """
        X_temp = X.copy()
        existing_cols = [col for col in self.columns if col in X_temp.columns]
        if "Breed" in X_temp.columns:
            X_temp["Breed"] = extract_primary_breed(X_temp["Breed"])
        if "Color" in X_temp.columns:
            X_temp["Color"] = extract_primary_color(X_temp["Color"])

        self.grouper_ = RareCategoriesGrouper(
            columns=existing_cols, max_other_ratio=self.max_other_ratio
        )
        self.grouper_.fit(X_temp)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract features, simplify categorical values, and group rare categories.

        Derives a binary 'is_mix' feature if 'Breed' is present, converts 'Breed' and
        'Color' to their primary representations, and delegates rare label binning
        to the fitted internal `RareCategoriesGrouper`.

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
        RuntimeError
            If transform is called before the transformer is fitted.
        """
        if self.grouper_ is None:
            raise RuntimeError(
                "CategoricalFeaturesEngineer instance is not fitted. Call 'fit' before 'transform'."
            )
        
        X_out=X.copy()
        if "Breed" in X.columns:
            is_mix_series = X_out["Breed"].str.contains(
                "Mix", na=False, case=False
            ) | X_out["Breed"].str.contains("/", na=False)
            X_out["is_mix"] = is_mix_series.astype(int)

            X_out["Breed"] = extract_primary_breed(X["Breed"])

        if "Color" in X_out.columns:
            X_out["Color"] = extract_primary_color(X_out["Color"])


        return self.grouper_.transform(X_out)
    
