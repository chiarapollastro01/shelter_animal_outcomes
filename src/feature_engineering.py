"""
Feature Engineering and Transformation Module for the Shelter Animal Outcomes Dataset.

This module provides custom Scikit-Learn compliant transformers designed to extract, 
engineer, and encode domain-specific features from cleaned shelter data.

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
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from dataclasses import dataclass, field
logger = logging.getLogger(__name__)

@dataclass
class TemporalFeaturesExtractor(TransformerMixin, BaseEstimator):
    """Extracts cyclic and high-level operational temporal features from DateTime."""
    datetime_col: str = "DateTime"

    def fit(self, df: pd.DataFrame, y=None) -> "TemporalFeaturesExtractor":
        """Stateless transformer: nothing to learn during fit."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw DateTime strings into cyclic sine/cosine coordinates."""
        if self.datetime_col not in df.columns:
            return df

        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        dt_series = df[self.datetime_col]
        two_pi = 2 * np.pi

        hours = dt_series.dt.hour
        df["Hour_sin"] = np.sin(two_pi * hours / 24)
        df["Hour_cos"] = np.cos(two_pi * hours / 24)

        weekday = dt_series.dt.dayofweek
        df["Wday_sin"] = np.sin(two_pi * weekday / 7)
        df["Wday_cos"] = np.cos(two_pi * weekday / 7)


        df["IsWeekend"] = np.where(
            weekday.isna(),
            np.nan,
            (weekday >= 5).astype(float)
        )

        doy = dt_series.dt.dayofyear
        df["DoY_sin"] = np.sin(two_pi * doy / 365.25)
        df["DoY_cos"] = np.cos(two_pi * doy / 365.25)


        return df.drop(columns=[self.datetime_col])


def extract_primary_color(color_series: pd.Series) -> pd.Series:
    """Extract the primary color from a Series by splitting on '/'
       and taking only the first color.
    """
    return color_series.str.split("/").str[0].str.strip()


def extract_primary_breed(breed_series: pd.Series) -> pd.Series:
    """Extract the primary breed from a Series by splitting on '/'
      and stripping the trailing 'Mix' keyword.
    """
    primary_breed = breed_series.str.split("/").str[0]
    primary_breed = primary_breed.str.replace(
        r"\s+Mix$", "", regex=True, case=False
    )
    return primary_breed.str.strip()


@dataclass
class RareCategoriesGrouper(BaseEstimator, TransformerMixin):
    """Dynamic categorical grouper that solves high-cardinality issues.

    Keeps the most frequent categories required to cover at least (1 -
    max_other_ratio) of the dataset, grouping the remaining rare ones into
    'Other'.
    """

    columns: list[str]
    max_other_ratio: float = 0.15
    frequent_categories_: dict[str, list] | None = field(default=None, init=False, repr=False)
    def fit(self, X: pd.DataFrame, y=None) -> "RareCategoriesGrouper":
       """Identify the minimal set of categories needed to meet the
           information retention constraint.
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
        """Replace rare categories in the specified columns with 'Other'."""
        if self.frequent_categories_ is None:
            raise RuntimeError(
                "RareCategoriesGrouper instance is not fitted. Call 'fit' before 'transform'."
            )


        if X.empty:
            return X
        
        for col in self.columns:
            if col in X.columns and col in self.frequent_categories_:
                frequent = self.frequent_categories_[col]
                X[col] = np.where(
                    X[col].isin(frequent) | X[col].isna(),
                    X[col],
                    "Other",
                )

                other_ratio = (X[col] == "Other").mean()
                if other_ratio > self.max_other_ratio:
                  msg = (
                        f"Column '{col}' has an 'Other' ratio of {other_ratio:.3f}, "
                        f"which exceeds the configured max_other_ratio of {self.max_other_ratio:.3f}."
                    )
                  logger.warning(msg)
                  warnings.warn(msg, RuntimeWarning)
                    
        return X
    
@dataclass
class CategoricalFeaturesEngineer(BaseEstimator, TransformerMixin):
    """Feature engineer for high-cardinality columns (Breed, Color).

    Extracts clean primary representations, determines if the animal is a mix,
    and dynamically bins low-frequency categories to prevent overfitting.
    """
    columns: list[str] = field(default_factory=lambda: ["Breed", "Color"])
    max_other_ratio: float = 0.15
    grouper_: RareCategoriesGrouper | None = field(default=None, init=False, repr=False)

    def fit(self, X: pd.DataFrame, y=None)-> "CategoricalFeaturesEngineer":
        """Fit the internal RareCategoriesGrouper on primary representations of
           Breed and Color.
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
        """Transform high-cardinality categorical variables, replacing them

        with clean primary values and grouping rare ones.
        """
        if self.grouper_ is None:
            raise RuntimeError(
                "CategoricalFeaturesEngineer instance is not fitted. Call 'fit' before 'transform'."
            )
        

        if "Breed" in X.columns:
            is_mix_series = X["Breed"].str.contains(
                "Mix", na=False, case=False
            ) | X["Breed"].str.contains("/", na=False)
            X["is_mix"] = is_mix_series.astype(int)

            X["Breed"] = extract_primary_breed(X["Breed"])

        if "Color" in X.columns:
            X["Color"] = extract_primary_color(X["Color"])

            X = self.grouper_.transform(X)

        return X
    
@dataclass
class SexFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Feature extractor focused solely on the highly predictive reproductive status."""
    sex_col: str = "SexuponOutcome"
    
    def fit(self, df: pd.DataFrame, y=None) -> "SexFeaturesExtractor":
        """Stateless transformer: nothing to learn during fit."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map raw sex strings to simplified reproductive states."""
        if self.sex_col not in df.columns:
            return df

        is_neutered = df[self.sex_col].str.contains("Neutered|Spayed", na=False, case=False)
        is_intact = df[self.sex_col].str.contains("Intact", na=False, case=False)
        
        df["Reproductive_Status"] = np.where(
            is_neutered, "Neutered/Spayed", 
            np.where(is_intact, "Intact", "Unknown")
        )

        df = df.drop(columns=[self.sex_col])

        return df
    
@dataclass
class NameFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Feature extractor that converts the high-cardinality text column 'Name'
       into a binary indicator 'has_name'.
    """
    name_col: str = "Name"
    def fit(self, df: pd.DataFrame, y=None) -> "NameFeaturesExtractor":
        """Stateless transformer: nothing to learn during fit."""
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform 'Name' into 'has_name', then drop the original column."""
        if self.name_col not in df.columns:
            return df

        clean_name = (
            df[self.name_col].fillna("").astype(str).str.strip()
        )

        df["has_name"] = (
            (clean_name.str.len() > 0)
            & (clean_name.str.lower() != "unknown")
        ).astype(int)

        df = df.drop(columns=[self.name_col])

        return df