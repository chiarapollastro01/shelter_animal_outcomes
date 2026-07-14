import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.preprocessing import TemporalFeaturesExtractor


class AdvancedTemporalFeaturesExtractor(TemporalFeaturesExtractor):
    """Advanced feature extractor for high-level operational features."""

    def __init__(self, datetime_col: str = "DateTime", add_kitten_season: bool = False):
        """Initialize with option to conditionally add the Kitten Season flag."""
        super().__init__(datetime_col)
        self.add_kitten_season = add_kitten_season

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Obtain Hour and Weekday from TemporalFeaturesExtractor and add

        advanced ones.
        """
        df_out = super().transform(df)

        if "Weekday" not in df_out.columns or "Hour" not in df_out.columns:
            return df_out

        df_out["IsWeekend"] = (df_out["Weekday"] >= 5).astype(int)

        bins = [-1, 5, 11, 16, 23]
        labels = ["Night", "Morning", "Afternoon", "Evening"]
        df_out["TimeOfDay"] = pd.cut(
            df_out["Hour"], bins=bins, labels=labels, right=True
        ).astype(str)

 
        if self.add_kitten_season:

            doy = df_out[self.datetime_col].dt.dayofyear
            df_out["IsKittenSeason"] = ((doy >= 91) & (doy <= 304)).astype(int)

        return df_out


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


class RareCategoriesGrouper(BaseEstimator, TransformerMixin):
    """Dynamic categorical grouper that solves high-cardinality issues.

    Identifies frequent categories on training data and groups low-frequency
    ones into 'Other', preventing data leakage during train/test splits.
    """

    def __init__(self, columns: list, threshold: float = 0.015):
        """Initialize with target columns and a relative frequency threshold."""
        self.columns = columns
        self.threshold = threshold
        self.frequent_categories_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Identify frequent categories (above the threshold) from the training
           data.
        """
        n_rows = len(X)
        for col in self.columns:
            if col in X.columns:
                freqs = X[col].value_counts() / n_rows
                self.frequent_categories_[col] = freqs[
                    freqs >= self.threshold
                ].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace rare categories in the specified columns with 'Other'."""
        X_out = X.copy()
        for col in self.columns:
            if col in X_out.columns and col in self.frequent_categories_:
                frequent = self.frequent_categories_[col]
                X_out[col] = np.where(
                    X_out[col].isin(frequent) | X_out[col].isna(),
                    X_out[col],
                    "Other",
                )
        return X_out


class CategoricalFeaturesEngineer(BaseEstimator, TransformerMixin):
    """Feature engineer for high-cardinality columns (Breed, Color).

    Extracts clean primary representations, determines if the animal is a mix,
    and dynamically bins low-frequency categories to prevent overfitting.
    """

    def __init__(self, columns: list = None, threshold: float = 0.015):
        """Initialize with target columns and the grouping threshold."""
        self.columns = columns if columns is not None else ["Breed", "Color"]
        self.threshold = threshold
        self.grouper_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the internal RareCategoriesGrouper on primary representations of
           Breed and Color.
        """
        X_temp = X.copy()
        if "Breed" in X_temp.columns:
            X_temp["Breed"] = extract_primary_breed(X_temp["Breed"])
        if "Color" in X_temp.columns:
            X_temp["Color"] = extract_primary_color(X_temp["Color"])

        self.grouper_ = RareCategoriesGrouper(
            columns=self.columns, threshold=self.threshold
        )
        self.grouper_.fit(X_temp)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform high-cardinality categorical variables, replacing them

        with clean primary values and grouping rare ones.
        """
        X_out = X.copy()

        if "Breed" in X_out.columns:
            is_mix_series = X_out["Breed"].str.contains(
                "Mix", na=False, case=False
            ) | X_out["Breed"].str.contains("/", na=False)
            X_out["is_mix"] = is_mix_series.astype(int)

            X_out["Breed"] = extract_primary_breed(X_out["Breed"])

        if "Color" in X_out.columns:
            X_out["Color"] = extract_primary_color(X_out["Color"])

        if self.grouper_ is not None:
            X_out = self.grouper_.transform(X_out)

        return X_out
    

class SexFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Feature extractor focused solely on the highly predictive reproductive status."""

    def __init__(self, sex_col: str = "SexuponOutcome"):
        self.sex_col = sex_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if self.sex_col not in df_out.columns:
            return df_out

        is_neutered = df_out[self.sex_col].str.contains("Neutered|Spayed", na=False, case=False)
        is_intact = df_out[self.sex_col].str.contains("Intact", na=False, case=False)
        
        df_out["Reproductive_Status"] = np.where(
            is_neutered, "Neutered/Spayed", 
            np.where(is_intact, "Intact", "Unknown")
        )

        df_out = df_out.drop(columns=[self.sex_col])

        return df_out