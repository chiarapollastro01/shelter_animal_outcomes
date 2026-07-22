"""Model pipeline for the Shelter Animal Outcomes classifiers.

Builds the full end-to-end pipeline: cleaning -> feature engineering ->
encoding/scaling -> SMOTE -> classifier. 
"""

from __future__ import annotations

import logging
from typing import Callable

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.feature_engineering import (
    CategoricalFeaturesEngineer,
    NameFeaturesExtractor,
    SexFeaturesExtractor,
    TemporalFeaturesExtractor,
)
from src.preprocessing import DataCleaner

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# These names are produced by the feature-engineering step: if an
# extractor renames its output, this is the only place to update.
NUM_SCALE_COLS: tuple[str, ...] = (
    "Hour_sin", "Hour_cos",
    "Wday_sin", "Wday_cos",
    "DoY_sin", "DoY_cos",
    "log_age_in_days",
)
CAT_ENCODE_COLS: tuple[str, ...] = ("Breed", "Color", "Reproductive_Status")
# Binary columns (IsWeekend, is_mix, has_name) are already 0/1: passthrough.

# Registry: adding a model = adding one entry for eventual future modifications.
_CLASSIFIERS: dict[str, Callable[[], ClassifierMixin]] = {
    "knn": lambda: KNeighborsClassifier(),
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ),
    "random_forest": lambda: RandomForestClassifier(random_state=RANDOM_STATE),
}


def available_models() -> tuple[str, ...]:
    """Return the names of the supported classifier families.

    Returns
    -------
    tuple[str, ...]
        Tuple of registered model identifier strings.
    """
    return tuple(_CLASSIFIERS)


def build_preprocess_transformer() -> ColumnTransformer:
    """Build the feature encoding and scaling ColumnTransformer step.

    Applies One-Hot Encoding to categorical variables with unknown category handling,
    and Min-Max scaling to numerical variables to ensure distance compatibility.

    Returns
    -------
    ColumnTransformer
        Configured ColumnTransformer step ready for pipeline inclusion.
    """
    return ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CAT_ENCODE_COLS),
            ),
            ("scale_num", MinMaxScaler(), list(NUM_SCALE_COLS)),
        ],
        remainder="passthrough",
    )


def get_model_pipeline(model_type: str = "knn") -> ImbPipeline:
    """Build the end-to-end pipeline for the requested classifier family.

    Parameters
    ----------
    model_type : str
        One of ``available_models()``.

    Returns
    -------
    ImbPipeline
        Unfitted pipeline: cleaning -> features -> encoding -> SMOTE -> clf.

    Raises
    ------
    ValueError
        If *model_type* is not a registered classifier family.
    """
    try:
        clf = _CLASSIFIERS[model_type]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            f"Available: {available_models()}"
        ) from exc

    logger.info("Building pipeline for model type '%s'", model_type)

    return ImbPipeline([
        ("cleaner", DataCleaner()),
        ("temporal", TemporalFeaturesExtractor()),
        ("categorical_eng", CategoricalFeaturesEngineer(max_other_ratio=0.15)),
        ("sex_eng", SexFeaturesExtractor()),
        ("name_eng", NameFeaturesExtractor()),
        ("onehot_and_scale", build_preprocess_transformer()),
        # SMOTE rebalances the classes inside the pipeline. It only runs only on the training folds 
        # (imblearn contract), so the validation folds always keep the true class distribution (no data leakage)
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", clf),
        ])