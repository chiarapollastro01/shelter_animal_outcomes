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

from src import config

logger = logging.getLogger(__name__)


_CLASSIFIERS: dict[str, Callable[[], ClassifierMixin]] = {
    "knn": lambda: KNeighborsClassifier(),
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, random_state=config.RANDOM_STATE
    ),
    "random_forest": lambda: RandomForestClassifier(random_state=config.RANDOM_STATE),
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
                list(config.CAT_ENCODE_COLS),
            ),
            ("scale_num", MinMaxScaler(), list(config.NUM_SCALE_COLS)),
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
    overlapping_cols = set(config.COLUMNS_TO_REMOVE) & set(config.ESSENTIAL_COLS)
    if overlapping_cols:
        raise ValueError(
            "CRITICAL CONFIGURATION ERROR: COLUMNS_TO_REMOVE contains essential "
            f"pipeline feature columns: {overlapping_cols}. "
            "Remove these from config.COLUMNS_TO_REMOVE to prevent breaking downstream feature engineering."
        )

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
        ("categorical_eng", CategoricalFeaturesEngineer(max_other_ratio=config.MAX_OTHER_RATIO)),
        ("sex_eng", SexFeaturesExtractor()),
        ("name_eng", NameFeaturesExtractor()),
        ("onehot_and_scale", build_preprocess_transformer()),
        # SMOTE rebalances the classes inside the pipeline. It only runs only on the training folds 
        # (imblearn contract), so the validation folds always keep the true class distribution (no data leakage)
        ("smote", SMOTE(random_state=config.RANDOM_STATE)),
        ("clf", clf),
        ])