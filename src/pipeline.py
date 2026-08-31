"""Model pipeline for the Shelter Animal Outcomes classifiers.

Builds the full end-to-end pipeline: cleaning -> feature engineering ->
encoding/scaling -> SMOTE -> classifier.

Exported Functions
------------------
available_models() -> tuple[str, ...]
    Names of the classifier families the pipeline can build.

build_preprocess_transformer() -> ColumnTransformer
    The encoding and scaling step, one-hot for the categoricals and min-max
    for the numericals.

get_model_pipeline(model_type) -> ImbPipeline
    The end-to-end pipeline for one classifier family.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import logging

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src import config
from src.feature_engineering import (
    CategoricalFeaturesEngineer,
    NameFeaturesExtractor,
    SexFeaturesExtractor,
    TemporalFeaturesExtractor,
)
from src.preprocessing import DataCleaner

logger = logging.getLogger(__name__)

_CLASSIFIERS: dict[str, Callable[[], ClassifierMixin]] = {
    "knn": KNeighborsClassifier,
    "logistic_regression": partial(
        LogisticRegression, random_state=config.RANDOM_STATE
    ),
    "random_forest": partial(
        RandomForestClassifier, random_state=config.RANDOM_STATE
    ),
}


def available_models() -> tuple[str, ...]:
    """Return the names of the supported classifier families.

    Returns
    -------
    tuple[str, ...]
        Tuple of registered model identifier strings.

    Examples
    --------
    >>> available_models()
    ('knn', 'logistic_regression', 'random_forest')
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
        # Everything not listed in either branch reaches the classifier through the
        # passthrough remainder: the three binary indicators (weekend, mix and name),
        # already in [0, 1] and therefore compatible with the min-max scaled features
        # KNN compares them to.
        remainder="passthrough",
    )


def get_model_pipeline(model_type: str) -> ImbPipeline:
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
        # Order matters at both ends: DataCleaner comes first because it fills
        # Name, Breed and Color with "Unknown", so the transformers below never
        # see a missing value in them; SMOTE comes last before the classifier
        # because it must resample the encoded matrix, not the raw frame.
        ("cleaner", DataCleaner()),
        ("temporal", TemporalFeaturesExtractor()),
        ("categorical_eng", CategoricalFeaturesEngineer()),
        ("sex_eng", SexFeaturesExtractor()),
        ("name_eng", NameFeaturesExtractor()),
        ("onehot_and_scale", build_preprocess_transformer()),
        # SMOTE rebalances the classes inside the pipeline. It only runs on the training folds
        # (imblearn contract), so the validation folds always keep the true class distribution
        # (no data leakage)
        ("smote", SMOTE(random_state=config.RANDOM_STATE)),
        ("clf", clf),
    ])
