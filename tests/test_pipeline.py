"""
Test suite for the pipeline module.
"""
import pandas as pd
import numpy as np 
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer

from src import config
from src.pipeline import (
    available_models,
    build_preprocess_transformer,
    get_model_pipeline,
)

EXPECTED_STEPS = [
    "cleaner",
    "temporal",
    "categorical_eng",
    "sex_eng",
    "name_eng",
    "onehot_and_scale",
    "smote",
    "clf",
]

@pytest.fixture
def raw_mock_shelter_data() -> tuple[pd.DataFrame, pd.Series]:
    """Raw mock dataset (no AnimalType) matching the schema the pipeline receives in production.
    """
    n = 40
    # There's no AnimalType column because the splitting happens before the application of the pipeline
    # Each class has at least 10 elements to stay above the the default SMOTE
    # k_neighbors (5)
    X = pd.DataFrame(
        {
            config.ID_COL: [f"A{i}" for i in range(n)],
            config.NAME_COL: ["Bella", "Max", None, "Luna"] * 10,
            config.DATETIME_COL: pd.date_range("2026-01-01", periods=n, freq="h").astype(str),
            config.SEX_COL: [
                "Neutered Male",
                "Spayed Female",
                "Intact Male",
                "Unknown",
            ]
            * 10,
            config.AGE_COL: ["1 year", "2 years", "3 weeks", "5 months"] * 10,
            config.BREED_COL: ["Labrador Mix", "Siamese", "Beagle", "Persian"] * 10,
            config.COLOR_COL: ["Black/White", "Brown Tabby", "White", "Red"] * 10,
        }
    )
    y = pd.Series(["Adoption", "Adoption", "Transfer", "Euthanasia"] * 10, name=config.TARGET_COL)
    return X, y


def test_available_models_returns_registered_families():
    """
    GIVEN: the pipeline module
    WHEN: available_models is called
    THEN: it returns a tuple with exactly the registered classifier keys
    """
    models = available_models()
    
    assert isinstance(models, tuple)
    assert set(models) == {"knn", "logistic_regression", "random_forest"}


def test_get_model_pipeline_invalid_type_raises_value_error():
    """
    GIVEN: an unsupported model type identifier
    WHEN: get_model_pipeline is executed
    THEN: a ValueError is raised, chained to the original KeyError,
          and its message lists the available families
    """
    with pytest.raises(ValueError, match="Unknown model type") as exc_info:
        get_model_pipeline(model_type="xgboost_unregistered")
        
    assert isinstance(exc_info.value.__cause__, KeyError)
    assert "knn" in str(exc_info.value)

def test_get_model_pipeline_accepts_valid_config():
    """Verify that the configuration guard stays silent with a consistent config.

    GIVEN: the real project config, where COLUMNS_TO_REMOVE and ESSENTIAL_COLS
           are disjoint
    WHEN: get_model_pipeline is executed
    THEN: no configuration error is raised and the pipeline is built normally
    """
    pipeline = get_model_pipeline(model_type="knn")

    assert pipeline is not None


def test_get_model_pipeline_raises_on_config_overlap(monkeypatch):
    """Verify fail-fast behavior when COLUMNS_TO_REMOVE overlaps ESSENTIAL_COLS.

    GIVEN: a corrupted config where COLUMNS_TO_REMOVE contains an essential
           pipeline column (patched via monkeypatch, auto-restored after the test)
    WHEN: get_model_pipeline is executed
    THEN: a ValueError is raised flagging the critical configuration error
    """
    monkeypatch.setattr(
        config, "COLUMNS_TO_REMOVE", (config.ID_COL, config.DATETIME_COL)
    )

    with pytest.raises(ValueError, match="CRITICAL CONFIGURATION ERROR"):
        get_model_pipeline(model_type="knn")


def test_get_model_pipeline_config_error_names_offending_columns(monkeypatch):
    """Verify that the configuration error message identifies every offending column.

    GIVEN: a corrupted config where two essential columns appear in COLUMNS_TO_REMOVE
    WHEN: get_model_pipeline is executed
    THEN: the raised message names both overlapping columns, so the fix is
          immediate without debugging
    """
    monkeypatch.setattr(
        config, "COLUMNS_TO_REMOVE", (config.DATETIME_COL, config.SEX_COL)
    )

    with pytest.raises(ValueError) as exc_info:
        get_model_pipeline(model_type="knn")

    assert config.DATETIME_COL in str(exc_info.value)
    assert config.SEX_COL in str(exc_info.value)

def test_get_model_pipeline_handles_empty_config_tuples(monkeypatch):
    """Verify that pipeline creation succeeds when configuration tuples are empty.

    GIVEN: a configuration where COLUMNS_TO_REMOVE is an empty tuple
    WHEN: get_model_pipeline is executed
    THEN: set intersection is empty, no ValueError is raised, and pipeline is built normally
    """
    monkeypatch.setattr(config, "COLUMNS_TO_REMOVE", ())

    pipeline = get_model_pipeline(model_type="knn")

    assert pipeline is not None


def test_build_preprocess_transformer_structure():
    """
    GIVEN: the build_preprocess_transformer
    WHEN: executed
    THEN: it returns a ColumnTransformer with 'onehot' and 'scale_num'
          steps and passthrough for the already-binary columns
    """
    transformer = build_preprocess_transformer()

    assert isinstance(transformer, ColumnTransformer)
    assert transformer.remainder == "passthrough"
    names = [name for name, _, _ in transformer.transformers]
    assert names == ["onehot", "scale_num"]


@pytest.mark.parametrize("model_type", available_models())
def test_get_model_pipeline_returns_valid_imb_pipeline(model_type: str):
    """
    GIVEN: any supported model family
    WHEN: get_model_pipeline is invoked
    THEN: it returns an ImbPipeline with the expected ordered steps
    """
    pipeline = get_model_pipeline(model_type=model_type)

    assert isinstance(pipeline, ImbPipeline)
    assert [name for name, _ in pipeline.steps] == EXPECTED_STEPS


@pytest.mark.parametrize("model_type", available_models())
def test_pipeline_end_to_end_fit_and_predict(raw_mock_shelter_data, model_type: str):
    """
    GIVEN: raw mock features and an imbalanced target
    WHEN: fit and predict run on the full pipeline
    THEN: training completes and predictions match the input row count
    """
    X_raw, y_raw = raw_mock_shelter_data
    pipeline = get_model_pipeline(model_type=model_type)

    pipeline.fit(X_raw, y_raw)
    predictions = pipeline.predict(X_raw)

    assert len(predictions) == len(X_raw)
    assert hasattr(pipeline.named_steps["clf"], "classes_")

@pytest.mark.parametrize("model_type", available_models())
def test_pipeline_predict_proba_shape_and_sum(
    raw_mock_shelter_data, model_type: str
):
    """GIVEN: raw mock features and a fitted pipeline for any supported model family.

    WHEN: predict_proba is invoked on the pipeline
    THEN: probability matrix matches input row count, column count matches target classes,
          and row-wise probabilities sum to 1.0
    """
    X_raw, y_raw = raw_mock_shelter_data
    pipeline = get_model_pipeline(model_type=model_type)

    pipeline.fit(X_raw, y_raw)
    probs = pipeline.predict_proba(X_raw)

    assert probs.shape == (len(X_raw), len(np.unique(y_raw)))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_pipeline_predict_handles_unseen_categories(raw_mock_shelter_data):
    """
    GIVEN: a fitted pipeline and a sample with categories unseen in training
    WHEN: predict is invoked
    THEN: OneHotEncoder ignores the unknown levels without raising
    """
    X_train, y_train = raw_mock_shelter_data
    pipeline = get_model_pipeline(model_type="logistic_regression")
    pipeline.fit(X_train, y_train)
    
    X_test_unseen = X_train.iloc[:1].copy()
    X_test_unseen["Breed"] = "Dragon Mix"
    X_test_unseen["Color"] = "Sparkly Golden"
    
    predictions = pipeline.predict(X_test_unseen)
    assert len(predictions) == 1

