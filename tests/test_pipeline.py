"""
Test suite for the pipeline module.

Validates the registry used for classifiers, the column transformer setup,
handling of unknown model types, and end-to-end fit/predict for
every registered classifier.

"""
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer

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
    X = pd.DataFrame({
        "AnimalID": [f"A{i}" for i in range(n)],
        "Name": ["Bella", "Max", None, "Luna"] * 10,
        "DateTime": pd.date_range("2026-01-01", periods=n, freq="h").astype(str),
        "SexuponOutcome": ["Neutered Male", "Spayed Female", "Intact Male", "Unknown"] * 10,
        "AgeuponOutcome": ["1 year", "2 years", "3 weeks", "5 months"] * 10,
        "Breed": ["Labrador Mix", "Siamese", "Beagle", "Persian"] * 10,
        "Color": ["Black/White", "Brown Tabby", "White", "Red"] * 10,
        "OutcomeSubtype": ["Partner", "Foster"] * 20,
    })
    y = pd.Series(["Adoption", "Adoption", "Transfer", "Euthanasia"] * 10)
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