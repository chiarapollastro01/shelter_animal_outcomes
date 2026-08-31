"""
Test suite for the pipeline module.
"""
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import pandas as pd
import pytest

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
    """Raw mock dataset matching the schema the pipeline receives in production.
    No animal type column: the split by species happens before the pipeline
    runs, and train drops that column right after. Every class holds at least
    ten rows, above the five neighbours SMOTE needs by default.
    """
    n = 40
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

class TestAvailableModels:
    """Testing the listing of the registered classifier families."""

    def test_available_models_returns_registered_families(self):
        """Verify that available_models returns the correct tuple of registered classifier families.

        GIVEN: the pipeline module
        WHEN: available_models is called
        THEN: it returns a tuple with exactly the registered classifier keys
        """
        models = available_models()

        assert isinstance(models, tuple)
        assert set(models) == {"knn", "logistic_regression", "random_forest"}

class TestGetModelPipeline:
    """Testing the assembly of the full imbalanced-learn pipeline for a given model family."""

    def test_get_model_pipeline_invalid_type_raises_value_error(self):
        """Verify that get_model_pipeline raises a ValueError for an unsupported model type.

        GIVEN: an unsupported model type identifier
        WHEN: get_model_pipeline is executed
        THEN: a ValueError is raised, chained to the original KeyError,
              and its message lists the available families
        """
        with pytest.raises(ValueError, match="Unknown model type") as exc_info:
            get_model_pipeline(model_type="xgboost_unregistered")

        assert isinstance(exc_info.value.__cause__, KeyError)
        assert "knn" in str(exc_info.value)

    @pytest.mark.parametrize("model_type", available_models())
    def test_get_model_pipeline_returns_valid_imb_pipeline(self, model_type: str):
        """Verify that get_model_pipeline returns a valid ImbPipeline for each supported model type.

        GIVEN: any supported model family
        WHEN: get_model_pipeline is invoked
        THEN: it returns an ImbPipeline with the expected ordered steps
        """
        pipeline = get_model_pipeline(model_type=model_type)

        assert isinstance(pipeline, ImbPipeline)
        assert [name for name, _ in pipeline.steps] == EXPECTED_STEPS

class TestBuildPreprocessTransformer:
    """Testing the column transformer routing categorical and numerical columns to their branch."""

    def test_build_preprocess_transformer_routes_each_column_group(self):
        """Verify that each branch receives the column group it is meant for.

        GIVEN: the preprocessing transformer
        WHEN: its branches are inspected
        THEN: the encoder gets the categorical group and the scaler the
              numerical one, a swap between the two being silent otherwise
        """
        transformer = build_preprocess_transformer()

        routed = {name: cols for name, _, cols in transformer.transformers}

        assert routed["onehot"] == list(config.CAT_ENCODE_COLS)
        assert routed["scale_num"] == list(config.NUM_SCALE_COLS)

    def test_the_undeclared_columns_pass_through(self):
        """Verify that the columns in neither branch reach the classifier as they are.

        GIVEN: the preprocessing transformer
        WHEN: its remainder policy is inspected
        THEN: it passes them through rather than dropping them, which is how
              the three binary indicators survive: already in [0, 1], they need
              no scaling to sit beside the min-max scaled features
        """
        transformer = build_preprocess_transformer()

        assert transformer.remainder == "passthrough"

class TestPipelineEndToEnd:
    """Testing the full pipeline fitted and used for prediction on mock shelter data."""

    @pytest.mark.parametrize("model_type", available_models())
    def test_pipeline_end_to_end_fit_and_predict(self, raw_mock_shelter_data, model_type: str):
        """Verify that the full pipeline can be fit and used for predictions.

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
        self,raw_mock_shelter_data, model_type: str
    ):
        """Verify that the probability predictions have the correct shape and sum to 1.0.

        GIVEN: raw mock features and a fitted pipeline for any supported model family
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


    def test_pipeline_predict_handles_unseen_categories(self, raw_mock_shelter_data):
        """Verify that the pipeline can handle unseen categories in the test set.

        GIVEN: a fitted pipeline and a sample with categories unseen in training
        WHEN: predict is invoked
        THEN: OneHotEncoder ignores the unknown levels without raising
        """
        X_train, y_train = raw_mock_shelter_data
        pipeline = get_model_pipeline(model_type="logistic_regression")
        pipeline.fit(X_train, y_train)

        X_test_unseen = X_train.iloc[:1].copy()
        X_test_unseen[config.BREED_COL] = "Dragon Mix"
        X_test_unseen[config.COLOR_COL] = "Sparkly Golden"

        predictions = pipeline.predict(X_test_unseen)
        assert len(predictions) == 1

    def test_pipeline_rejects_a_frame_missing_a_required_column(
        self, raw_mock_shelter_data
    ):
        """Verify that a missing source column surfaces from inside the pipeline.

        GIVEN: raw features without the datetime column
        WHEN: fit is executed on the full pipeline
        THEN: a ValueError names the column, the transformers refusing to run
              rather than quietly producing a matrix without those features
        """
        X_raw, y_raw = raw_mock_shelter_data
        pipeline = get_model_pipeline(model_type="knn")

        with pytest.raises(ValueError, match=config.DATETIME_COL):
            pipeline.fit(X_raw.drop(columns=[config.DATETIME_COL]), y_raw)

    def test_fitting_the_pipeline_does_not_mutate_the_input(
        self, raw_mock_shelter_data
    ):
        """Verify that the caller's frame comes back unchanged after fitting.

        GIVEN: raw features and a fitted pipeline
        WHEN: fit and predict have both run
        THEN: the original frame is identical to a copy taken beforehand,
              a single step forgetting to copy being enough to break this
        """
        X_raw, y_raw = raw_mock_shelter_data
        X_before = X_raw.copy()

        get_model_pipeline(model_type="knn").fit(X_raw, y_raw)

        pd.testing.assert_frame_equal(X_raw, X_before)

    def test_pipeline_rejects_a_class_too_small_for_smote(
        self, raw_mock_shelter_data
    ):
        """Verify that a class with too few rows is reported at fit time.

        GIVEN: a target where one class holds fewer rows than the neighbours
               SMOTE interpolates between
        WHEN: fit is executed on the full pipeline
        THEN: a ValueError is raised, since there is nothing to interpolate
              from, which is why the fixture keeps every class above that floor
        """
        X_raw, _ = raw_mock_shelter_data
        y_rare = pd.Series(
            ["Adoption"] * 38 + ["Euthanasia"] * 2, name=config.TARGET_COL
        )
        pipeline = get_model_pipeline(model_type="knn")

        with pytest.raises(ValueError, match="n_neighbors"):
            pipeline.fit(X_raw, y_rare)
