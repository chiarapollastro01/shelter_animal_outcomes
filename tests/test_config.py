"""Unit tests for the project configuration module (src/config.py).

"""

from pathlib import Path
import pytest
import yaml
from sklearn.metrics import get_scorer
from src.pipeline import available_models

from src import config

# Constants are addressed by name rather than by value: getattr resolves at
# call time, so a renamed constant fails the single test that reads it instead
# of aborting the collection of the whole module.
RAW_SCHEMA_NAMES = (
    "ID_COL",
    "NAME_COL",
    "DATETIME_COL",
    "TARGET_COL",
    "SUBTARGET_COL",
    "SPECIES_COL",
    "SEX_COL",
    "AGE_COL",
    "BREED_COL",
    "COLOR_COL",
)

ENGINEERED_SCHEMA_NAMES = (
    "HOUR_SIN_COL",
    "HOUR_COS_COL",
    "WDAY_SIN_COL",
    "WDAY_COS_COL",
    "DOY_SIN_COL",
    "DOY_COS_COL",
    "IS_WEEKEND_COL",
    "LOG_AGE_COL",
    "HAS_NAME_COL",
    "IS_MIX_COL",
    "REPRODUCTIVE_STATUS_COL",
)

RAW_GROUP_NAMES = (
    "NON_FEATURE_COLS",
    "CATEGORICAL_COLS",
    "FILL_TARGETS",
    "COLUMNS_TO_REMOVE",
    "ROW_REQUIRED_COLS",
)

ALL_GROUP_NAMES = RAW_GROUP_NAMES + ("NUM_SCALE_COLS", "CAT_ENCODE_COLS", "SPECIES")

CSV_FILE_NAMES = (
    "TRAIN_FEATURES_FILE",
    "TRAIN_TARGET_FILE",
    "TEST_FEATURES_FILE",
    "TEST_TARGET_FILE",
)

REQUIRED_PARAM_KEYS = (
    "test_size",
    "holdout_size",
    "cv_n_splits",
    "scoring",
    "refit",
    "search_spaces",
)

PROPORTION_PARAM_KEYS = ("test_size", "holdout_size")


@pytest.fixture
def params_file(tmp_path: Path) -> Path:
    """A minimal YAML parameter file mixing scalars, sequences and nesting."""
    path = tmp_path / "params.yaml"
    path.write_text(
        "test_size: 0.2\n"
        'scoring: ["f1_macro", "accuracy"]\n'
        "search_spaces:\n"
        "  knn:\n"
        "    clf__n_neighbors: [3, 5]\n",
        encoding="utf-8",
    )
    return path


# =====================================================================
#                        RAW COLUMN SCHEMA
# =====================================================================


class TestRawSchema:
    """Testing constants naming the columns of the raw Kaggle file."""

    @pytest.mark.parametrize("constant_name", RAW_SCHEMA_NAMES)
    def test_raw_column_is_a_nonempty_string(self, constant_name: str):
        """Every raw column constant is a usable pandas column key.

        GIVEN: the name of a raw schema constant declared in config
        WHEN: its value is looked up
        THEN: the value is a non-empty string
        """
        value = getattr(config, constant_name)

        assert isinstance(value, str) and value

    def test_raw_column_names_are_unique(self):
        """No two raw constants name the same column.

        GIVEN: every raw schema constant
        WHEN: their values are collected into a set
        THEN: the set is as large as the tuple, so no constant aliases another
        """
        values = tuple(getattr(config, name) for name in RAW_SCHEMA_NAMES)

        assert len(set(values)) == len(values)


# =====================================================================
#                     ENGINEERED COLUMN SCHEMA
# =====================================================================


class TestEngineeredSchema:
    """Testing constants naming the columns created by the feature engineering."""

    @pytest.mark.parametrize("constant_name", ENGINEERED_SCHEMA_NAMES)
    def test_engineered_column_is_a_nonempty_string(self, constant_name: str):
        """Every engineered column constant is a usable pandas column key.

        GIVEN: the name of an engineered schema constant declared in config
        WHEN: its value is looked up
        THEN: the value is a non-empty string
        """
        value = getattr(config, constant_name)

        assert isinstance(value, str) and value

    def test_engineered_column_names_are_unique(self):
        """No two engineered constants name the same column.

        GIVEN: every engineered schema constant
        WHEN: their values are collected into a set
        THEN: the set is as large as the tuple
        """
        values = tuple(getattr(config, name) for name in ENGINEERED_SCHEMA_NAMES)

        assert len(set(values)) == len(values)

    def test_engineered_names_never_shadow_raw_names(self):
        """A derived column never reuses the name of a raw one.

        GIVEN: the raw and the engineered schema constants
        WHEN: the two sets of values are intersected
        THEN: the intersection is empty, since a transformer assigning a
              derived column would otherwise overwrite raw data in place
        """
        raw = {getattr(config, name) for name in RAW_SCHEMA_NAMES}
        engineered = {getattr(config, name) for name in ENGINEERED_SCHEMA_NAMES}

        assert not raw & engineered


# =====================================================================
#                          FEATURE GROUPS
# =====================================================================


class TestFeatureGroups:
    """Testing tuples grouping columns by the role they play in the pipeline."""

    @pytest.mark.parametrize("group_name", ALL_GROUP_NAMES)
    def test_group_lists_no_entry_twice(self, group_name: str):
        """No group repeats an entry.

        GIVEN: the name of a group constant declared in config
        WHEN: its entries are collected into a set
        THEN: the set is as large as the tuple
        """
        group = getattr(config, group_name)

        assert len(set(group)) == len(group)

    @pytest.mark.parametrize("group_name", RAW_GROUP_NAMES)
    def test_raw_group_references_declared_columns_only(self, group_name: str):
        """Raw-facing groups only name columns the schema declares.

        GIVEN: the name of a group that addresses raw dataset columns
        WHEN: its entries are compared with the declared raw schema
        THEN: every entry belongs to the schema, so a typo or a half-finished
              rename fails here instead of dropping a column silently
        """
        group = getattr(config, group_name)
        raw = {getattr(config, name) for name in RAW_SCHEMA_NAMES}

        assert set(group) <= raw


    def test_scaled_and_encoded_columns_are_disjoint(self):
        """No column is both scaled and encoded.

        GIVEN: NUM_SCALE_COLS and CAT_ENCODE_COLS
        WHEN: intersected
        THEN: the intersection is empty, since a column listed in both
              ColumnTransformer branches would be duplicated in the matrix
        """
        assert not set(config.NUM_SCALE_COLS) & set(config.CAT_ENCODE_COLS)

    def test_binned_categoricals_all_reach_the_encoder(self):
        """Every rare-binned categorical column is also encoded.

        GIVEN: CATEGORICAL_COLS, whose rare levels are collapsed upstream
        WHEN: compared with CAT_ENCODE_COLS
        THEN: each binned column is also listed for encoding, so the binning
              is not wasted on a column the encoder never sees
        """
        assert set(config.CATEGORICAL_COLS) <= set(config.CAT_ENCODE_COLS)

    def test_scaled_columns_are_all_engineered(self):
        """Everything scaled is a column the pipeline itself creates.

        GIVEN: NUM_SCALE_COLS
        WHEN: compared with the engineered schema constants
        THEN: every scaled column is an engineered one, since no raw column
              reaches the scaler untransformed
        """
        engineered = {getattr(config, name) for name in ENGINEERED_SCHEMA_NAMES}

        assert set(config.NUM_SCALE_COLS) <= engineered


# =====================================================================
#                       EXECUTION SETTINGS
# =====================================================================

class TestExecutionSettings:
    """Testing the two constants fixed for the whole project rather than per run."""

    def test_random_state_is_a_non_negative_int(self):
        """The seed is usable by every sklearn estimator.

        GIVEN: the RANDOM_STATE constant
        WHEN: its value is inspected
        THEN: it is a non-negative integer, the range sklearn accepts
        """
        assert isinstance(config.RANDOM_STATE, int)
        assert config.RANDOM_STATE >= 0

    def test_species_roster_is_not_empty(self):
        """The tournament runs on at least one species.

        GIVEN: the SPECIES tuple
        WHEN: its length is inspected
        THEN: it holds at least one entry, since an empty roster would let
              train complete without producing a single model
        """
        assert config.SPECIES

    def test_every_species_is_a_nonempty_string(self):
        """Every species is comparable with the species column.

        GIVEN: the SPECIES tuple
        WHEN: its entries are inspected
        THEN: all of them are non-empty strings
        """
        assert all(isinstance(name, str) and name for name in config.SPECIES)

# =====================================================================
#                             PATHS
# =====================================================================

class TestPaths:
    """Testing filesystem locations derived from the project root."""

    def test_project_root_is_the_directory_above_the_package(self):
        """The root sits one level above the package, not inside it.

        GIVEN: the PROJECT_ROOT constant and the config module's own file
        WHEN: the module's parent directory is compared with the root
        THEN: the root is its parent
        """
        package_dir = Path(config.__file__).resolve().parent

        assert package_dir.parent == config.PROJECT_ROOT

# =====================================================================
#                       FILE NAMING CONVENTIONS
# =====================================================================

class TestFileNames:
    """Testing the names of the artefacts exchanged between the pipeline steps."""

    def test_split_file_names_are_distinct(self):
        """The four split artefacts never overwrite one another.

        GIVEN: the four split-file constants
        WHEN: their values are collected into a set
        THEN: the set is as large as the tuple, so writing all four leaves
              four files in the output directory
        """
        values = tuple(getattr(config, name) for name in CSV_FILE_NAMES)

        assert len(set(values)) == len(values)

    def test_model_template_separates_the_species(self):
        """Two species never map onto the same model file.

        GIVEN: the SPECIES tuple and the model template
        WHEN: the template is formatted once per species
        THEN: the filenames are all distinct, so no tournament overwrites the
              winner of another, which a template missing its {species} field
              would silently cause
        """
        filenames = {
            config.MODEL_FILE_TEMPLATE.format(species=species.lower())
            for species in config.SPECIES
        }

        assert len(filenames) == len(config.SPECIES)

# =====================================================================
#                           LOAD_PARAMS
# =====================================================================

class TestLoadParams:
    """Testing the reading of the run parameters out of a YAML file."""

    def test_document_is_deserialised_with_its_structure(self, params_file: Path):
        """Scalars, sequences and nesting all survive the round trip.

        GIVEN: a YAML file mixing a float, a sequence and a nested mapping
        WHEN: load_params is called on its path
        THEN: each entry comes back with its Python counterpart, reachable
              through the same keys the file declares
        """
        params = config.load_params(params_file)

        assert params["test_size"] == pytest.approx(0.2)
        assert params["scoring"] == ["f1_macro", "accuracy"]
        assert params["search_spaces"]["knn"]["clf__n_neighbors"] == [3, 5]

    def test_empty_file_yields_none(self, tmp_path: Path):
        """An empty file produces None rather than an empty mapping.

        GIVEN: a YAML file with no content at all
        WHEN: load_params is called on its path
        THEN: None comes back, which is what safe_load returns and what a
              caller subscripting the result has to be ready for
        """
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")

        assert config.load_params(path) is None

    def test_comment_only_file_yields_none(self, tmp_path: Path):
        """A file holding only comments produces None as well.

        GIVEN: a YAML file whose every line is a comment
        WHEN: load_params is called on its path
        THEN: None comes back, since comments carry no document
        """
        path = tmp_path / "comments.yaml"
        path.write_text("# nothing declared here\n", encoding="utf-8")

        assert config.load_params(path) is None

    def test_missing_file_raises(self, tmp_path: Path):
        """Pointing at a file that does not exist fails immediately.

        GIVEN: a path with no file behind it
        WHEN: load_params is called on it
        THEN: a FileNotFoundError is raised
        """
        with pytest.raises(FileNotFoundError):
            config.load_params(tmp_path / "absent.yaml")

    def test_malformed_document_raises(self, tmp_path: Path):
        """Unparsable YAML fails at load time, not later.

        GIVEN: a file holding an unterminated YAML sequence
        WHEN: load_params is called on its path
        THEN: a YAMLError is raised
        """
        path = tmp_path / "broken.yaml"
        path.write_text("scoring: [f1_macro,\n", encoding="utf-8")

        with pytest.raises(yaml.YAMLError):
            config.load_params(path)


# =====================================================================
#                     THE SHIPPED CONFIG.YAML
# =====================================================================


class TestShippedParameters:
    """Testing invariants of the config.yaml that travels with the repository.
    """

    @pytest.fixture
    def shipped_params(self) -> dict:
        """The run parameters as declared by the repository's config.yaml."""
        return config.load_params()

    @pytest.mark.parametrize("key", REQUIRED_PARAM_KEYS)
    def test_required_key_is_declared(self, shipped_params: dict, key: str):
        """Every key the pipeline reads is present in the file.

        GIVEN: one of the keys prepare_data or train looks up
        WHEN: the shipped parameters are inspected
        THEN: the key is there, so no step fails with a KeyError mid-run
        """
        assert key in shipped_params

    @pytest.mark.parametrize("key", PROPORTION_PARAM_KEYS)
    def test_split_proportion_is_strictly_between_zero_and_one(
        self, shipped_params: dict, key: str
    ):
        """Both split proportions leave data on either side of the split.

        GIVEN: one of the two split-proportion keys
        WHEN: its value is inspected
        THEN: it lies strictly between 0 and 1, so neither side is empty
        """
        assert 0.0 < shipped_params[key] < 1.0

    def test_fold_count_permits_cross_validation(self, shipped_params: dict):
        """The declared fold count is one sklearn accepts.

        GIVEN: the cv_n_splits key
        WHEN: its value is inspected
        THEN: it is an integer of at least 2, sklearn's minimum
        """
        assert isinstance(shipped_params["cv_n_splits"], int)
        assert shipped_params["cv_n_splits"] >= 2

    def test_every_declared_metric_resolves_to_a_scorer(self, shipped_params: dict):
        """Every scoring entry names a metric sklearn knows.

        GIVEN: the scoring list declared in the shipped parameters
        WHEN: each entry is resolved through sklearn's scorer registry
        THEN: a scorer comes back for all of them, so a typo like 'f1-macro'
              fails here rather than deep inside GridSearchCV
        """
        assert all(get_scorer(metric) for metric in shipped_params["scoring"])

    def test_refit_metric_is_one_of_the_scored_ones(self, shipped_params: dict):
        """The winner is selected on a metric that is actually computed.

        GIVEN: the refit key and the scoring list
        WHEN: the former is looked for among the latter
        THEN: it is present, which is what GridSearchCV requires of refit
        """
        assert shipped_params["refit"] in shipped_params["scoring"]

    def test_search_spaces_declare_a_shared_grid(self, shipped_params: dict):
        """The search spaces carry the entry merged into every family.

        GIVEN: the search_spaces mapping
        WHEN: the shared entry is looked up
        THEN: it is present, since merge_search_grids reads it unguarded
        """
        assert "common" in shipped_params["search_spaces"]

    def test_search_spaces_declare_at_least_one_family(self, shipped_params: dict):
        """The search spaces describe at least one classifier family.

        GIVEN: the search_spaces mapping
        WHEN: the shared entry is set aside
        THEN: at least one family remains, otherwise the tournament would
              run no grid search at all
        """
        families = set(shipped_params["search_spaces"]) - {"common"}

        assert families

    def test_declared_families_are_registered_classifiers(self, shipped_params: dict):
        """Every search space names a family the pipeline can build.

        GIVEN: the search_spaces mapping with the shared entry set aside
        WHEN: its keys are compared with the registered classifier families
        THEN: all of them are registered, so a misspelt or misindented entry
              fails here instead of aborting the tournament halfway
        """
        families = set(shipped_params["search_spaces"]) - {"common"}

        assert families <= set(available_models())