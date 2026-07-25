import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.eda import (
    _save_single_figure,
    add_eda_features,
    compute_age_percentiles,
    compute_missing_values,
    compute_outcome_crosstab,
    main,
    parse_args,
    plot_age_distribution,
    plot_missing_values,
    plot_outcome_by_feature,
    plot_target_distribution,
    plot_temporal_outcomes,
    plot_top_categories,
    save_figures,
    split_by_species,
)


@pytest.fixture
def raw_X() -> pd.DataFrame:
    """Raw frame mirroring the original dataset schema (40 rows)."""
    n = 40
    return pd.DataFrame({
        "AnimalID": [f"A{i}" for i in range(n)],
        "Name": ["Bella", None, "Max", "Luna"] * (n // 4),
        "DateTime": pd.date_range("2026-01-01", periods=n, freq="6h").astype(str),
        "OutcomeType": ["Adoption", "Transfer", "Adoption", "Euthanasia"] * (n // 4),
        "OutcomeSubtype": ["Partner", None] * (n // 2),
        "AnimalType": ["Dog", "Cat"] * (n // 2),
        "SexuponOutcome": ["Neutered Male", "Intact Female"] * (n // 2),
        "AgeuponOutcome": ["1 year", "2 months", "3 weeks", None] * (n // 4),
        "Breed": ["Labrador Mix", "Siamese", "Beagle", "Persian"] * (n // 4),
        "Color": ["Black/White", "Orange", "Brown", "Calico"] * (n // 4),
    })


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure after each test to avoid memory buildup."""
    yield
    plt.close("all")


# =====================================================================
#                     DATA PREPARATION TESTS
# =====================================================================

def test_compute_missing_values_detects_and_sorts(raw_X):
    """Verify that columns with missing values are correctly identified and sorted.

    GIVEN: a raw DataFrame containing columns with missing values
    WHEN: compute_missing_values is executed
    THEN: only columns with missing values are returned, sorted descending by count
    """
    missing = compute_missing_values(raw_X)

    assert set(missing.index) == {"Name", "OutcomeSubtype", "AgeuponOutcome"}
    assert list(missing.values) == sorted(missing.values, reverse=True)


def test_compute_missing_values_empty_when_complete(raw_X):
    """Verify that an empty Series is returned when no missing values exist.

    GIVEN: a complete DataFrame without missing values
    WHEN: compute_missing_values is executed
    THEN: an empty pandas Series is returned
    """
    assert compute_missing_values(raw_X.dropna(axis=1)).empty


def test_add_eda_features_creates_columns(raw_X):
    """Verify that add_eda_features correctly derives all temporal and age columns.

    GIVEN: a raw input DataFrame
    WHEN: add_eda_features is executed
    THEN: age_in_days, Month, Hour, and ordered Weekday_Name columns are created
    """
    X_eda = add_eda_features(raw_X)

    assert {"age_in_days", "Month", "Hour", "Weekday_Name"} <= set(
        X_eda.columns
    )
    assert X_eda["Weekday_Name"].cat.ordered
    assert X_eda.loc[0, "age_in_days"] == 365.0


def test_add_eda_features_weekday_categorical_order(raw_X):
    """Verify exact order and categories of derived Weekday_Name feature.

    GIVEN: a raw DataFrame with DateTime entries
    WHEN: add_eda_features is executed
    THEN: Weekday_Name has exactly 7 ordered categories from 'Mon' to 'Sun'
    """
    X_eda = add_eda_features(raw_X)

    categories = list(X_eda["Weekday_Name"].cat.categories)
    assert categories == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def test_add_eda_features_does_not_mutate_input(raw_X):
    """Verify that add_eda_features preserves input DataFrame purity.

    GIVEN: a raw input DataFrame
    WHEN: add_eda_features is executed
    THEN: the input DataFrame remains completely unchanged
    """
    original = raw_X.copy(deep=True)

    add_eda_features(raw_X)

    pd.testing.assert_frame_equal(raw_X, original)


def test_split_by_species_partitions_correctly(raw_X):
    """Verify that split_by_species partitions the dataset by species without data loss.

    GIVEN: a DataFrame containing both Dog and Cat entries
    WHEN: split_by_species is executed
    THEN: separate sub-frames are returned per species with no rows lost
    """
    by_species = split_by_species(raw_X)

    assert set(by_species) == {"Dog", "Cat"}
    assert (by_species["Dog"]["AnimalType"] == "Dog").all()
    assert (by_species["Cat"]["AnimalType"] == "Cat").all()
    assert len(by_species["Dog"]) + len(by_species["Cat"]) == len(raw_X)


def test_split_by_species_ignores_unknown_species(raw_X):
    """Verify that split_by_species filters out animal types not in SPECIES_COLORS.

    GIVEN: a DataFrame containing standard species ('Dog', 'Cat') and an unknown species ('Bird')
    WHEN: split_by_species is executed
    THEN: 'Bird' is ignored and only 'Dog' and 'Cat' are returned in the dictionary
    """
    X_with_unknown = raw_X.copy()
    X_with_unknown.loc[0, "AnimalType"] = "Bird"

    by_species = split_by_species(X_with_unknown)

    assert "Bird" not in by_species
    assert set(by_species.keys()) == {"Dog", "Cat"}


def test_compute_outcome_crosstab_rows_sum_to_one(raw_X):
    """Verify that cross-tabulation row proportions sum up to 1.0.

    GIVEN: a DataFrame with AnimalType and OutcomeType columns
    WHEN: compute_outcome_crosstab is executed with default index normalization
    THEN: every row of proportions sums exactly to 1.0
    """
    ct = compute_outcome_crosstab(raw_X, "AnimalType")

    assert np.allclose(ct.sum(axis=1), 1.0)


def test_compute_outcome_crosstab_normalization_modes(raw_X):
    """Verify compute_outcome_crosstab supports column, total, and raw count normalizations.

    GIVEN: a DataFrame with AnimalType and OutcomeType columns
    WHEN: compute_outcome_crosstab is executed with 'columns', 'all', and None normalization
    THEN: columns sum to 1.0, total matrix sums to 1.0, and raw integer counts are returned respectively
    """
    ct_cols = compute_outcome_crosstab(raw_X, "AnimalType", normalize="columns")
    ct_all = compute_outcome_crosstab(raw_X, "AnimalType", normalize="all")
    ct_raw = compute_outcome_crosstab(raw_X, "AnimalType", normalize=None)

    assert np.allclose(ct_cols.sum(axis=0), 1.0)
    assert np.isclose(ct_all.values.sum(), 1.0)
    assert ct_raw.values.sum() == len(raw_X.dropna(subset=["AnimalType", "OutcomeType"]))


def test_compute_age_percentiles(raw_X):
    """Verify reference percentiles for age in days are monotonic and exclude NaNs.

    GIVEN: a DataFrame enriched with age_in_days containing some NaN values
    WHEN: compute_age_percentiles is executed
    THEN: five non-null, strictly monotonic reference percentiles are returned
    """
    X_eda = add_eda_features(raw_X)

    percentiles = compute_age_percentiles(X_eda)

    assert len(percentiles) == 5
    assert not percentiles.isnull().any()
    assert percentiles.is_monotonic_increasing


def test_compute_age_percentiles_handles_all_nans():
    """Verify compute_age_percentiles returns empty result gracefully when age is all NaN.

    GIVEN: a DataFrame where age_in_days contains exclusively NaN values
    WHEN: compute_age_percentiles is executed
    THEN: an empty pandas Series is returned without raising exceptions
    """
    X_nans = pd.DataFrame({"age_in_days": [np.nan, np.nan]})

    percentiles = compute_age_percentiles(X_nans)

    assert percentiles.dropna().empty


# =====================================================================
#                            PLOTTING 
# =====================================================================

def test_plot_target_distribution(raw_X):
    """Verify that the target distribution chart renders one bar per outcome class.

    GIVEN: a raw DataFrame containing outcome target classes
    WHEN: plot_target_distribution is executed
    THEN: a (fig, ax) tuple is returned with one bar per target category
    """
    fig, ax = plot_target_distribution(raw_X)

    assert isinstance(fig, plt.Figure)
    assert len(ax.patches) == raw_X["OutcomeType"].nunique()
    assert "OutcomeType" in ax.get_title()


def test_plot_missing_values_returns_none_when_complete(raw_X):
    """Verify that missing values plot returns None when the DataFrame has no NaNs.

    GIVEN: a complete DataFrame with no missing values
    WHEN: plot_missing_values is executed
    THEN: None is returned instead of a Matplotlib figure
    """
    assert plot_missing_values(raw_X.dropna(axis=1)) is None


def test_plot_missing_values_returns_figure(raw_X):
    """Verify that missing values plot returns a valid figure when NaNs are present.

    GIVEN: a DataFrame containing missing values
    WHEN: plot_missing_values is executed
    THEN: a valid (fig, ax) pair is returned
    """
    result = plot_missing_values(raw_X)

    assert result is not None
    assert isinstance(result[0], plt.Figure)


def test_plot_outcome_by_feature(raw_X):
    """Verify that stacked outcome proportion plot returns a figure with title set.

    GIVEN: a raw DataFrame with a categorical feature and outcome target
    WHEN: plot_outcome_by_feature is executed with a title
    THEN: a valid (fig, ax) pair is returned with the title correctly configured
    """
    fig, ax = plot_outcome_by_feature(
        raw_X, "AnimalType", "Outcome by AnimalType"
    )

    assert isinstance(fig, plt.Figure)
    assert ax.get_title() == "Outcome by AnimalType"


def test_plot_outcome_by_feature_empty_dataframe():
    """Verify plot_outcome_by_feature handles completely empty DataFrames without errors.

    GIVEN: an empty DataFrame (0 rows)
    WHEN: plot_outcome_by_feature is executed
    THEN: a valid Figure is returned with title set and no exception is raised
    """
    empty_df = pd.DataFrame(columns=["SexuponOutcome", "OutcomeType"])

    fig, ax = plot_outcome_by_feature(empty_df, "SexuponOutcome", "Empty Title")

    assert isinstance(fig, plt.Figure)
    assert ax.get_title() == "Empty Title"


def test_plot_outcome_by_sex_per_species(raw_X):
    """Verify outcome proportion plot renders correctly for SexuponOutcome per species.

    GIVEN: a species-filtered DataFrame containing SexuponOutcome
    WHEN: plot_outcome_by_feature is executed for SexuponOutcome
    THEN: a valid Figure and Axes are returned with the correct title
    """
    X_dogs = split_by_species(raw_X)["Dog"]
    title = "Outcome Distribution by Sex upon Outcome (Dogs)"
    fig, ax = plot_outcome_by_feature(X_dogs, "SexuponOutcome", title)

    assert isinstance(fig, plt.Figure)
    assert ax.get_title() == title


def test_plot_age_distribution(raw_X):
    """Verify that age distribution plot generates a panel with boxplot and histogram.

    GIVEN: a species-filtered DataFrame enriched with age_in_days
    WHEN: plot_age_distribution is executed for a given species
    THEN: a valid Figure and an array of two Axes are returned
    """
    X_dogs = split_by_species(add_eda_features(raw_X))["Dog"]

    fig, axes = plot_age_distribution(X_dogs, "Dog")

    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2


def test_plot_top_categories_annotates_unique_count(raw_X):
    """Verify that top-N category bar chart includes unique category count text.

    GIVEN: a species-filtered DataFrame with high-cardinality features
    WHEN: plot_top_categories is executed
    THEN: the rendered plot contains text annotating total unique categories
    """
    X_dogs = split_by_species(raw_X)["Dog"]

    fig, ax = plot_top_categories(X_dogs, "Breed", "Dog", top_n=5)

    assert any("Unique Categories" in t.get_text() for t in ax.texts)


def test_plot_top_categories_respects_top_n(raw_X):
    """Verify that plot_top_categories restricts displayed bars to top_n.

    GIVEN: a species dataset with multiple categories and top_n=2
    WHEN: plot_top_categories is executed
    THEN: exactly top_n bars are rendered in the figure
    """
    X_dogs = split_by_species(raw_X)["Dog"]

    fig, ax = plot_top_categories(X_dogs, "Breed", "Dog", top_n=2)

    assert len(ax.patches) == 2


def test_plot_temporal_outcomes(raw_X):
    """Verify temporal outcome plot renders a three-panel grid across time features.

    GIVEN: a species-filtered DataFrame enriched with temporal features
    WHEN: plot_temporal_outcomes is executed
    THEN: a valid Figure with three Axes (Month, Weekday, Hour) is returned
    """
    X_dogs = split_by_species(add_eda_features(raw_X))["Dog"]

    fig, axes = plot_temporal_outcomes(X_dogs, "Dog")

    assert isinstance(fig, plt.Figure)
    assert len(axes) == 3


# =====================================================================
#                        EDGE CASES & ROBUSTNESS TESTS
# =====================================================================

def test_plot_outcome_by_feature_handles_nans(raw_X):
    """Verify plot_outcome_by_feature handles missing values (NaN) in SexuponOutcome cleanly.

    GIVEN: a DataFrame where SexuponOutcome contains NaN values
    WHEN: plot_outcome_by_feature is executed
    THEN: it renders successfully without raising errors
    """
    X_with_nans = raw_X.copy()
    X_with_nans.loc[0:5, "SexuponOutcome"] = np.nan

    fig, ax = plot_outcome_by_feature(
        X_with_nans, "SexuponOutcome", "Outcome by Sex (with NaNs)"
    )

    assert isinstance(fig, plt.Figure)
    assert ax.get_title() == "Outcome by Sex (with NaNs)"


def test_save_figures_single_species_only(raw_X, tmp_path):
    """Verify save_figures handles datasets containing only one species (e.g. only Dogs).

    GIVEN: a DataFrame containing only 'Dog' in AnimalType
    WHEN: save_figures is executed
    THEN: only dog-related species figures are generated without crashing
    """
    figures_dir = tmp_path / "figures"
    X_dogs_only = raw_X[raw_X["AnimalType"] == "Dog"].copy()

    written = save_figures(X_dogs_only, figures_dir)

    names = {path.stem for path in written}
    assert "outcome_by_sex_dog" in names
    assert "outcome_by_sex_cat" not in names


def test_plot_top_categories_fewer_categories_than_top_n(raw_X):
    """Verify plot_top_categories works when total unique categories < top_n.

    GIVEN: a species frame with only 2 unique breeds and top_n=30 requested
    WHEN: plot_top_categories is executed
    THEN: it renders exactly 2 bars without error
    """
    X_dogs = split_by_species(raw_X)["Dog"]

    fig, ax = plot_top_categories(X_dogs, "Breed", "Dog", top_n=30)

    assert len(ax.patches) == X_dogs["Breed"].nunique()
    assert len(ax.patches) < 30


def test_compute_outcome_crosstab_single_outcome_value(raw_X):
    """Verify compute_outcome_crosstab handles case when target variable has only 1 unique class.

    GIVEN: a DataFrame where all rows have the same OutcomeType ('Adoption')
    WHEN: compute_outcome_crosstab is executed
    THEN: cross-tabulation returns a valid 1-column DataFrame where row sums equal 1.0
    """
    X_single_outcome = raw_X.copy()
    X_single_outcome["OutcomeType"] = "Adoption"

    ct = compute_outcome_crosstab(X_single_outcome, "SexuponOutcome")

    assert ct.shape[1] == 1
    assert "Adoption" in ct.columns
    assert np.allclose(ct.sum(axis=1), 1.0)


# =====================================================================
#                          ORCHESTRATION & CLI
# =====================================================================

def test_save_single_figure_worker(raw_X, tmp_path):
    """Verify worker function _save_single_figure writes a single figure in a thread context.

    GIVEN: a valid Matplotlib Figure and a output directory
    WHEN: _save_single_figure is executed
    THEN: the figure PNG is saved to disk and the Path object is returned
    """
    fig, _ = plot_target_distribution(raw_X)
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_path = _save_single_figure(("test_plot", fig, figures_dir))

    assert saved_path.exists()
    assert saved_path.name == "test_plot.png"
    assert saved_path.stat().st_size > 1000


def test_save_figures_writes_files(raw_X, tmp_path):
    """Verify save_figures renders and writes species-split figures to disk as PNGs.

    GIVEN: a raw DataFrame and a non-existent output directory
    WHEN: save_figures is executed
    THEN: the directory is created and all expected PNG figures are written to disk
    """
    figures_dir = tmp_path / "figures"

    written = save_figures(raw_X, figures_dir)

    assert figures_dir.exists()
    assert len(written) > 0
    assert all(path.suffix == ".png" and path.exists() for path in written)
    names = {path.stem for path in written}
    
    assert "target_distribution" in names
    assert "outcome_by_animal_type" in names
    assert "outcome_by_sex_dog" in names
    assert "outcome_by_sex_cat" in names
    assert "age_distribution_dog" in names
    assert "age_distribution_cat" in names
    
    assert not any("binned" in name for name in names)


def test_save_figures_existing_directory(raw_X, tmp_path):
    """Verify that save_figures handles pre-existing target directories without error.

    GIVEN: a figures directory that already exists on disk
    WHEN: save_figures is executed
    THEN: figures are saved cleanly without raising FileExistsError
    """
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    written = save_figures(raw_X, figures_dir)

    assert len(written) > 0


def test_save_figures_overwrites_existing_files(raw_X, tmp_path):
    """Verify that save_figures updates and overwrites pre-existing figure files.

    GIVEN: a figures directory containing a pre-existing dummy file
    WHEN: save_figures is executed
    THEN: the dummy file is successfully overwritten with a valid, non-empty PNG image
    """
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dummy_path = figures_dir / "target_distribution.png"
    dummy_path.write_text("old dummy content")

    save_figures(raw_X, figures_dir)

    assert dummy_path.exists()
    assert dummy_path.read_text(errors="ignore") != "old dummy content"
    assert dummy_path.stat().st_size > 1000


def test_main_end_to_end(raw_X, tmp_path):
    """Verify full end-to-end execution of the main EDA pipeline script.

    GIVEN: a raw train CSV file on disk and a target figures directory
    WHEN: main is executed with valid paths
    THEN: all figures are generated and saved without raising errors
    """
    csv_path = tmp_path / "train.csv"
    raw_X.to_csv(csv_path, index=False)
    figures_dir = tmp_path / "figures"

    main(csv_path, figures_dir)

    assert len(list(figures_dir.glob("*.png"))) > 0


def test_parse_args_defaults():
    """Verify that CLI argument parser returns correct default paths.

    GIVEN: an empty argument list
    WHEN: parse_args is executed
    THEN: documented default paths for raw CSV and figures directory are assigned
    """
    args = parse_args([])

    assert args.csv_path.name == "train.csv"
    assert args.figures_dir == Path("reports/figures")


def test_parse_args_custom_values():
    """Verify that CLI argument parser correctly overrides default arguments.

    GIVEN: explicit command-line argument strings
    WHEN: parse_args is executed
    THEN: provided paths override default configurations
    """
    args = parse_args(["my.csv", "--figures-dir", "out/figs"])

    assert args.csv_path.name == "my.csv"
    assert str(args.figures_dir) == "out/figs"


# =====================================================================
#                             LOGGING 
# =====================================================================

def test_compute_missing_values_logs_info(raw_X, caplog):
    """Verify that compute_missing_values logs the count of columns with missing values.

    GIVEN: a DataFrame with missing values in 3 columns
    WHEN: compute_missing_values is executed with INFO log level
    THEN: an INFO log record with the missing column count is captured
    """
    with caplog.at_level(logging.INFO):
        compute_missing_values(raw_X)

    assert "Found 3 columns with missing values" in caplog.text


def test_plot_age_distribution_logs_percentiles(raw_X, caplog):
    """Verify that plot_age_distribution logs species reference age percentiles.

    GIVEN: a species-filtered DataFrame enriched with age_in_days
    WHEN: plot_age_distribution is executed for 'Dog' with INFO log level
    THEN: an INFO log record containing the formatted age percentiles is captured
    """
    X_dogs = split_by_species(add_eda_features(raw_X))["Dog"]

    with caplog.at_level(logging.INFO):
        plot_age_distribution(X_dogs, "Dog")

    assert "[Dog] age percentiles:" in caplog.text


def test_save_figures_logs_summary(raw_X, tmp_path, caplog):
    """Verify that save_figures logs a completion summary upon saving figures to disk.

    GIVEN: a raw DataFrame and an output directory
    WHEN: save_figures is executed with INFO log level
    THEN: an INFO log record confirming the number of saved figures is captured
    """
    figures_dir = tmp_path / "figures"

    with caplog.at_level(logging.INFO):
        save_figures(raw_X, figures_dir)

    assert "Saved" in caplog.text
    assert f"figures to {figures_dir}" in caplog.text


def test_main_logs_loading_progress(raw_X, tmp_path, caplog):
    """Verify that main logs dataset loading status and DataFrame dimensions.

    GIVEN: a raw CSV file on disk
    WHEN: main is executed with INFO log level
    THEN: INFO log records for file path loading and row/column counts are captured
    """
    csv_path = tmp_path / "train.csv"
    raw_X.to_csv(csv_path, index=False)
    figures_dir = tmp_path / "figures"

    with caplog.at_level(logging.INFO):
        main(csv_path, figures_dir)

    assert f"Loading raw dataset from {csv_path}" in caplog.text
    assert f"Loaded {len(raw_X)} rows, {len(raw_X.columns)} columns" in caplog.text