"""Unit tests for the EDA module.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import logging

from src.eda import (
    add_eda_features,
    compute_age_percentiles,
    compute_binned_frequencies,
    compute_missing_values,
    compute_outcome_crosstab,
    main,
    parse_args,
    plot_age_distribution,
    plot_binned_frequencies,
    plot_missing_values,
    plot_outcome_by_feature,
    plot_target_distribution,
    plot_temporal_outcomes,
    plot_top_categories,
    save_figures,
    split_by_species,
)

# =====================================================================
#                             FIXTURES
# =====================================================================

@pytest.fixture
def raw_df() -> pd.DataFrame:
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

def test_compute_missing_values_detects_and_sorts(raw_df):
    """Verify that columns with missing values are correctly identified and sorted.

    GIVEN: a frame with missing values in Name/OutcomeSubtype/AgeuponOutcome
    WHEN: compute_missing_values is executed
    THEN: only those columns are returned, sorted descending
    """
    missing = compute_missing_values(raw_df)

    assert set(missing.index) == {"Name", "OutcomeSubtype", "AgeuponOutcome"}
    assert list(missing.values) == sorted(missing.values, reverse=True)


def test_compute_missing_values_empty_when_complete(raw_df):
    """Verify that an empty Series is returned when no missing values exist.

    GIVEN: a complete frame without any missing values
    WHEN: compute_missing_values is executed
    THEN: an empty pandas Series is returned
    """
    assert compute_missing_values(raw_df.dropna(axis=1)).empty


def test_add_eda_features_creates_columns(raw_df):
    """Verify that add_eda_features correctly derives all temporal and age columns.

    GIVEN: a raw input frame
    WHEN: add_eda_features is executed
    THEN: age_in_days, Month, Hour, and ordered Weekday_Name columns are created
    """
    df_eda = add_eda_features(raw_df)

    assert {"age_in_days", "Month", "Hour", "Weekday_Name"} <= set(
        df_eda.columns
    )
    assert df_eda["Weekday_Name"].cat.ordered
    assert df_eda.loc[0, "age_in_days"] == 365.0

def test_add_eda_features_weekday_categorical_order(raw_df):
    """Verify exact order and categories of derived Weekday_Name feature.

    GIVEN: a raw DataFrame with DateTime entries
    WHEN: add_eda_features is executed
    THEN: Weekday_Name has exactly 7 ordered categories from 'Mon' to 'Sun'
    """
    df_eda = add_eda_features(raw_df)

    categories = list(df_eda["Weekday_Name"].cat.categories)
    assert categories == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def test_add_eda_features_does_not_mutate_input(raw_df):
    """Verify that add_eda_features preserves input DataFrame purity.

    GIVEN: a raw input frame
    WHEN: add_eda_features is executed
    THEN: the input frame remains completely unchanged
    """
    original = raw_df.copy(deep=True)

    add_eda_features(raw_df)

    pd.testing.assert_frame_equal(raw_df, original)


def test_split_by_species_partitions_correctly(raw_df):
    """Verify that split_by_species partitions the dataset by species without data loss.

    GIVEN: a frame containing both Dog and Cat entries
    WHEN: split_by_species is executed
    THEN: separate sub-frames are returned per species with no rows lost
    """
    by_species = split_by_species(raw_df)

    assert set(by_species) == {"Dog", "Cat"}
    assert (by_species["Dog"]["AnimalType"] == "Dog").all()
    assert (by_species["Cat"]["AnimalType"] == "Cat").all()
    assert len(by_species["Dog"]) + len(by_species["Cat"]) == len(raw_df)

def test_split_by_species_ignores_unknown_species(raw_df):
    """Verify that split_by_species filters out animal types not in SPECIES_COLORS.

    GIVEN: a DataFrame containing standard species ('Dog', 'Cat') and an unknown species ('Bird')
    WHEN: split_by_species is executed
    THEN: 'Bird' is ignored and only 'Dog' and 'Cat' are returned in the dictionary
    """
    df_with_unknown = raw_df.copy()
    df_with_unknown.loc[0, "AnimalType"] = "Bird"

    by_species = split_by_species(df_with_unknown)

    assert "Bird" not in by_species
    assert set(by_species.keys()) == {"Dog", "Cat"}


def test_compute_outcome_crosstab_rows_sum_to_one(raw_df):
    """Verify that cross-tabulation row proportions sum up to 1.0.

    GIVEN: a frame with AnimalType and OutcomeType columns
    WHEN: compute_outcome_crosstab is executed with default index normalization
    THEN: every row of proportions sums exactly to 1.0
    """
    ct = compute_outcome_crosstab(raw_df, "AnimalType")

    assert np.allclose(ct.sum(axis=1), 1.0)

def test_compute_outcome_crosstab_normalization_modes(raw_df):
    """Verify that compute_outcome_crosstab supports column, total, and raw count normalizations.

    GIVEN: a DataFrame with AnimalType and OutcomeType columns
    WHEN: compute_outcome_crosstab is executed with 'columns', 'all', and None normalization
    THEN: columns sum to 1.0, total matrix sums to 1.0, and raw integer counts are returned respectively
    """
    ct_cols = compute_outcome_crosstab(raw_df, "AnimalType", normalize="columns")
    ct_all = compute_outcome_crosstab(raw_df, "AnimalType", normalize="all")
    ct_raw = compute_outcome_crosstab(raw_df, "AnimalType", normalize=None)

    assert np.allclose(ct_cols.sum(axis=0), 1.0)
    assert np.isclose(ct_all.values.sum(), 1.0)
    assert ct_raw.values.sum() == len(raw_df.dropna(subset=["AnimalType", "OutcomeType"]))


def test_compute_age_percentiles(raw_df):
    """Verify that reference percentiles for age in days are monotonic and exclude NaNs.

    GIVEN: a frame enriched with age_in_days containing some NaN values
    WHEN: compute_age_percentiles is executed
    THEN: five non-null, strictly monotonic reference percentiles are returned
    """
    df_eda = add_eda_features(raw_df)

    percentiles = compute_age_percentiles(df_eda)

    assert len(percentiles) == 5
    assert not percentiles.isnull().any()
    assert percentiles.is_monotonic_increasing

def test_compute_age_percentiles_handles_all_nans():
    """Verify that compute_age_percentiles returns empty result gracefully when age is all NaN.

    GIVEN: a DataFrame where age_in_days contains exclusively NaN values
    WHEN: compute_age_percentiles is executed
    THEN: an empty pandas Series is returned without raising exceptions
    """
    df_nans = pd.DataFrame({"age_in_days": [np.nan, np.nan]})

    percentiles = compute_age_percentiles(df_nans)

    assert percentiles.dropna().empty

def test_compute_binned_frequencies_other_is_last():
    """Verify that binned category frequencies sum to 1.0 with 'Other' listed last.

    GIVEN: a categorical Series containing rare labels
    WHEN: compute_binned_frequencies is executed with a threshold
    THEN: relative frequencies sum to 1.0 and 'Other' is ordered as the final category
    """
    series = pd.Series(["A"] * 8 + ["B"] * 8 + ["C"] * 2 + ["D"] * 2)

    freqs = compute_binned_frequencies(series, max_other_ratio=0.25)

    assert np.isclose(freqs.sum(), 1.0)
    assert freqs.index[-1] == "Other"
    assert "A" in freqs.index and "B" in freqs.index

def test_compute_binned_frequencies_without_other():
    """Verify binned frequencies calculation when no rare categories exceed the threshold.

    GIVEN: a categorical Series where keeping all categories is required to satisfy max_other_ratio
    WHEN: compute_binned_frequencies is executed with max_other_ratio=0.1 (requiring 90% coverage)
    THEN: frequencies sum to 1.0 and 'Other' category is not generated
    """
    series = pd.Series(["A"] * 10 + ["B"] * 10)

    # Con max_other_ratio=0.1 serve trattenere almeno il 90% dei dati.
    # 'A' (50%) non basta, quindi mantiene sia 'A' che 'B' e 'Other' NON viene creato.
    freqs = compute_binned_frequencies(series, max_other_ratio=0.1)

    assert np.isclose(freqs.sum(), 1.0)
    assert "Other" not in freqs.index
    assert set(freqs.index) == {"A", "B"}


# =====================================================================
#                            PLOTTING 
# =====================================================================

def test_plot_target_distribution(raw_df):
    """Verify that the target distribution chart renders one bar per outcome class.

    GIVEN: a raw frame containing outcome target classes
    WHEN: plot_target_distribution is executed
    THEN: a (fig, ax) tuple is returned with one bar per target category
    """
    fig, ax = plot_target_distribution(raw_df)

    assert isinstance(fig, plt.Figure)
    assert len(ax.patches) == raw_df["OutcomeType"].nunique()
    assert "OutcomeType" in ax.get_title()


def test_plot_missing_values_returns_none_when_complete(raw_df):
    """Verify that missing values plot returns None when the DataFrame has no NaNs.

    GIVEN: a complete frame with no missing values
    WHEN: plot_missing_values is executed
    THEN: None is returned instead of a Matplotlib figure
    """
    assert plot_missing_values(raw_df.dropna(axis=1)) is None


def test_plot_missing_values_returns_figure(raw_df):
    """Verify that missing values plot returns a valid figure when NaNs are present.

    GIVEN: a frame containing missing values
    WHEN: plot_missing_values is executed
    THEN: a valid (fig, ax) pair is returned
    """
    result = plot_missing_values(raw_df)

    assert result is not None
    assert isinstance(result[0], plt.Figure)


def test_plot_outcome_by_feature(raw_df):
    """Verify that stacked outcome proportion plot returns a figure with title set.

    GIVEN: a raw frame with a categorical feature and outcome target
    WHEN: plot_outcome_by_feature is executed with a title
    THEN: a valid (fig, ax) pair is returned with the title correctly configured
    """
    fig, ax = plot_outcome_by_feature(
        raw_df, "AnimalType", "Outcome by AnimalType"
    )

    assert isinstance(fig, plt.Figure)
    assert ax.get_title() == "Outcome by AnimalType"


def test_plot_age_distribution(raw_df):
    """Verify that age distribution plot generates a panel with boxplot and histogram.

    GIVEN: a species-filtered frame enriched with age_in_days
    WHEN: plot_age_distribution is executed for a given species
    THEN: a valid Figure and an array of two Axes are returned
    """
    df_dogs = split_by_species(add_eda_features(raw_df))["Dog"]

    fig, axes = plot_age_distribution(df_dogs, "Dog")

    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2


def test_plot_top_categories_annotates_unique_count(raw_df):
    """Verify that top-N category bar chart includes unique category count text.

    GIVEN: a species-filtered frame with high-cardinality features
    WHEN: plot_top_categories is executed
    THEN: the rendered plot contains text annotating total unique categories
    """
    df_dogs = split_by_species(raw_df)["Dog"]

    fig, ax = plot_top_categories(df_dogs, "Breed", "Dog", top_n=5)

    assert any("Unique Categories" in t.get_text() for t in ax.texts)


def test_plot_binned_frequencies():
    """Verify that binned relative frequencies plot renders horizontal bars.

    GIVEN: a prepared Series of normalized category frequencies
    WHEN: plot_binned_frequencies is executed
    THEN: a valid (fig, ax) tuple is returned with the specified title set
    """
    freqs = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "Other"])

    fig, ax = plot_binned_frequencies(freqs, "Binned Distribution")

    assert isinstance(fig, plt.Figure)
    assert ax.get_title() == "Binned Distribution"

def test_plot_top_categories_respects_top_n(raw_df):
    """Verify that plot_top_categories restricts displayed bars to top_n.

    GIVEN: a species dataset with multiple categories and top_n=2
    WHEN: plot_top_categories is executed
    THEN: exactly top_n bars are rendered in the figure
    """
    df_dogs = split_by_species(raw_df)["Dog"]

    fig, ax = plot_top_categories(df_dogs, "Breed", "Dog", top_n=2)

    assert len(ax.patches) == 2

def test_plot_binned_frequencies():
    """Verify that plot_binned_frequencies renders a valid horizontal bar chart with title.

    GIVEN: a prepared Series of relative category frequencies and a custom title
    WHEN: plot_binned_frequencies is executed
    THEN: a valid (fig, ax) tuple is returned with the specified title correctly configured
    """
    freqs = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "Other"])

    fig, ax = plot_binned_frequencies(freqs, "Binned Distribution")

    assert isinstance(fig, plt.Figure)
    assert ax.get_title() == "Binned Distribution"


def test_plot_temporal_outcomes(raw_df):
    """Verify that temporal outcome plot renders a three-panel grid across time features.

    GIVEN: a species-filtered frame enriched with temporal features
    WHEN: plot_temporal_outcomes is executed
    THEN: a valid Figure with three Axes (Month, Weekday, Hour) is returned
    """
    df_dogs = split_by_species(add_eda_features(raw_df))["Dog"]

    fig, axes = plot_temporal_outcomes(df_dogs, "Dog")

    assert isinstance(fig, plt.Figure)
    assert len(axes) == 3


# =====================================================================
#                          ORCHESTRATION & CLI
# =====================================================================


def test_save_figures_writes_files(raw_df, tmp_path):
    """Verify that save_figures renders and writes all EDA figures to disk as PNGs.

    GIVEN: a raw frame and a non-existent output directory
    WHEN: save_figures is executed
    THEN: the directory is created and all expected PNG figures are written to disk
    """
    figures_dir = tmp_path / "figures"

    written = save_figures(raw_df, figures_dir)

    assert figures_dir.exists()
    assert len(written) > 0
    assert all(path.suffix == ".png" and path.exists() for path in written)
    names = {path.stem for path in written}
    assert "target_distribution" in names
    assert "age_distribution_dog" in names
    assert "age_distribution_cat" in names

def test_save_figures_existing_directory(raw_df, tmp_path):
    """Verify that save_figures handles pre-existing target directories without error.

    GIVEN: a figures directory that already exists on disk
    WHEN: save_figures is executed
    THEN: figures are saved cleanly without raising FileExistsError
    """
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    written = save_figures(raw_df, figures_dir)

    assert len(written) > 0

def test_save_figures_overwrites_existing_files(raw_df, tmp_path):
    """Verify that save_figures updates and overwrites pre-existing figure files.

    GIVEN: a figures directory containing a pre-existing dummy file
    WHEN: save_figures is executed
    THEN: the dummy file is successfully overwritten with a valid, non-empty PNG image
    """
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dummy_path = figures_dir / "target_distribution.png"
    dummy_path.write_text("old dummy content")

    save_figures(raw_df, figures_dir)

    assert dummy_path.exists()
    assert dummy_path.read_text(errors="ignore") != "old dummy content"
    assert dummy_path.stat().st_size > 1000  # A real PNG binary is well over 1 KB


def test_main_end_to_end(raw_df, tmp_path):
    """Verify full end-to-end execution of the main EDA pipeline script.

    GIVEN: a raw train CSV file on disk and a target figures directory
    WHEN: main is executed with valid paths
    THEN: all figures are generated and saved without raising errors
    """
    csv_path = tmp_path / "train.csv"
    raw_df.to_csv(csv_path, index=False)
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


def test_compute_missing_values_logs_info(raw_df, caplog):
    """Verify that compute_missing_values logs the count of columns with missing values.

    GIVEN: a frame with missing values in 3 columns
    WHEN: compute_missing_values is executed with INFO log level
    THEN: an INFO log record with the missing column count is captured
    """
    with caplog.at_level(logging.INFO):
        compute_missing_values(raw_df)

    assert "Found 3 columns with missing values" in caplog.text


def test_plot_age_distribution_logs_percentiles(raw_df, caplog):
    """Verify that plot_age_distribution logs species reference age percentiles.

    GIVEN: a species-filtered frame enriched with age_in_days
    WHEN: plot_age_distribution is executed for 'Dog' with INFO log level
    THEN: an INFO log record containing the formatted age percentiles is captured
    """
    df_dogs = split_by_species(add_eda_features(raw_df))["Dog"]

    with caplog.at_level(logging.INFO):
        plot_age_distribution(df_dogs, "Dog")

    assert "[Dog] age percentiles:" in caplog.text


def test_save_figures_logs_summary(raw_df, tmp_path, caplog):
    """Verify that save_figures logs a completion summary upon saving figures to disk.

    GIVEN: a raw frame and an output directory
    WHEN: save_figures is executed with INFO log level
    THEN: an INFO log record confirming the number of saved figures is captured
    """
    figures_dir = tmp_path / "figures"

    with caplog.at_level(logging.INFO):
        save_figures(raw_df, figures_dir)

    assert "Saved" in caplog.text
    assert f"figures to {figures_dir}" in caplog.text


def test_main_logs_loading_progress(raw_df, tmp_path, caplog):
    """Verify that main logs dataset loading status and DataFrame dimensions.

    GIVEN: a raw CSV file on disk
    WHEN: main is executed with INFO log level
    THEN: INFO log records for file path loading and row/column counts are captured
    """
    csv_path = tmp_path / "train.csv"
    raw_df.to_csv(csv_path, index=False)
    figures_dir = tmp_path / "figures"

    with caplog.at_level(logging.INFO):
        main(csv_path, figures_dir)

    assert f"Loading raw dataset from {csv_path}" in caplog.text
    assert f"Loaded {len(raw_df)} rows, {len(raw_df.columns)} columns" in caplog.text