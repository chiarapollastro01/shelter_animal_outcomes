"Testing suite for the EDA module"
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import config, eda
from src.eda import (
    add_eda_features,
    build_figures,
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
        config.ID_COL: [f"A{i}" for i in range(n)],
        config.NAME_COL: ["Bella", None, "Max", "Luna"] * (n // 4),
        config.DATETIME_COL: pd.date_range("2026-01-01", periods=n, freq="6h").astype(str),
        config.TARGET_COL: ["Adoption", "Transfer", "Adoption", "Euthanasia"] * (n // 4),
        config.SUBTARGET_COL: ["Partner", None] * (n // 2),
        config.SPECIES_COL: ["Dog", "Cat"] * (n // 2),
        config.SEX_COL: ["Neutered Male", "Intact Female"] * (n // 2),
        config.AGE_COL : ["1 year", "2 months", "3 weeks", None] * (n // 4),
        config.BREED_COL: ["Labrador Mix", "Siamese", "Beagle", "Persian"] * (n // 4),
        config.COLOR_COL: ["Black/White", "Orange", "Brown", "Calico"] * (n // 4),
    })


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure after each test to avoid memory buildup."""
    yield
    plt.close("all")


class TestDataPreparation:

    def test_only_the_columns_with_gaps_are_reported(self, raw_X):
        """Verify that complete columns are left out of the result.

        GIVEN: a raw frame where three columns of ten hold missing values
        WHEN: compute_missing_values is executed
        THEN: those three come back with their exact counts and nothing else
        """
        missing = compute_missing_values(raw_X)

        assert missing.to_dict() == {
            config.SUBTARGET_COL: 20,
            config.NAME_COL: 10,
            config.AGE_COL: 10,
        }

    def test_the_columns_come_back_worst_first(self, raw_X):
        """Verify that the result is ordered by how many values are missing.

        GIVEN: a raw frame whose columns have different numbers of gaps
        WHEN: compute_missing_values is executed
        THEN: the counts descend, so the plot reads worst-first without the
              caller having to sort it again
        """
        missing = compute_missing_values(raw_X)

        assert list(missing.values) == sorted(missing.values, reverse=True)


    def test_compute_missing_values_empty_when_complete(self, raw_X):
        """Verify that an empty Series is returned when no missing values exist.

        GIVEN: a complete DataFrame without missing values
        WHEN: compute_missing_values is executed
        THEN: an empty pandas Series is returned
        """
        assert compute_missing_values(raw_X.dropna(axis=1)).empty


    def test_add_eda_features_creates_columns(self, raw_X):
        """Verify that add_eda_features correctly derives all temporal and age columns.

        GIVEN: a raw input DataFrame
        WHEN: add_eda_features is executed
        THEN: age_in_days, Month, Hour, and ordered Weekday_Name columns are created
        """
        X_eda = add_eda_features(raw_X)

        assert {eda.AGE_IN_DAYS_COL, eda.MONTH_COL, eda.HOUR_COL, eda.WEEKDAY_NAME_COL} <= set(
            X_eda.columns
        )
        assert X_eda[eda.WEEKDAY_NAME_COL].cat.ordered
        assert X_eda.loc[0, eda.AGE_IN_DAYS_COL] == 365.0


    def test_add_eda_features_weekday_categorical_order(self, raw_X):
        """Verify exact order and categories of derived Weekday_Name feature.

        GIVEN: a raw DataFrame with datetime entries
        WHEN: add_eda_features is executed
        THEN: Weekday_Name has exactly 7 ordered categories from 'Mon' to 'Sun'
        """
        X_eda = add_eda_features(raw_X)

        categories = list(X_eda[eda.WEEKDAY_NAME_COL].cat.categories)
        assert categories == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


    def test_add_eda_features_does_not_mutate_input(self, raw_X):
        """Verify that add_eda_features preserves input DataFrame purity.

        GIVEN: a raw input DataFrame
        WHEN: add_eda_features is executed
        THEN: the input DataFrame remains completely unchanged
        """
        original = raw_X.copy(deep=True)

        add_eda_features(raw_X)

        pd.testing.assert_frame_equal(raw_X, original)


    def test_an_unparsable_timestamp_becomes_missing(self, raw_X, caplog):
        """Verify that a bad timestamp does not stop the analysis.

        GIVEN: a raw frame where one timestamp cannot be parsed
        WHEN: add_eda_features is executed with WARNING logging captured
        THEN: its derived features come back missing and a warning counts the
              row, an exploratory pass being where such rows are discovered
        """
        X_bad = raw_X.copy()
        X_bad.loc[0, config.DATETIME_COL] = "not a date"

        with caplog.at_level(logging.WARNING):
            X_eda = add_eda_features(X_bad)

        assert pd.isna(X_eda.loc[0, eda.MONTH_COL])
        assert "could not be parsed" in caplog.text

    def test_an_already_missing_timestamp_triggers_no_warning(self, raw_X, caplog):
        """Verify that an empty cell is not counted as a parsing failure.

        GIVEN: a raw frame where one timestamp is missing rather than malformed
        WHEN: add_eda_features is executed with WARNING logging captured
        THEN: nothing is logged, the count subtracting the cells that were
              already empty from the ones parsing turned into NaT
        """
        X_missing = raw_X.copy()
        X_missing.loc[0, config.DATETIME_COL] = None

        with caplog.at_level(logging.WARNING):
            add_eda_features(X_missing)

        assert "could not be parsed" not in caplog.text

    def test_a_frame_of_unknown_species_yields_no_subsets(self, raw_X):
        """Verify that species the project does not declare are left out.

        GIVEN: a frame whose species column holds only a value absent
        WHEN: split_by_species is executed
        THEN: the mapping comes back empty, the analysis being scoped to the
              species the pipeline trains a model for
        """
        X_alien = raw_X.copy()
        X_alien[config.SPECIES_COL] = "Bird"

        by_species = split_by_species(X_alien)

        assert by_species == {}

    def test_split_by_species_partitions_correctly(self, raw_X):
        """Verify that split_by_species partitions the dataset by species without data loss.

        GIVEN: a DataFrame containing both Dog and Cat entries
        WHEN: split_by_species is executed
        THEN: separate sub-frames are returned per species with no rows lost
        """
        by_species = split_by_species(raw_X)

        assert set(by_species) == {"Dog", "Cat"}
        assert (by_species["Dog"][config.SPECIES_COL] == "Dog").all()
        assert (by_species["Cat"][config.SPECIES_COL] == "Cat").all()
        assert len(by_species["Dog"]) + len(by_species["Cat"]) == len(raw_X)

    def test_split_by_species_ignores_unknown_species(self, raw_X):
        """Verify that split_by_species filters out animal types not present in config.SPECIES.

        GIVEN: a DataFrame containing standard species ('Dog', 'Cat') and an unknown species ('Bird')
        WHEN: split_by_species is executed
        THEN: 'Bird' is ignored and only 'Dog' and 'Cat' are returned in the dictionary
        """
        X_with_unknown = raw_X.copy()
        X_with_unknown.loc[0, config.SPECIES_COL] = "Bird"

        by_species = split_by_species(X_with_unknown)

        assert "Bird" not in by_species
        assert set(by_species.keys()) == {"Dog", "Cat"}

    def test_compute_outcome_crosstab_rows_sum_to_one(self, raw_X):
        """Verify that cross-tabulation row proportions sum up to 1.0.

        GIVEN: a DataFrame with species and outcome columns
        WHEN: compute_outcome_crosstab is executed with default index normalization
        THEN: every row of proportions sums exactly to 1.0
        """
        ct = compute_outcome_crosstab(raw_X, config.SPECIES_COL)

        assert np.allclose(ct.sum(axis=1), 1.0)

    def test_compute_outcome_crosstab_normalization_modes(self, raw_X):
        """Verify compute_outcome_crosstab supports column, total, and raw count normalizations.

        GIVEN: a DataFrame with species and outcome columns
        WHEN: compute_outcome_crosstab is executed with 'columns', 'all', and None normalization
        THEN: columns sum to 1.0, total matrix sums to 1.0, and raw integer counts are returned respectively
        """
        ct_cols = compute_outcome_crosstab(raw_X, config.SPECIES_COL, normalize="columns")
        ct_all = compute_outcome_crosstab(raw_X, config.SPECIES_COL, normalize="all")
        ct_raw = compute_outcome_crosstab(raw_X, config.SPECIES_COL, normalize=None)

        assert np.allclose(ct_cols.sum(axis=0), 1.0)
        assert np.isclose(ct_all.values.sum(), 1.0)
        assert ct_raw.values.sum() == len(raw_X.dropna(subset=[config.SPECIES_COL, config.TARGET_COL]))

    def test_an_unknown_normalize_strategy_raises(self, raw_X):
        """Verify that a misspelt strategy is refused rather than ignored.

        GIVEN: a normalize argument that names no known strategy
        WHEN: compute_outcome_crosstab is executed
        THEN: a ValueError is raised, the counts it would otherwise return
              being silently different from the proportions asked for
        """
        with pytest.raises(ValueError, match="normalize"):
            compute_outcome_crosstab(raw_X, config.SPECIES_COL, normalize="rows")

    def test_compute_outcome_crosstab_single_outcome_value(self, raw_X):
        """Verify compute_outcome_crosstab handles case when target variable has only 1 unique class.

        GIVEN: a DataFrame where all rows have the same OutcomeType ('Adoption')
        WHEN: compute_outcome_crosstab is executed
        THEN: cross-tabulation returns a valid 1-column DataFrame where row sums equal 1.0
        """
        X_single_outcome = raw_X.copy()
        X_single_outcome[config.TARGET_COL] = "Adoption"

        ct = compute_outcome_crosstab(X_single_outcome, config.SEX_COL)

        assert ct.shape[1] == 1
        assert "Adoption" in ct.columns
        assert np.allclose(ct.sum(axis=1), 1.0)

    def test_an_empty_frame_yields_an_empty_crosstab(self):
        """Verify that a frame with no rows produces no table.

        GIVEN: a frame carrying the schema but no rows
        WHEN: compute_outcome_crosstab is executed
        THEN: an empty frame comes back, groupby on no rows having nothing to
              tabulate
        """
        empty = pd.DataFrame(columns=[config.SEX_COL, config.TARGET_COL])

        assert compute_outcome_crosstab(empty, config.SEX_COL).empty

    def test_an_unobserved_category_yields_zeros_and_not_gaps(self):
        """Verify that a declared but empty category does not become missing.

        GIVEN: a categorical feature declaring a level no row uses
        WHEN: compute_outcome_crosstab is executed with index normalization
        THEN: that row comes back as zeros rather than as NaN, which dividing
              an empty row by its own zero sum would otherwise produce
        """
        feature = pd.Categorical(["a", "a", "b"], categories=["a", "b", "c"])
        X = pd.DataFrame({"f": feature, config.TARGET_COL: ["A", "T", "A"]})

        ct = compute_outcome_crosstab(X, "f")

        assert not ct.isna().any().any()
        assert ct.loc["c"].sum() == 0.0

    def test_percentiles_without_the_age_column_raise(self):
        """Verify that asking for percentiles before enrichment is an error.

        GIVEN: a frame that has not been through add_eda_features
        WHEN: compute_age_percentiles is executed
        THEN: a KeyError names the column, the enrichment being a precondition
              rather than something the function does for itself
        """
        with pytest.raises(KeyError, match=eda.AGE_IN_DAYS_COL):
            compute_age_percentiles(pd.DataFrame({"x": [1, 2]}))


    def test_compute_age_percentiles(self, raw_X):
        """Verify reference percentiles for age in days are monotonic and exclude NaNs.

        GIVEN: a DataFrame enriched with age_in_days containing some NaN values
        WHEN: compute_age_percentiles is executed
        THEN: five non-null, strictly monotonic reference percentiles are returned
        """
        X_eda = add_eda_features(raw_X)

        percentiles = compute_age_percentiles(X_eda)

        assert list(percentiles.index) == list(eda.AGE_PERCENTILES)
        assert not percentiles.isnull().any()
        assert percentiles.is_monotonic_increasing


    def test_percentiles_of_an_all_missing_column_are_all_missing(self):
        """Verify that no age at all yields the full index with no values.

        GIVEN: a frame whose age column holds only missing values
        WHEN: compute_age_percentiles is executed
        THEN: the five requested quantiles come back as missing, the shape of
              the result not depending on whether there was data to quantify
        """
        X_nans = pd.DataFrame({eda.AGE_IN_DAYS_COL: [np.nan, np.nan]})

        percentiles = compute_age_percentiles(X_nans)

        assert list(percentiles.index) == list(eda.AGE_PERCENTILES)
        assert percentiles.isna().all()


class TestPlotting:
    def test_plot_target_distribution(self, raw_X):
        """Verify that the target distribution chart renders one bar per outcome class.

        GIVEN: a raw DataFrame containing outcome target classes
        WHEN: plot_target_distribution is executed
        THEN: a (fig, ax) tuple is returned with one bar per target category
        """
        fig, ax = plot_target_distribution(raw_X)

        assert isinstance(fig, plt.Figure)
        assert len(ax.patches) == raw_X[config.TARGET_COL].nunique()
        assert config.TARGET_COL in ax.get_title()


    def test_plot_missing_values_returns_none_when_complete(self, raw_X):
        """Verify that missing values plot returns None when the DataFrame has no NaNs.

        GIVEN: a complete DataFrame with no missing values
        WHEN: plot_missing_values is executed
        THEN: None is returned instead of a Matplotlib figure
        """
        assert plot_missing_values(raw_X.dropna(axis=1)) is None


    def test_plot_missing_values_returns_figure(self, raw_X):
        """Verify that missing values plot returns a valid figure when NaNs are present.

        GIVEN: a DataFrame containing missing values
        WHEN: plot_missing_values is executed
        THEN: a valid (fig, ax) pair is returned
        """
        result = plot_missing_values(raw_X)

        assert result is not None
        assert isinstance(result[0], plt.Figure)
        assert len(result[1].patches) == len(compute_missing_values(raw_X))


    def test_plot_outcome_by_feature(self, raw_X):
        """Verify that stacked outcome proportion plot returns a figure with title set.

        GIVEN: a raw DataFrame with a categorical feature and outcome target
        WHEN: plot_outcome_by_feature is executed with a title
        THEN: a valid (fig, ax) pair is returned with the title correctly configured
        """
        fig, ax = plot_outcome_by_feature(
            raw_X, config.SPECIES_COL, "Outcome by Animal Type"
        )

        assert isinstance(fig, plt.Figure)
        assert ax.get_title() == "Outcome by Animal Type"


    def test_plot_outcome_by_feature_empty_dataframe(self):
        """Verify plot_outcome_by_feature handles completely empty DataFrames without errors.

        GIVEN: an empty DataFrame (0 rows)
        WHEN: plot_outcome_by_feature is executed
        THEN: a valid Figure is returned with title set and no exception is raised
        """
        empty_df = pd.DataFrame(columns=[config.SEX_COL, config.TARGET_COL])

        fig, ax = plot_outcome_by_feature(empty_df, config.SEX_COL, "Empty Title")


        assert isinstance(fig, plt.Figure)
        assert ax.get_title() == "Empty Title"


    def test_plot_age_distribution(self, raw_X):
        """Verify that age distribution plot generates a panel with boxplot and histogram.

        GIVEN: a species-filtered DataFrame enriched with age_in_days
        WHEN: plot_age_distribution is executed for a given species
        THEN: a valid Figure and an array of two Axes are returned
        """
        X_dogs = split_by_species(add_eda_features(raw_X))["Dog"]

        fig, axes = plot_age_distribution(X_dogs, "Dog")

        assert isinstance(fig, plt.Figure)
        assert len(axes) == 2


    def test_plot_top_categories_annotates_unique_count(self, raw_X):
        """Verify that top-N category bar chart includes unique category count text.

        GIVEN: a species-filtered DataFrame with high-cardinality features
        WHEN: plot_top_categories is executed
        THEN: the rendered plot contains text annotating total unique categories
        """
        X_dogs = split_by_species(raw_X)["Dog"]

        _, ax = plot_top_categories(X_dogs, config.BREED_COL, "Dog", top_n=5)

        assert any("Unique Categories" in t.get_text() for t in ax.texts)


    def test_plot_top_categories_respects_top_n(self, raw_X):
        """Verify that plot_top_categories restricts displayed bars to top_n.

        GIVEN: a species dataset with multiple categories and top_n=2
        WHEN: plot_top_categories is executed
        THEN: exactly top_n bars are rendered in the figure
        """
        X_dogs = split_by_species(raw_X)["Dog"]

        _, ax = plot_top_categories(X_dogs, config.BREED_COL, "Dog", top_n=2)

        assert len(ax.patches) == 2

    def test_plot_top_categories_fewer_categories_than_top_n(self, raw_X):
        """Verify plot_top_categories works when total unique categories < top_n.

        GIVEN: a species frame with only 2 unique breeds and top_n=30 requested
        WHEN: plot_top_categories is executed
        THEN: it renders exactly 2 bars without error
        """
        X_dogs = split_by_species(raw_X)["Dog"]

        _, ax = plot_top_categories(X_dogs, config.BREED_COL, "Dog", top_n=30)

        assert len(ax.patches) == X_dogs[config.BREED_COL].nunique()
        assert len(ax.patches) < 30


    def test_plot_temporal_outcomes(self, raw_X):
        """Verify temporal outcome plot renders a three-panel grid across time features.

        GIVEN: a species-filtered DataFrame enriched with temporal features
        WHEN: plot_temporal_outcomes is executed
        THEN: a valid Figure with three Axes (Month, Weekday, Hour) is returned
        """
        X_dogs = split_by_species(add_eda_features(raw_X))["Dog"]

        fig, axes = plot_temporal_outcomes(X_dogs, "Dog")

        assert isinstance(fig, plt.Figure)
        assert len(axes) == 3


    def test_plot_outcome_by_feature_handles_nans(self, raw_X):
        """Verify plot_outcome_by_feature handles missing values (NaN) in SexuponOutcome cleanly.

        GIVEN: a DataFrame where SexuponOutcome contains NaN values
        WHEN: plot_outcome_by_feature is executed
        THEN: it renders successfully without raising errors
        """
        X_with_nans = raw_X.copy()
        X_with_nans.loc[0:5, config.SEX_COL] = np.nan

        fig, ax = plot_outcome_by_feature(
            X_with_nans, config.SEX_COL, "Outcome by Sex (with NaNs)"
        )

        assert isinstance(fig, plt.Figure)
        assert ax.get_title() == "Outcome by Sex (with NaNs)"

class TestBuildFigures:

    def test_only_the_overall_figures_survive_without_a_known_species(self, raw_X):
            """Verify that the catalogue degrades to its species-independent part.

            GIVEN: a frame whose species column holds no declared species
            WHEN: build_figures is executed
            THEN: only the figures that do not depend on a species are built, the
                  per-species loop having nothing to iterate over
            """
            X_alien = raw_X.copy()
            X_alien[config.SPECIES_COL] = "Bird"

            figures = build_figures(X_alien)

            assert not any(
                key.endswith(("_dog", "_cat")) for key in figures
            )
            assert "target_distribution" in figures


    def test_build_figures_covers_every_expected_plot(self, raw_X):
            """Verify that the catalogue holds one entry per figure the report needs.

            GIVEN: a raw frame carrying both species
            WHEN: build_figures is executed
            THEN: the keys name the overall figures and the per-species ones, and
                  every value is a Figure the caller now owns
            """
            figures = build_figures(raw_X)

            assert {"target_distribution", "outcome_by_animal_type"} <= set(figures)
            assert {"age_distribution_dog", "age_distribution_cat"} <= set(figures)
            assert all(isinstance(fig, plt.Figure) for fig in figures.values())

    def test_every_declared_species_gets_its_five_figures(self, raw_X):
            """Verify that no per-species figure silently drops out of the catalogue.

            GIVEN: a frame carrying both declared species
            WHEN: build_figures is executed
            THEN: each species contributes the same five figures, so that a plot
                  removed from the loop is caught here and not by its absence from the report
            """
            figures = build_figures(raw_X)

            per_species = {
                species.lower(): {
                    key for key in figures if key.endswith(f"_{species.lower()}")
                }
                for species in config.SPECIES
            }

            assert all(len(keys) == 5 for keys in per_species.values())

    def test_a_complete_frame_yields_no_missing_values_figure(self, raw_X):
        """Verify that the missing-values plot is omitted when there is nothing to show.

        GIVEN: a raw frame whose gaps have all been filled
        WHEN: build_figures is executed
        THEN: no missing-values figure is built, plot_missing_values having
              returned None rather than an empty chart
        """
        complete = raw_X.fillna(
            {config.NAME_COL: "Rex", config.SUBTARGET_COL: "Partner",
             config.AGE_COL: "1 year"}
        )

        figures = build_figures(complete)

        assert "missing_values" not in figures

class TestSaveFigures:

    def test_save_figures_single_species_only(self, raw_X, tmp_path):
        """Verify save_figures handles datasets containing only one species (e.g. only Dogs).

        GIVEN: a DataFrame containing only 'Dog' in the species column
        WHEN: save_figures is executed
        THEN: only dog-related species figures are generated without crashing
        """
        figures_dir = tmp_path / "figures"
        X_dogs_only = raw_X[raw_X[config.SPECIES_COL] == "Dog"].copy()

        written = save_figures(X_dogs_only, figures_dir)

        names = {path.stem for path in written}
        assert "outcome_by_sex_dog" in names
        assert "outcome_by_sex_cat" not in names


    def test_save_figures_writes_files(self, raw_X, tmp_path):
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

    def test_save_figures_overwrites_existing_files(self, raw_X, tmp_path):
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

    def test_the_written_paths_come_back_sorted(self, raw_X, tmp_path):
        """Verify that the returned paths are ordered, as the docstring says.

        GIVEN: a raw frame and a destination directory
        WHEN: save_figures is executed
        THEN: the paths come back sorted, dictionary insertion order not being
              something the caller should have to rely on
        """
        written = save_figures(raw_X, tmp_path / "figures")

        assert written == sorted(written)

class TestMain:

    def test_main_end_to_end(self, raw_X, tmp_path):
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

class TestParseArgs:

    def test_parse_args_defaults(self):
        """Verify that CLI argument parser returns correct default paths.

        GIVEN: an empty argument list
        WHEN: parse_args is executed
        THEN: documented default paths for raw CSV and figures directory are assigned
        """
        args = parse_args([])

        assert args.csv_path == config.RAW_DATA_PATH
        assert args.figures_dir == config.FIGURES_DIR


    def test_parse_args_custom_values(self):
        """Verify that CLI argument parser correctly overrides default arguments.

        GIVEN: explicit command-line argument strings
        WHEN: parse_args is executed
        THEN: provided paths override default configurations
        """
        args = parse_args(["my.csv", "--figures-dir", "out/figs"])

        assert args.csv_path.name == "my.csv"
        assert str(args.figures_dir) == "out/figs"

    def test_an_unknown_option_is_refused(self):
            """Verify that a misspelt option stops the run.

            GIVEN: an argument list holding an option the parser does not declare
            WHEN: parse_args is executed
            THEN: SystemExit is raised, argparse reporting the error itself
            """
            with pytest.raises(SystemExit):
                parse_args(["--nonexistent", "value"])




