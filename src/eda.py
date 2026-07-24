"""Exploratory Data Analysis script for the Shelter Animal Outcomes dataset.

Loads the raw dataset, performs feature enrichment (age in days, temporal breakdowns),
and computes statistical distributions. Generates a comprehensive suite of publication-ready
visualization plots—including target imbalances, missing value profiles, species-split age
distributions, binned high-cardinality features, and temporal trends—and persists them to disk
without displaying.

CLI Usage
---------
Using default paths (run as a module from the project root):
python -m src.eda

Or specifying custom input data and output figures directory:
python -m src.eda data/raw_data/train.csv --figures-dir reports/figures
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

from src.feature_engineering import RareCategoriesGrouper, extract_primary_breed
from src.preprocessing import extract_age_in_days

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global Constants
# ---------------------------------------------------------------------------

TARGET_COL = "OutcomeType"
SPECIES_COLORS = {"Dog": "teal", "Cat": "salmon"}
WEEKDAY_ORDER = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
DAY_OF_WEEK_MAP = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
AGE_PERCENTILES = (0.5, 0.9, 0.95, 0.99, 0.999)
DEFAULT_TOP_N = 30

# ---------------------------------------------------------------------------
# Data preparation 
# ---------------------------------------------------------------------------

def compute_missing_values(X: pd.DataFrame) -> pd.Series:
    """Identify and count missing values per column.

    Parameters
    ----------
    X : pd.DataFrame
        Input DataFrame to inspect for missing values.

    Returns
    -------
    pd.Series
        Series mapping column names to missing value counts, sorted descending.
        Only columns with at least one missing value are returned.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"A": [1, None], "B": [2, 3]})
    >>> compute_missing_values(df).to_dict()
    {'A': 1}
    """
    missing = X.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    logger.info("Found %d columns with missing values", len(missing))
    return missing


def add_eda_features(X: pd.DataFrame) -> pd.DataFrame:
    """Enrich input DataFrame with temporal features and age in days for EDA.

    Parameters
    ----------
    X : pd.DataFrame
        Raw input DataFrame containing 'AgeuponOutcome' and 'DateTime' columns.

    Returns
    -------
    pd.DataFrame
        A copy of `X` enriched with 'age_in_days', 'Month', 'Hour', and
        ordered categorical 'Weekday_Name'.
    """
    X = X.copy()
    X["age_in_days"] = extract_age_in_days(X["AgeuponOutcome"])
    X["DateTime"] = pd.to_datetime(X["DateTime"])
    X["Month"] = X["DateTime"].dt.month
    X["Hour"] = X["DateTime"].dt.hour
    
    weekday_abbr = X["DateTime"].dt.dayofweek.map(DAY_OF_WEEK_MAP)
    X["Weekday_Name"] = pd.Categorical(
        weekday_abbr, categories=list(WEEKDAY_ORDER), ordered=True
    )
    return X


def split_by_species(X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Partition DataFrame into separate sub-frames per species ('Dog', 'Cat').

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing an 'AnimalType' column.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping species names ('Dog', 'Cat') to their respective DataFrame subsets.
    """
    return {
        species: X[X["AnimalType"] == species].copy()
        for species in SPECIES_COLORS
        if species in SPECIES_COLORS
    }


def compute_outcome_crosstab(
    X: pd.DataFrame, feature: str, normalize: str = "index"
) -> pd.DataFrame:
    """Compute cross-tabulation of a target feature against OutcomeType.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing the feature and 'OutcomeType'.
    feature : str
        Name of the feature column to cross-tabulate against 'OutcomeType'.
    normalize : str, default="index"
        Normalization strategy passed to `pd.crosstab` ('index', 'columns', or 'all').

    Returns
    -------
    pd.DataFrame
        Cross-tabulated frequency or proportion table.
    """
    ct = (
        X.groupby([feature, TARGET_COL], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    if normalize == "index":
        return ct.div(ct.sum(axis=1), axis=0)
    elif normalize == "columns":
        return ct.div(ct.sum(axis=0), axis=1)
    elif normalize == "all":
        return ct / ct.values.sum()
    return ct


def compute_age_percentiles(X: pd.DataFrame) -> pd.Series:
    """Calculate reference quantiles for the 'age_in_days' feature.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing the 'age_in_days' column.

    Returns
    -------
    pd.Series
        Series of quantile values corresponding to `AGE_PERCENTILES` (0.5 to 0.999).
    """
    return X["age_in_days"].dropna().quantile(list(AGE_PERCENTILES))


def compute_binned_frequencies(
    series: pd.Series, max_other_ratio: float = 0.15
) -> pd.Series:
    """Compute normalized category frequencies after grouping rare labels into 'Other'.

    Leverages `RareCategoriesGrouper` to mirror pipeline transformation logic and lists
    the grouped 'Other' category last.

    Parameters
    ----------
    series : pd.Series
        Categorical Series to bin.
    max_other_ratio : float, default=0.15
        Maximum proportion threshold allowed for the 'Other' category.

    Returns
    -------
    pd.Series
        Normalized category frequencies with 'Other' placed at the end of the index.
    """
    frame = series.rename("value").to_frame()
    grouper = RareCategoriesGrouper(columns=["value"], max_other_ratio=max_other_ratio)
    binned = grouper.fit_transform(frame)["value"]
    freqs = binned.value_counts(normalize=True)
    if "Other" in freqs.index:
        freqs = pd.concat([freqs.drop("Other"), freqs.loc[["Other"]]])
    return freqs


# ---------------------------------------------------------------------------
# Plotting 
# ---------------------------------------------------------------------------

def plot_target_distribution(
    X: pd.DataFrame, figsize: tuple[int, int] = (8, 5)
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart of target variable distribution with percentage annotations.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing the `TARGET_COL` ('OutcomeType').
    figsize : tuple[int, int], default=(8, 5)
        Dimensions of the output figure.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib Figure and Axes objects containing the rendered plot.
    """
    counts = X[TARGET_COL].value_counts()
    pct = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(counts.index, counts.values,
                  color=sns.color_palette("Set2", len(counts)))
    ax.bar_label(bars, labels=[f"{p:.1f}%" for p in pct], padding=3, fontsize=10)
    ax.set_title(f"Target Distribution ({TARGET_COL})", fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig, ax


def plot_missing_values(
    X: pd.DataFrame, figsize: tuple[int, int] = (8, 4)
) -> tuple[plt.Figure, plt.Axes] | None:
    """Bar chart of missing value counts per outcome value.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame to inspect for missing values.
    figsize : tuple[int, int], default=(8, 4)
        Dimensions of the output figure.

    Returns
    -------
    tuple[plt.Figure, plt.Axes] | None
        Figure and Axes objects if missing values exist; None if no missing values are found.
    """
    missing = compute_missing_values(X)
    if missing.empty:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(missing.index, missing.values, color="salmon")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
    ax.set_title("Missing Values by Column", fontweight="bold")
    ax.set_ylabel("Missing Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig, ax


def plot_outcome_by_feature(
    X: pd.DataFrame, feature: str, title: str,
    figsize: tuple[int, int] = (10, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Stacked bar chart of normalized outcome proportions across each level of the specified feature.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing target column and the specified feature.
    feature : str
        Column name of the feature to analyze.
    title : str
        Title for the plot.
    figsize : tuple[int, int], default=(10, 5)
        Dimensions of the output figure.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib Figure and Axes objects containing the rendered plot.
    """
    ct = compute_outcome_crosstab(X, feature)

    fig, ax = plt.subplots(figsize=figsize)
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Proportion")
    ax.set_xlabel(feature)
    ax.legend(title=TARGET_COL, loc="upper right", bbox_to_anchor=(1.25, 1))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig, ax


def plot_age_distribution(
    X: pd.DataFrame, species: str, figsize: tuple[int, int] = (15, 5)
) -> tuple[plt.Figure, Any]:
    """Side-by-side boxplot and histogram of 'age_in_days' for a species.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing 'age_in_days'.
    species : str
        Target species name ('Dog' or 'Cat') for plot styling.
    figsize : tuple[int, int], default=(15, 5)
        Dimensions of the output figure.

    Returns
    -------
    tuple[plt.Figure, Any]
        Matplotlib Figure and array of Axes objects containing the two plots.
    """
    color = SPECIES_COLORS.get(species, "grey")
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sns.boxplot(data=X, x="age_in_days", ax=axes[0], color=color)
    axes[0].set_title(f"Boxplot of Age in Days ({species}s)")
    axes[0].set_xlabel("Age (days)")

    sns.histplot(data=X, x="age_in_days", ax=axes[1], bins=50, color=color)
    axes[1].set_title(f"Distribution of Age in Days ({species}s)")
    axes[1].set_xlabel("Age (days)")

    fig.tight_layout()
    logger.info("[%s] age percentiles:\n%s",
                species, compute_age_percentiles(X).to_string())
    return fig, axes


def plot_top_categories(
    X: pd.DataFrame, feature: str, species: str,
    top_n: int = DEFAULT_TOP_N, figsize: tuple[int, int] = (12, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart of Top-N categories for high-cardinality features.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing target feature column.
    feature : str
        Target categorical feature name (e.g., 'Breed', 'Color').
    species : str
        Target species name ('Dog' or 'Cat') for plot styling.
    top_n : int, default=30
        Number of top categories to display.
    figsize : tuple[int, int], default=(12, 5)
        Dimensions of the output figure.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib Figure and Axes objects containing the rendered plot.
    """
    counts = X[feature].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(counts.index, counts.values, color=SPECIES_COLORS.get(species, "grey"))
    ax.set_title(f"Top {top_n} {species} {feature}s", fontweight="bold")
    ax.set_ylabel("Count")
    ax.text(0.95, 0.85, f"Unique Categories: {X[feature].nunique()}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"))
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return fig, ax


def plot_binned_frequencies(
    freqs: pd.Series, title: str, palette: str = "plasma",
    figsize: tuple[int, int] = (8, 7),
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal bar chart of binned category frequencies with percentage labels.

    Parameters
    ----------
    freqs : pd.Series
        Series of normalized category frequencies.
    title : str
        Title for the plot.
    palette : str, default="plasma"
        Seaborn color palette name.
    figsize : tuple[int, int], default=(8, 7)
        Dimensions of the output figure.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib Figure and Axes objects containing the rendered plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=freqs.values, y=freqs.index, ax=ax,
                hue=freqs.index, palette=palette, legend=False)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Relative Frequency")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for container in ax.containers:
        ax.bar_label(container, fmt=lambda v: f"{v * 100:.1f}%",
                     padding=4, fontsize=8.5)
    sns.despine(fig=fig, top=True, right=True)
    fig.tight_layout()
    return fig, ax


def plot_temporal_outcomes(
    X: pd.DataFrame, species: str, figsize: tuple[int, int] = (18, 5)
) -> tuple[plt.Figure, Any]:
    """3-panel plot showing outcome proportions across Month, Day of Week, and Hour.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame enriched with temporal features ('Month', 'Weekday_Name', 'Hour').
    species : str
        Target species name ('Dog' or 'Cat').
    figsize : tuple[int, int], default=(18, 5)
        Dimensions of the output figure.

    Returns
    -------
    tuple[plt.Figure, Any]
        Matplotlib Figure and array of Axes objects containing the three temporal plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    temporal_features = {
        "Month": "Month",
        "Weekday_Name": "Day of Week",
        "Hour": "Hour of Day",
    }

    # Scompattiamo sia la chiave (feature) che il valore (display_name)
    for ax, (feature, display_name) in zip(axes, temporal_features.items()):
        ct = compute_outcome_crosstab(X, feature)
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", legend=False)
        ax.set_title(
            f"Outcome by {display_name} ({species}s)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(display_name)
        ax.tick_params(axis="x", rotation=0, labelsize=9)

    axes[0].set_ylabel("Proportion")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title=TARGET_COL,
        loc="center left",
        bbox_to_anchor=(0.91, 0.85),
    )
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    return fig, axes

# ---------------------------------------------------------------------------
# Orchestration & CLI
# ---------------------------------------------------------------------------

def save_figures(X: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Render and save the complete set of EDA figures to disk.

    Parameters
    ----------
    X : pd.DataFrame
        Raw input DataFrame.
    figures_dir : Path
        Target directory where output PNG files will be saved.

    Returns
    -------
    list[Path]
        Sorted list of file paths written to disk.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    X_eda = add_eda_features(X)
    by_species = split_by_species(X_eda)

    figures: dict[str, plt.Figure] = {
        "target_distribution": plot_target_distribution(X)[0],
        "outcome_by_animal_type": plot_outcome_by_feature(
            X, "AnimalType", "Outcome Distribution by AnimalType")[0],
    }
    missing = plot_missing_values(X)
    if missing is not None:
        figures["missing_values"] = missing[0]

    for species, X_species in by_species.items():
        key = species.lower()
        figures[f"age_distribution_{key}"] = plot_age_distribution(
            X_species, species)[0]
        figures[f"top_breeds_{key}"] = plot_top_categories(
            X_species, "Breed", species)[0]
        figures[f"top_colors_{key}"] = plot_top_categories(
            X_species, "Color", species)[0]
        figures[f"temporal_outcomes_{key}"] = plot_temporal_outcomes(
            X_species, species)[0]

        primary_breed_freqs = compute_binned_frequencies(
            extract_primary_breed(X_species["Breed"]))
        figures[f"primary_breed_binned_{key}"] = plot_binned_frequencies(
            primary_breed_freqs, f"Primary Breed Distribution - {species}s",
            palette="plasma" if species == "Dog" else "viridis")[0]

    written = []
    for name, fig in figures.items():
        path = figures_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    logger.info("Saved %d figures to %s", len(written), figures_dir)
    return sorted(written)


def main(csv_path: Path, figures_dir: Path) -> None:
    """Load raw dataset and trigger full EDA figure generation.

    Parameters
    ----------
    csv_path : Path
        Path to the raw CSV data file.
    figures_dir : Path
        Target directory path for generated figures.
    """
    logger.info("Loading raw dataset from %s", csv_path)
    X = pd.read_csv(csv_path)
    logger.info("Loaded %d rows, %d columns", *X.shape)
    save_figures(X, figures_dir)


def parse_args(args_list: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for CLI execution.

    Parameters
    ----------
    args_list : list[str] | None, default=None
        Explicit list of argument strings. If None, parses `sys.argv`.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, nargs="?",
                        default=Path("data/raw_data/train.csv"),
                        help="Path to the raw train.csv")
    parser.add_argument("--figures-dir", type=Path,
                        default=Path("reports/figures"),
                        help="Directory where figures are written")
    return parser.parse_args(args_list)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    main(args.csv_path, args.figures_dir)