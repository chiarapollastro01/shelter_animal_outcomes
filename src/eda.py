"""Exploratory Data Analysis script for the Shelter Animal Outcomes dataset.

Loads the raw dataset, performs feature enrichment (age in days, temporal breakdowns),
and computes statistical distributions. Generates a comprehensive suite
of plots, including target imbalances, missing value profiles, species-split age
distributions, species-split sex distributions, top breed and color categories, and temporal trends; and
and and persists them to disk one at a time, without displaying them.

Explanations about these generated visualizations are documented in the accompanying EDA report (`reports/eda.md`).

Exported Functions
------------------
Data preparation: compute_missing_values, add_eda_features, split_by_species,
    compute_outcome_crosstab, compute_age_percentiles.

Plotting: one function per figure, each returning the Figure and Axes without
    writing anything, so that a plot can be inspected in a notebook or asserted
    on in a test.

Orchestration: build_figures assembles the whole catalogue in memory,
    save_figures writes it to disk, main reads the CSV and calls them.
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
import numpy as np
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
 
from src import config
from src.preprocessing import extract_age_in_days

logger = logging.getLogger(__name__)


SPECIES_COLORS: dict[str, str] = {"Dog": "teal", "Cat": "salmon"}
DAY_OF_WEEK_MAP: dict[int, str] = {
    0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"
}
WEEKDAY_ORDER: tuple[str, ...] = tuple(DAY_OF_WEEK_MAP[i] for i in range(7))
AGE_PERCENTILES: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99, 0.999)
DEFAULT_TOP_N: int = 30

# Columns this module derives for the analysis. They stay here rather than in
# config because nothing outside the EDA reads them.
AGE_IN_DAYS_COL: str = "age_in_days"
MONTH_COL: str = "Month"
HOUR_COL: str = "Hour"
WEEKDAY_NAME_COL: str = "Weekday_Name"

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
    """Enrich the frame with the temporal features and the age in days.

    Unparsable timestamps become NaT (not a time) and their derived features NaN, rather
    than stopping the analysis.

    Parameters
    ----------
    X : pd.DataFrame
        Raw frame carrying the age and datetime columns named in config.

    Returns
    -------
    pd.DataFrame
        A copy enriched with AGE_IN_DAYS_COL, MONTH_COL, HOUR_COL and an ordered
        categorical WEEKDAY_NAME_COL.
    """
    X_eda = X.copy()
    X_eda[AGE_IN_DAYS_COL] = extract_age_in_days(X_eda[config.AGE_COL])

    parsed = pd.to_datetime(X_eda[config.DATETIME_COL], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    n_unparsable = int(parsed.isna().sum() - X_eda[config.DATETIME_COL].isna().sum())
    if n_unparsable:
        logger.warning(
            "%d rows carry a timestamp that could not be parsed", n_unparsable
        )

    X_eda[config.DATETIME_COL] = parsed
    X_eda[MONTH_COL] = parsed.dt.month
    X_eda[HOUR_COL] = parsed.dt.hour
    X_eda[WEEKDAY_NAME_COL] = pd.Categorical(
        parsed.dt.dayofweek.map(DAY_OF_WEEK_MAP),
        categories=list(WEEKDAY_ORDER),
        ordered=True,
    )
    return X_eda



def split_by_species(X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Partition DataFrame into separate sub-frames per species.

    The set of species comes from `config.SPECIES`. 
    Only species with at least 1 row present in X will be included.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing species column (`config.SPECIES_COL`).

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping species names to their respective non-empty DataFrame subsets.
    """
    return {
        species: sub
        for species in config.SPECIES
        if not (sub := X[X[config.SPECIES_COL] == species].copy()).empty
    }


def compute_outcome_crosstab(
    X: pd.DataFrame, feature: str, normalize: str | None = "index"
) -> pd.DataFrame:
    """Compute cross-tabulation of a target feature against config.TARGET_COL safely.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing the feature and `config.TARGET_COL`.
    feature : str
        Name of the feature column to cross-tabulate against `config.TARGET_COL`.
    normalize : str, default="index"
        Normalization strategy ('index', 'columns', 'all', or None for raw counts).

    Returns
    -------
    pd.DataFrame
        Cross-tabulated frequency or proportion table.
    
    Raises
    -------
    ValueError
        If the normalize value is unknown.
    """
    if X.empty:
        return pd.DataFrame()

    ct = (
        X.groupby([feature, config.TARGET_COL], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    if normalize == "index":
    # replace(0, 1) guards the division: a declared category nobody used
    # has a row summing to zero, and 0/0 would spread NaN into the plots.
        row_sums = ct.sum(axis=1)
        return ct.div(row_sums.replace(0, 1), axis=0).fillna(0)
    if normalize == "columns":
        col_sums = ct.sum(axis=0)
        return ct.div(col_sums.replace(0, 1), axis=1).fillna(0)
    if normalize == "all":
        total = ct.values.sum()
        return (ct / total).fillna(0) if total > 0 else ct
    if normalize is None:
        return ct

    raise ValueError(
        f"Unknown normalize strategy {normalize!r}: expected 'index', "
        "'columns', 'all' or None."
    )


def compute_age_percentiles(X: pd.DataFrame) -> pd.Series:
    """Calculate reference quantiles for the AGE_IN_DAYS_COL feature.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing the AGE_IN_DAYS_COL column.

    Returns
    -------
    pd.Series
        Series of quantile values corresponding to `AGE_PERCENTILES` (0.5 to 0.999).
    """
    return X[AGE_IN_DAYS_COL].dropna().quantile(list(AGE_PERCENTILES))


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
        DataFrame containing `config.TARGET_COL`.
    figsize : tuple[int, int], default=(8, 5)
        Dimensions of the output figure.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib Figure and Axes objects containing the rendered plot.
    """
    counts = X[config.TARGET_COL].value_counts()
    pct = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(counts.index, counts.to_numpy(),
                  color=sns.color_palette("Set2", len(counts)))
    ax.bar_label(bars, labels=[f"{p:.1f}%" for p in pct], padding=3, fontsize=10)
    ax.set_title(f"Target Distribution ({config.TARGET_COL})", fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig, ax


def plot_missing_values(
    X: pd.DataFrame, figsize: tuple[int, int] = (8, 4)
) -> tuple[plt.Figure, plt.Axes] | None:
    """Bar chart of missing value counts per column.

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
    bars = ax.bar(missing.index, missing.to_numpy(), color="salmon")
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
    fig, ax = plt.subplots(figsize=figsize)
    if X.empty:
        ax.set_title(title, fontweight="bold")
        fig.tight_layout()
        return fig, ax

    ct = compute_outcome_crosstab(X, feature)
    if not ct.empty:
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Proportion")
    ax.set_xlabel(feature)
    ax.legend(title=config.TARGET_COL, loc="upper right", bbox_to_anchor=(1.25, 1))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig, ax


def plot_age_distribution(
    X: pd.DataFrame, species: str, figsize: tuple[int, int] = (15, 5)
) -> tuple[plt.Figure, np.ndarray]:
    """Side-by-side boxplot and histogram of AGE_IN_DAYS_COL for a species.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing AGE_IN_DAYS_COL.
    species : str
        Target species name ('Dog' or 'Cat') for plot styling.
    figsize : tuple[int, int], default=(15, 5)
        Dimensions of the output figure.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Matplotlib Figure and a 1-D array of the two Axes.
    """
    color = SPECIES_COLORS.get(species, "grey")
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sns.boxplot(data=X, x=AGE_IN_DAYS_COL, ax=axes[0], color=color)
    axes[0].set_title(f"Boxplot of Age in Days ({species}s)")
    axes[0].set_xlabel("Age (days)")

    sns.histplot(data=X, x=AGE_IN_DAYS_COL, ax=axes[1], bins=50, color=color)
    axes[1].set_title(f"Distribution of Age in Days ({species}s)")
    axes[1].set_xlabel("Age (days)")

    fig.tight_layout()
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
        Target categorical feature name (e.g., config.BREED_COL, config.COLOR_COL).
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
    ax.bar(counts.index, counts.to_numpy(), color=SPECIES_COLORS.get(species, "grey"))
    ax.set_title(f"Top {top_n} {species} {feature}s", fontweight="bold")
    ax.set_ylabel("Count")
    ax.text(0.95, 0.85, f"Unique Categories: {X[feature].nunique()}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"))
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return fig, ax


def plot_temporal_outcomes(
    X: pd.DataFrame, species: str, figsize: tuple[int, int] = (18, 5)
) -> tuple[plt.Figure, np.ndarray]:
    """3-panel plot showing outcome proportions across Month, Day of Week, and Hour.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame enriched with temporal features (MONTH_COL, WEEKDAY_NAME_COL, HOUR_COL).
    species : str
        Target species name ('Dog' or 'Cat').
    figsize : tuple[int, int], default=(18, 5)
        Dimensions of the output figure.

    Returns
    -------
    tuple[plt.Figure, Any]
        Matplotlib Figure and a 1-D array of the two Axes.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    temporal_features = {
        MONTH_COL: "Month",
        WEEKDAY_NAME_COL: "Day of Week",
        HOUR_COL: "Hour of Day",
    }

    for ax, (feature, display_name) in zip(axes, temporal_features.items()):
        ct = compute_outcome_crosstab(X, feature)
        if not ct.empty:
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
    if handles:
        fig.legend(
            handles,
            labels,
            title=config.TARGET_COL,
            loc="center left",
            bbox_to_anchor=(0.91, 0.85),
        )
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    return fig, axes


# ---------------------------------------------------------------------------
# Orchestration & CLI
# ---------------------------------------------------------------------------

def build_figures(X: pd.DataFrame) -> dict[str, plt.Figure]:
    """Build every EDA figure and return them keyed by output file stem.
 
    Kept separate from the saving step so that the figure catalogue can be
    inspected and tested without touching the filesystem.
 
    Parameters
    ----------
    X : pd.DataFrame
        Raw dataset, as read from the source CSV.
 
    Returns
    -------
    dict[str, plt.Figure]
        Mapping between the output file stem and its rendered Figure. The
        caller owns the figures and is responsible for closing them.
    """
    X_eda = add_eda_features(X)
    by_species = split_by_species(X_eda)
 
    figures: dict[str, plt.Figure] = {
        "target_distribution": plot_target_distribution(X)[0],
        "outcome_by_animal_type": plot_outcome_by_feature(
            X, config.SPECIES_COL, f"Outcome Distribution by {config.SPECIES_COL}"
        )[0],
    }
 
    missing = plot_missing_values(X)
    if missing is not None:
        figures["missing_values"] = missing[0]
 
    for species, X_species in by_species.items():
        key = species.lower()
        logger.info(
            "[%s] age percentiles:\n%s",
            species, compute_age_percentiles(X_species).to_string(),
        )
        figures[f"age_distribution_{key}"] = plot_age_distribution(X_species, species)[0]
        figures[f"outcome_by_sex_{key}"] = plot_outcome_by_feature(
            X_species, config.SEX_COL, f"Outcome Distribution by Sex upon Outcome ({species}s)"
        )[0]
        figures[f"top_breeds_{key}"] = plot_top_categories(X_species, config.BREED_COL, species)[0]
        figures[f"top_colors_{key}"] = plot_top_categories(X_species, config.COLOR_COL, species)[0]
        figures[f"temporal_outcomes_{key}"] = plot_temporal_outcomes(X_species, species)[0]
 
    return figures
 
 
def save_figures(X: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Render every EDA figure and write it to disk.
 
    Figures are written one at a time. Matplotlib does not guarantee
    thread-safety, and this step produces a handful of files once, offline, so
    concurrency would buy a couple of seconds in exchange for a correctness
    argument that cannot be made from the library documentation.
 
    Parameters
    ----------
    X : pd.DataFrame
        Raw dataset, as read from the source CSV.
    figures_dir : Path
        Destination directory. Created if it does not exist.
 
    Returns
    -------
    list[Path]
        Sorted paths of the figures written to disk.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = build_figures(X)
 
    written: list[Path] = []
    for name, fig in figures.items():
        out_path = figures_dir / f"{name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)  # release the figure as soon as it is on disk
        written.append(out_path)
 
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
                        default=config.RAW_DATA_PATH,
                        help="Path to the raw train.csv")
    parser.add_argument("--figures-dir", type=Path,
                        default=config.FIGURES_DIR,
                        help="Directory where figures are written")
    return parser.parse_args(args_list)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    main(args.csv_path, args.figures_dir)