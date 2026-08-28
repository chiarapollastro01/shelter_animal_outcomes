# Shelter Animal Outcomes

Multi-class classification task regarding the Austin Animal Center dataset:
the goal is to predict what happens to an animal leaving the shelter. It can
either be adoption, transfer, return to owner, euthanasia or death.
Dogs and cats get their own model. The EDA (Exploratory Data Analysis) showed 
that the two populations behave quite differently. Cats are transferred half the 
time, dogs are returned to their owner six times more often, a single model
 would average away what distinguishes them.

## Requirements

Python 3.10 or later. The lower bound comes from the `X | None` syntax used
throughout the type hints.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The dependencies are declared in `pyproject.toml`, with a lower bound at the
version the project was developed against and an upper bound excluding the next
major, so that a future release cannot silently change the results.


## The data

The raw file is not versioned. Download `train.csv` from the
[Shelter Animal Outcomes](https://www.kaggle.com/c/shelter-animal-outcomes)
competition and place it at `data/raw_data/train.csv`.

`data/metadata/` documents where the file comes from, what its columns mean and
what is missing from them, and carries the checksum of the copy this analysis
was run on:

```bash
sha256sum -c data/metadata/checksums.txt
```

## Running the pipeline

```bash
snakemake --cores all
```

That produces, in order: the train/test split under `data/split_data/`, one
model per species under `models/`, and the metrics report at
`reports/metrics.json`. Everything is rebuilt only when its inputs are newer,
so a second run with nothing changed does nothing.

Useful variations:

```bash
snakemake --cores all -n            # show what would run, without running it
snakemake --cores all --forceall    # rebuild everything from scratch
snakemake --cores all --detailed-summary   # provenance of every output
```


### About the cores

`--cores` bounds how many Snakemake jobs run at once, but the two tournaments
are declared with `threads: workflow.cores`, so they run one after the other.
That is deliberate: `GridSearchCV` inside each of them already uses
`n_jobs=-1`, and overlapping the two would have them compete for the same CPUs.
The parallelism that pays here is in the search, not between the species.

### Running a single step

Every module is a command-line program, and the Snakefile calls those programs
through `shell:`. Each can be run on its own during development:

```bash
python -m src.prepare_data data/raw_data/train.csv --output-dir data/split_data
python -m src.train --species Dog
python -m src.evaluate
```

`python -m src.<module> --help` lists the options of each.

### The exploratory analysis

`src/eda.py` is not part of the Snakefile: it produces figures that no later
step consumes, and rebuilding them on every change of the raw file would cost
time for nothing. Run it when you want them:

```bash
python -m src.eda data/raw_data/train.csv --figures-dir reports/figures
```

## Tests

```bash
pytest
mypy src/
pylint src/ tests/
```

`pytest` collects both `tests/` and `src/`: the second because the docstring
examples are run as part of the suite, so an example cannot drift away from the
code it documents.

## Layout

```
config.yaml            run parameters: split sizes, folds, metrics, search grids
Snakefile              the dependency graph of the three pipeline steps
pyproject.toml         package metadata, dependencies, tool configuration
data/metadata/         provenance and checksum of the raw file
data/raw_data/         the Kaggle file (not versioned)
data/split_data/       train/test split (not versioned, rebuilt by the pipeline)
models/                fitted pipelines and their metadata (not versioned)
reports/eda.md         what the exploratory phase found
reports/results.md     what the trained models score, and what that means
reports/figures/       the EDA figures (not versioned, regenerated on demand)
src/                   the package
tests/                 the test suite
```

### Inside `src/`

| Module | Responsibility |
| --- | --- |
| `config.py` | The dataset schema, the paths, and the loader of `config.yaml`. |
| `prepare_data.py` | Reads the raw file, separates the target, splits train and test. |
| `preprocessing.py` | Row filtering, age parsing into days, and the cleaning transformer for the pipeline. |
| `feature_engineering.py` | The transformers that derive the model's features. |
| `pipeline.py` | Assembles cleaning, features, encoding, oversampling and classifier into one estimator. |
| `train.py` | Runs the tournament for one species and persists the winner. |
| `evaluate.py` | Scores the trained models on the test split. |
| `eda.py` | The exploratory analysis and its figures. |

## Two configuration files

`src/config.py` holds what describes the **data**: column names, feature
groups, the paths where the datasets live. Changing any of it is a code change,
so it sits next to the code.

`config.yaml` holds what describes a **run**: the split proportions, the number
of folds, the scoring metrics, the metric the winner is refit on, and the
hyperparameter search grids. Changing those is an analysis decision, so the
file can be swapped without a commit:

```bash
snakemake --cores all --configfile experiments/wide_grid.yaml
```

Adding a species is a one-line change to `config.SPECIES`: the Snakefile
derives its wildcards from it, so a third model would be trained and evaluated
without touching anything else.
