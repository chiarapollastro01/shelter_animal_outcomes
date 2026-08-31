"""Pipeline for the Shelter Animal Outcomes project.

This Snakefile chains the three steps that produce the metrics report: the raw file is split
into a training and a test set, one model is trained per species, and the two
are evaluated together.

The species come from src/config.py, being part of the data schema, all the other parameters
are read from the config.yaml by the modules themselves.

Usage
-----
snakemake --cores all          # run everything that is out of date
snakemake --cores all -n       # show what would run, without running it
snakemake --cores all --forceall   # rebuild from scratch
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as project_config



SPECIES_BY_KEY = {species.lower(): species for species in project_config.SPECIES}

SPLIT_DIR = "data/split_data/"
RAW = "data/raw_data/train.csv"
RUN_CONFIG = "config.yaml"
TRAIN_FEATURES = SPLIT_DIR + project_config.TRAIN_FEATURES_FILE
TRAIN_TARGET = SPLIT_DIR + project_config.TRAIN_TARGET_FILE
TEST_FEATURES = SPLIT_DIR + project_config.TEST_FEATURES_FILE
TEST_TARGET = SPLIT_DIR + project_config.TEST_TARGET_FILE
METRICS = "reports/" + project_config.METRICS_FILE

MODEL_PATTERN = "models/" + project_config.MODEL_FILE_TEMPLATE
MODEL_METADATA_PATTERN = "models/" + project_config.MODEL_METADATA_FILE_TEMPLATE


# Without this, asking for a model of an undeclared species would start the rule
# and fail inside it with a KeyError; here it fails as an unmatched target.
wildcard_constraints:
    species="|".join(SPECIES_BY_KEY),


def species_name(wildcards):
    """Map the lowercase file-name wildcard back to the species as config spells it.

    Parameters
    ----------
    wildcards : snakemake.io.Wildcards
        The wildcards of the job, carrying the lowercase species key.

    Returns
    -------
    str
        The species as it appears in config.SPECIES, which is what the
        --species option of src.train accepts.
    """
    return SPECIES_BY_KEY[wildcards.species]


rule all:
    input:
        METRICS,


rule prepare_data:
    input:
        raw=RAW,
        cfg=RUN_CONFIG,
    output:
        train_features=TRAIN_FEATURES,
        train_target=TRAIN_TARGET,
        test_features=TEST_FEATURES,
        test_target=TEST_TARGET,
    shell:
        """
        python -m src.prepare_data {input.raw} \
            --output-dir data/split_data --config {input.cfg}
        """


rule train:
    input:
        features=TRAIN_FEATURES,
        target=TRAIN_TARGET,
        cfg=RUN_CONFIG,
    output:
        model=MODEL_PATTERN,
        metadata=MODEL_METADATA_PATTERN,
    threads: workflow.cores or 1
    params:
        name=species_name,
    shell:
        """
        python -m src.train {input.features} {input.target} \
            --models-dir models --config {input.cfg} --species {params.name}
        """


rule evaluate:
    input:
        features=TEST_FEATURES,
        target=TEST_TARGET,
        models=expand(MODEL_PATTERN, species=list(SPECIES_BY_KEY)),
    output:
        METRICS,
    shell:
        """
        python -m src.evaluate \
            --test-features {input.features} --test-target {input.target} \
            --models-dir models --output-metrics {output}
        """