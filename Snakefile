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

configfile: "config.yaml"


SPECIES_BY_KEY = {species.lower(): species for species in project_config.SPECIES}
MODEL_PATTERN = "models/" + project_config.MODEL_FILE_TEMPLATE
MODEL_METADATA_PATTERN = "models/" + project_config.MODEL_METADATA_FILE_TEMPLATE

rule all:
    input:
        "reports/metrics.json"


rule prepare_data:
    input:
        raw="data/raw_data/train.csv",
        params="config.yaml",
    output:
        train_features="data/split_data/train_features.csv",
        train_target="data/split_data/train_target.csv",
        test_features="data/split_data/test_features.csv",
        test_target="data/split_data/test_target.csv",
    shell:
        """
        python -m src.prepare_data {input.raw} \
            --output-dir data/split_data --config {input.params}
        """


rule train:
    input:
        features="data/split_data/train_features.csv",
        target="data/split_data/train_target.csv",
        params="config.yaml",
    output:
        model=MODEL_PATTERN,
        metadata=MODEL_METADATA_PATTERN,
    threads: workflow.cores or 1
    params:
        name=lambda wildcards: SPECIES_BY_KEY[wildcards.species],
    shell:
        """
        python -m src.train {input.features} {input.target} \
            --models-dir models --config {input.params} --species {params.name}
        """


rule evaluate:
    input:
        features="data/split_data/test_features.csv",
        target="data/split_data/test_target.csv",
        models=expand(
            "models/best_shelter_model_{species}.pkl",
            species=list(SPECIES_BY_KEY.keys()),
        ),
    output:
        "reports/metrics.json",
    shell:
        """
        python -m src.evaluate \
            --test-features {input.features} --test-target {input.target} \
            --models-dir models --output-metrics {output}
        """