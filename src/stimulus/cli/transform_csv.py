#!/usr/bin/env python3
"""CLI module for transforming CSV data files."""

import argparse

import yaml

from stimulus.data.data_handlers import DatasetProcessor, TransformManager
from stimulus.data.loaders import TransformLoader
from stimulus.utils.yaml_data import SplitConfigDict


def get_args() -> argparse.Namespace:
    """Get the arguments when using from the commandline."""
    parser = argparse.ArgumentParser(
        description="CLI for transforming CSV data files using YAML configuration."
    )
    parser.add_argument(
        "-c",
        "--csv",
        type=str,
        required=True,
        metavar="FILE",
        help="The file path for the csv containing all data",
    )
    parser.add_argument(
        "-y",
        "--yaml",
        type=str,
        required=True,
        metavar="FILE",
        help="The YAML config file that holds all parameter info",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        metavar="FILE",
        help="The output file path to write the noised csv",
    )

    return parser.parse_args()


def main(data_csv: str, config_yaml: str, out_path: str) -> None:
    """Connect CSV and YAML configuration and handle sanity checks.

    This launcher will be the connection between the csv and one YAML configuration.
    It should also handle some sanity checks.
    """
    # initialize the csv processing class, it open and reads the csv in automatic
    processor = DatasetProcessor(config_path=config_yaml, csv_path=data_csv)

    # initialize the transform manager
    transform_config = processor.dataset_manager.config.transforms
    with open(config_yaml) as f:
        yaml_config = SplitConfigDict(**yaml.safe_load(f))
    transform_loader = TransformLoader(seed=yaml_config.global_params.seed)
    print(transform_config)
    transform_loader.initialize_column_data_transformers_from_config(transform_config)
    transform_manager = TransformManager(transform_loader)

    # apply the transformations to the data
    processor.apply_transformation_group(transform_manager)

    # write the transformed data to a new csv
    processor.save(out_path)


def run() -> None:
    """Run the CSV transformation script."""
    args = get_args()
    main(args.csv, args.yaml, args.output)


if __name__ == "__main__":
    run()
