#!/usr/bin/env python3
"""CLI module for shuffling CSV data files."""

import argparse

import yaml

from stimulus.data.data_handlers import DatasetProcessor
from stimulus.utils.yaml_data import SplitTransformDict


def get_args() -> argparse.Namespace:
    """Get the arguments when using from the commandline.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Shuffle rows in a CSV data file.")
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
        help="The YAML config file that hold all parameter info",
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
    """Shuffle the data and split it according to the default split method.

    Args:
        data_csv: Path to input CSV file.
        config_yaml: Path to config YAML file.
        out_path: Path to output shuffled CSV.

    TODO major changes when this is going to select a given shuffle method and integration with split.
    """
    # read the yaml file
    with open(config_yaml) as f:
        data_config: SplitTransformDict = SplitTransformDict(
            **yaml.safe_load(f),
        )
    # create a DatasetProcessor object from the config and the csv
    processor = DatasetProcessor(data_config=data_config, csv_path=data_csv)

    # shuffle the data with a default seed. TODO get the seed for the config if and when that is going to be set there.
    processor.shuffle_labels(seed=42)

    # save the modified csv
    processor.save(out_path)


def run() -> None:
    """Run the CSV shuffling script."""
    args = get_args()
    main(args.csv, args.yaml, args.output)


if __name__ == "__main__":
    run()
