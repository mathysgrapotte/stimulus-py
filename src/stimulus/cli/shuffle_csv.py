#!/usr/bin/env python3
"""CLI module for shuffling CSV data files."""

import argparse
import json
import os

from stimulus.data.data_handlers import CsvProcessing
from stimulus.utils.launch_utils import get_experiment


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
        "-j",
        "--json",
        type=str,
        required=True,
        metavar="FILE",
        help="The json config file that hold all parameter info",
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


def main(data_csv: str, config_json: str, out_path: str) -> None:
    """Shuffle the data and split it according to the default split method.

    Args:
        data_csv: Path to input CSV file.
        config_json: Path to config JSON file.
        out_path: Path to output shuffled CSV.

    TODO major changes when this is going to select a given shuffle method and integration with split.
    """
    # open and read Json, just to extract the experiment name, so all other fields are scratched
    config = None
    with open(config_json) as in_json:
        tmp = json.load(in_json)
        config = tmp
        # add fake transform informations
        config["transform"] = "shuffle (special case)"

    # write the config modified, this will be associated to the shuffled data. TODO better solution to renaming like this
    modified_json = os.path.splitext(os.path.basename(data_csv))[0].split("-split")[0] + "-shuffled-experiment.json"
    with open(modified_json, "w") as out_json:
        json.dump(config, out_json)

    # initialize the experiment class
    exp_obj = get_experiment(config["experiment"])

    # initialize the csv processing class, it open and reads the csv in automatic
    csv_obj = CsvProcessing(exp_obj, data_csv)

    # shuffle the data with a default seed. TODO get the seed for the config if and when that is going to be set there.
    csv_obj.shuffle_labels(seed=42)

    # save the modified csv
    csv_obj.save(out_path)


def run() -> None:
    """Run the CSV shuffling script."""
    args = get_args()
    main(args.csv, args.json, args.output)


if __name__ == "__main__":
    run()
