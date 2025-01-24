#!/usr/bin/env python3
"""CLI module for transforming CSV data files."""

import argparse
import json

from stimulus.data.data_handlers import CsvProcessing
from stimulus.utils.launch_utils import get_experiment


def get_args() -> argparse.Namespace:
    """Get the arguments when using from the commandline."""
    parser = argparse.ArgumentParser(description="")
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
    """Connect CSV and JSON configuration and handle sanity checks.

    This launcher will be the connection between the csv and one json configuration.
    It should also handle some sanity checks.
    """
    # open and read Json
    config = {}
    with open(config_json) as in_json:
        config = json.load(in_json)

    # initialize the experiment class
    exp_obj = get_experiment(config["experiment"])

    # initialize the csv processing class, it open and reads the csv in automatic
    csv_obj = CsvProcessing(exp_obj, data_csv)

    # Transform the data according to what defined in the experiment class and the specifics of the user in the Json
    # in case of no transformation specification so when the config has "augmentation" : None  just save a copy of the original csv file
    if config.get("transform") is not None:
        csv_obj.transform(config["transform"])

    # save the modified csv
    csv_obj.save(out_path)


def run() -> None:
    """Run the CSV transformation script."""
    args = get_args()
    main(args.csv, args.json, args.output)


if __name__ == "__main__":
    run()
