#!/usr/bin/env python3
"""CLI module for splitting CSV data files."""

import argparse
import json
import logging

from stimulus.data.data_handlers import CsvProcessing
from stimulus.utils.launch_utils import get_experiment


def get_args() -> argparse.Namespace:
    """Get the arguments when using from the commandline."""
    parser = argparse.ArgumentParser(description="Split a CSV data file.")
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

    Args:
        data_csv: Path to input CSV file.
        config_json: Path to config JSON file.
        out_path: Path to output split CSV.

    TODO what happens when the user write his own experiment class? how should he do it ? how does it integrates here?
    """
    # open and read Json
    config = {}
    with open(config_json) as in_json:
        config = json.load(in_json)

    # initialize the experiment class
    exp_obj = get_experiment(config["experiment"])

    # initialize the csv processing class, it open and reads the csv in automatic
    csv_obj = CsvProcessing(exp_obj, data_csv)

    # CASE 1: SPLIT in csv, not in json --> keep the split from the csv
    if "split" in csv_obj.check_and_get_categories() and config["split"] is None:
        pass

    # CASE 2: SPLIT in csv and in json --> use the split from the json
    # TODO change this behaviour to do both, maybe
    elif "split" in csv_obj.check_and_get_categories() and config["split"]:
        logging.info("SPLIT present in both csv and json --> use the split from the json")
        csv_obj.add_split(config["split"], force=True)

    # CASE 3: SPLIT nor in csv and or json --> use the default RandomSplitter
    elif "split" not in csv_obj.check_and_get_categories() and config["split"] is None:
        # In case no split is provided, we use the default RandomSplitter
        logging.warning("SPLIT nor in csv and or json --> use the default RandomSplitter")
        # if the user config is None then set to default splitter -> RandomSplitter.
        config_default = {"name": "RandomSplitter", "params": {}}
        csv_obj.add_split(config_default)

    # CASE 4: SPLIT in json, not in csv --> use the split from the json
    else:
        csv_obj.add_split(config["split"], force=True)

    # save the modified csv
    csv_obj.save(out_path)


def run() -> None:
    """Run the CSV splitting script."""
    args = get_args()
    main(args.csv, args.json, args.output)


if __name__ == "__main__":
    run()
