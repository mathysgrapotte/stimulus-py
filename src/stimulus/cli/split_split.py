#!/usr/bin/env python3
"""CLI module for splitting YAML configuration files into unique files for each split.

This module provides functionality to split a single YAML configuration file into multiple
YAML files, each containing a unique split.
The resulting YAML files can be used as input configurations for the stimulus package.
"""

import argparse
from typing import Any

import yaml

from stimulus.utils.yaml_data import (
    ConfigDict,
    SplitConfigDict,
    check_yaml_schema,
    dump_yaml_list_into_files,
    generate_split_configs,
)


def get_args() -> argparse.Namespace:
    """Get the arguments when using from the command line."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-y",
        "--yaml",
        type=str,
        required=True,
        metavar="FILE",
        help="The YAML config file that hold all transform - split - parameter info",
    )
    parser.add_argument(
        "-d",
        "--out_dir",
        type=str,
        required=False,
        nargs="?",
        const="./",
        default="./",
        metavar="DIR",
        # TODO: Change the output name
        help="The output dir where all the YAMLs are written to. Output YAML will be called split-#[number].yaml transform-#[number].yaml. Default -> ./",
    )

    return parser.parse_args()


def main(config_yaml: str, out_dir_path: str) -> None:
    """Reads a YAML config file and generates a file per unique split.

    This script reads a YAML with a defined structure and creates all the YAML files ready to be passed to
    the stimulus package.

    The structure of the YAML is described here -> TODO paste here link to documentation.
    This YAML and its structure summarize how to generate unique splits and all the transformations associated to this split.

    This script will always generate at least one YAML file that represent the combination that does not touch the data (no transform)
    and uses the default split behavior.
    """
    # read the yaml experiment config and load its to dictionary
    yaml_config: dict[str, Any] = {}
    with open(config_yaml) as conf_file:
        yaml_config = yaml.safe_load(conf_file)

    yaml_config_dict: ConfigDict = ConfigDict(**yaml_config)
    # check if the yaml schema is correct
    # FIXME: isn't it redundant to check and already class with pydantic ?
    check_yaml_schema(yaml_config_dict)

    # generate the yaml files per split
    split_configs: list[SplitConfigDict] = generate_split_configs(yaml_config_dict)

    # dump all the YAML configs into files
    dump_yaml_list_into_files(split_configs, out_dir_path, "test_split")


def run() -> None:
    """Run the split_yaml CLI."""
    args = get_args()
    main(args.yaml, args.out_dir)


if __name__ == "__main__":
    run()
