#!/usr/bin/env python3
"""CLI module for splitting YAML configuration files into unique files for each transform.

This module provides functionality to split a single YAML configuration file into multiple
YAML files, each containing a unique transform associated to a unique split.
The resulting YAML files can be used as input configurations for the stimulus package.
"""

import argparse
from typing import Any

import yaml

from stimulus.utils.yaml_data import (
    SplitConfigDict,
    SplitTransformDict,
    dump_yaml_list_into_files,
    generate_split_transform_configs,
)


def get_args() -> argparse.Namespace:
    """Get the arguments when using the command line."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-j",
        "--yaml",
        type=str,
        required=True,
        metavar="FILE",
        help="The YAML config file that hold all the transform per split parameter info",
    )
    parser.add_argument(
        "-d",
        "--out-dir",
        type=str,
        required=False,
        nargs="?",
        const="./",
        default="./",
        metavar="DIR",
        help="The output dir where all the YAMLs are written to. Output YAML will be called split_transform-#[number].yaml. Default -> ./",
    )

    return parser.parse_args()


def main(config_yaml: str, out_dir_path: str) -> None:
    """Reads a YAML config and generates files for all split - transform possible combinations.

    This script reads a YAML with a defined structure and creates all the YAML files ready to be passed to the stimulus package.

    The structure of the YAML is described here -> TODO: paste here the link to documentation
    This YAML and its structure summarize how to generate all the transform for the split and respective parameter combinations.

    This script will always generate at least one YAML file that represent the combination that does not touch the data (no transform).
    """
    # read the yaml experiment config and load its dictionnary
    yaml_config: dict[str, Any] = {}
    with open(config_yaml) as conf_file:
        yaml_config = yaml.safe_load(conf_file)

    yaml_config_dict: SplitConfigDict = SplitConfigDict(**yaml_config)

    # Generate the yaml files for each transform
    split_transform_configs: list[SplitTransformDict] = generate_split_transform_configs(yaml_config_dict)

    # Dump all the YAML configs into files
    dump_yaml_list_into_files(split_transform_configs, out_dir_path, "test_transforms")


if __name__ == "__main__":
    args = get_args()
    main(args.yaml, args.out_dir)
