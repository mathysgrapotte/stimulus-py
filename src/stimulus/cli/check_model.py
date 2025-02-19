#!/usr/bin/env python3
"""CLI module for checking model configuration and running initial tests."""

import argparse
import logging

import ray
import yaml
from torch.utils.data import DataLoader

from stimulus.data import handlertorch, loaders
from stimulus.learner import raytune_learner
from stimulus.utils import launch_utils, yaml_data, yaml_model_schema

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Get the arguments when using from the commandline.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Launch check_model.")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to input csv file.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to model file.",
    )
    parser.add_argument(
        "-e",
        "--data_config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to data config file.",
    )
    parser.add_argument(
        "-c",
        "--model_config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to yaml config training file.",
    )
    parser.add_argument(
        "-w",
        "--initial_weights",
        type=str,
        required=False,
        nargs="?",
        const=None,
        default=None,
        metavar="FILE",
        help="The path to the initial weights (optional).",
    )

    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        required=False,
        nargs="?",
        const=3,
        default=3,
        metavar="NUM_SAMPLES",
        help="Number of samples for tuning. Overwrites tune.tune_params.num_samples in config.",
    )
    parser.add_argument(
        "--ray_results_dirpath",
        type=str,
        required=False,
        nargs="?",
        const=None,
        default=None,
        metavar="DIR_PATH",
        help="Location where ray_results output dir should be written. If None, uses ~/ray_results.",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Activate debug mode for tuning. Default false, no debug.",
    )

    return parser.parse_args()


def main(
    model_path: str,
    data_path: str,
    data_config_path: str,
    model_config_path: str,
    initial_weights: str | None = None,  # noqa: ARG001
    num_samples: int = 3,
    ray_results_dirpath: str | None = None,
    *,
    debug_mode: bool = False,
) -> None:
    """Run the main model checking pipeline.

    Args:
        data_path: Path to input data file.
        model_path: Path to model file.
        data_config_path: Path to data config file.
        model_config_path: Path to model config file.
        initial_weights: Optional path to initial weights.
        num_samples: Number of samples for tuning.
        ray_results_dirpath: Directory for ray results.
        debug_mode: Whether to run in debug mode.
    """
    with open(data_config_path) as file:
        data_config = yaml.safe_load(file)
        data_config = yaml_data.SplitTransformDict(**data_config)

    with open(model_config_path) as file:
        model_config = yaml.safe_load(file)
        model_config = yaml_model_schema.Model(**model_config)

    encoder_loader = loaders.EncoderLoader()
    encoder_loader.initialize_column_encoders_from_config(
        column_config=data_config.columns
    )

    logger.info("Dataset loaded successfully.")

    model_class = launch_utils.import_class_from_file(model_path)

    logger.info("Model class loaded successfully.")

    ray_config_loader = yaml_model_schema.YamlRayConfigLoader(model=model_config)
    ray_config_dict = ray_config_loader.get_config().model_dump()
    ray_config_model = ray_config_loader.get_config()

    logger.info("Ray config loaded successfully.")

    sampled_model_params = {
        key: domain.sample() if hasattr(domain, "sample") else domain
        for key, domain in ray_config_dict["network_params"].items()
    }

    logger.info("Sampled model params loaded successfully.")

    model_instance = model_class(**sampled_model_params)

    logger.info("Model instance loaded successfully.")

    torch_dataset = handlertorch.TorchDataset(
        data_config=data_config,
        csv_path=data_path,
        encoder_loader=encoder_loader,
    )

    torch_dataloader = DataLoader(torch_dataset, batch_size=10, shuffle=True)

    logger.info("Torch dataloader loaded successfully.")

    # try to run the model on a single batch
    for batch in torch_dataloader:
        input_data, labels, metadata = batch
        # Log shapes of tensors in each dictionary
        for key, tensor in input_data.items():
            logger.debug(f"Input tensor '{key}' shape: {tensor.shape}")
        for key, tensor in labels.items():
            logger.debug(f"Label tensor '{key}' shape: {tensor.shape}")
        for key, list_object in metadata.items():
            logger.debug(f"Metadata lists '{key}' length: {len(list_object)}")
        output = model_instance(**input_data)
        logger.info("model ran successfully on a single batch")
        logger.debug(f"Output shape: {output.shape}")
        break

    logger.info("Model checking single pass completed successfully.")

    # override num_samples
    model_config.tune.tune_params.num_samples = num_samples

    tuner = raytune_learner.TuneWrapper(
        model_config=ray_config_model,
        data_config=data_config,
        model_class=model_class,
        data_path=data_path,
        encoder_loader=encoder_loader,
        seed=42,
        ray_results_dir=ray_results_dirpath,
        debug=debug_mode,
    )

    logger.info("Tuner initialized successfully.")

    tuner.tune()

    logger.info("Tuning completed successfully.")
    logger.info("Checks complete")


def run() -> None:
    """Run the model checking script."""
    ray.init(address="auto", ignore_reinit_error=True)
    args = get_args()
    main(
        data_path=args.data,
        model_path=args.model,
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        initial_weights=args.initial_weights,
        num_samples=args.num_samples,
        ray_results_dirpath=args.ray_results_dirpath,
        debug_mode=args.debug_mode,
    )


if __name__ == "__main__":
    run()
