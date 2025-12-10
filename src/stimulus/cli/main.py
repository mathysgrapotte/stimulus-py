"""Main entry point for stimulus-py cli."""

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

from typing import Optional

import click

from stimulus.cli.check_model import check_model as check_model_func
from stimulus.cli.compare_tensors import compare_tensors_and_save
from stimulus.cli.encode import encode as encode_func
from stimulus.cli.predict import predict as predict_func
from stimulus.cli.split import split as split_func
from stimulus.cli.split_yaml import split_yaml as split_yaml_func
from stimulus.cli.transform import transform as transform_func
from stimulus.cli.tuning import tune as tune_func
from stimulus.data.interface.dataset_interface import H5adDataset, HuggingFaceDataset, StimulusDataset


def _get_dataset_cls(data_path: str) -> type[StimulusDataset]:
    """Get the appropriate dataset class based on file extension."""
    if data_path.endswith(".h5ad"):
        return H5adDataset
    return HuggingFaceDataset


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version("stimulus-template"), "-v", "--version")
def cli() -> None:
    """Stimulus is an open-science framework for data processing and model training."""


@cli.command()
@click.option(
    "-d",
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="Path to input data file",
)
@click.option(
    "-m",
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to model file",
)
@click.option(
    "-c",
    "--model-config",
    type=click.Path(exists=True),
    required=True,
    help="Path to yaml config training file",
)
@click.option(
    "-r",
    "--optuna-results-dirpath",
    type=click.Path(),
    default="./optuna_results",
    help="Location for optuna results output directory [default: ./optuna_results]",
)
@click.option(
    "-f",
    "--force-device",
    default=None,
    help="Force the use of a specific device. Example: --force-device cuda:0",
)
def check_model(
    data: str,
    model: str,
    model_config: str,
    optuna_results_dirpath: str,
    force_device: Optional[str] = None,
) -> None:
    """Check model configuration and run initial tests."""
    check_model_func(
        data_path=data,
        model_path=model,
        model_config_path=model_config,
        optuna_results_dirpath=optuna_results_dirpath,
        force_device=force_device,
        dataset_cls=_get_dataset_cls(data),
    )


@cli.command()
@click.option(
    "-d",
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="The file path for the data containing all data",
)
@click.option(
    "-y",
    "--yaml",
    type=click.Path(exists=True),
    required=True,
    help="The YAML config file that holds all parameter info",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="The output file path to write the split dataset",
)
def split(
    data: str,
    yaml: str,
    output: str,
) -> None:
    """Split rows in a data file."""
    split_func(
        data_path=data,
        config_yaml=yaml,
        out_path=output,
        dataset_cls=_get_dataset_cls(data),
    )


@cli.command()
@click.option(
    "-y",
    "--yaml",
    type=click.Path(exists=True),
    required=True,
    help="The YAML config file to split into component configs",
)
@click.option(
    "-d",
    "--out-dir",
    type=click.Path(),
    required=False,
    default="./",
    help="The output directory where component YAML files will be written [default: ./]",
)
def split_yaml(
    yaml: str,
    out_dir: str,
) -> None:
    """Split a YAML configuration into separate component configs.

    Creates individual files for encoding, splits, and transforms:
    - encode.yaml: Encoding configuration
    - split1.yaml, split2.yaml, etc.: Individual split configurations
    - transform1.yaml, transform2.yaml, etc.: Individual transform configurations
    """
    split_yaml_func(config_yaml=yaml, out_dir_path=out_dir)


@cli.command()
@click.option(
    "-d",
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="The file path for the data to transform",
)
@click.option(
    "-y",
    "--yaml",
    type=click.Path(exists=True),
    required=True,
    help="The YAML config file that holds transformation parameters",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="The output file path to write the transformed dataset",
)
def transform(
    data: str,
    yaml: str,
    output: str,
) -> None:
    """Transform data in a file according to configuration."""
    transform_func(
        data_path=data,
        config_yaml=yaml,
        out_path=output,
        dataset_cls=_get_dataset_cls(data),
    )


@cli.command()
@click.option(
    "-d",
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="Path to input data file or directory (Parquet or HuggingFace dataset)",
)
@click.option(
    "-y",
    "--yaml",
    type=click.Path(exists=True),
    required=True,
    help="The YAML config file that holds encoder parameters",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="The output directory path to save the encoded dataset",
)
@click.option(
    "-p",
    "--num-proc",
    type=int,
    default=None,
    help="Number of processes to use for encoding [default: None (disable multiprocessing)]",
)
def encode(
    data: str,
    yaml: str,
    output: str,
    num_proc: Optional[int] = None,
) -> None:
    """Encode data according to configuration."""
    encode_func(
        data_path=data,
        config_yaml=yaml,
        out_path=output,
        num_proc=num_proc,
        dataset_cls=_get_dataset_cls(data),
    )


@cli.command()
@click.option(
    "-d",
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="Path to input csv file",
)
@click.option(
    "-m",
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to model file",
)
@click.option(
    "-c",
    "--model-config",
    type=click.Path(exists=True),
    required=True,
    help="Path to yaml config training file",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="best_model.safetensors",
    help="Path to save the best model [default: best_model.safetensors]",
)
@click.option(
    "-bo",
    "--best-optimizer",
    type=click.Path(),
    default="best_optimizer.pt",
    help="Path to save the best optimizer [default: best_optimizer.pt]",
)
@click.option(
    "-r",
    "--optuna-results-dirpath",
    type=click.Path(),
    default="./optuna_results",
    help="Location for optuna results output directory",
)
@click.option(
    "-f",
    "--force-device",
    default=None,
    help="Force the use of a specific device. Example: --force-device cuda:0",
)
def tune(
    data: str,
    model: str,
    model_config: str,
    output: str,
    best_optimizer: str,
    optuna_results_dirpath: str,
    force_device: Optional[str] = None,
) -> None:
    """Run hyperparameter tuning for a model."""
    tune_func(
        data_path=data,
        model_path=model,
        model_config_path=model_config,
        optuna_results_dirpath=optuna_results_dirpath,
        best_model_path=output,
        best_optimizer_path=best_optimizer,
        force_device=force_device,
    )


@cli.command()
@click.option(
    "-d",
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="Path to input data file",
)
@click.option(
    "-m",
    "--model",
    type=click.Path(exists=True),
    required=True,
    help="Path to model file",
)
@click.option(
    "-c",
    "--model-config",
    type=click.Path(exists=True),
    required=True,
    help="Path to model config file",
)
@click.option(
    "-w",
    "--model-weight",
    type=click.Path(exists=True),
    required=True,
    help="Path to model weight file in safetensors format",
)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=256,
    help="Batch size for prediction [default: 256]",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="predictions.safetensors",
    help="Path to save the predictions [default: predictions.safetensors]",
)
def predict(
    data: str,
    model: str,
    model_config: str,
    model_weight: str,
    batch_size: int = 256,
    output: str = "predictions.safetensors",
) -> None:
    """Use model to predict on data."""
    predict_func(
        data_path=data,
        model_path=model,
        model_config_path=model_config,
        weight_path=model_weight,
        output=output,
        batch_size=batch_size,
        dataset_cls=_get_dataset_cls(data),
    )


@cli.command()
@click.argument("tensor_paths", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["cosine_similarity", "discrete_comparison"]),
    default="cosine_similarity",
    help="Similarity metric to use for comparison",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="comparison_results.csv",
    help="Path to save pairwise comparison results [default: comparison_results.csv]",
)
def compare_tensors(
    tensor_paths: tuple,
    mode: str,
    output: str,
) -> None:
    """Compare multiple tensor files with each other.

    Accepts an arbitrary number of tensor file paths and computes pairwise similarities.

    Example:
        stimulus compare-tensors tensor1.safetensors tensor2.safetensors tensor3.safetensors --mode cosine_similarity
    """
    if len(tensor_paths) < 2:  # noqa: PLR2004
        click.echo("Error: At least two tensor files are required for comparison.")
        return

    click.echo(f"Comparing {len(tensor_paths)} tensors using {mode}...")
    click.echo(f"Saving results to: {output}")

    compare_tensors_and_save(
        tensor_paths=list(tensor_paths),
        output_logs=output,
        mode=mode,
    )

    click.echo("Comparison complete!")
