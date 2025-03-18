"""Main entry point for stimulus-py cli."""

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

import click


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version("stimulus-py"), "-v", "--version")
def cli() -> None:
    """Stimulus is an open-science framework for data processing and model training."""


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
    "-e",
    "--data-config",
    type=click.Path(exists=True),
    required=True,
    help="Path to data config file",
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
def check_model(
    data: str,
    model: str,
    data_config: str,
    model_config: str,
    optuna_results_dirpath: str,
) -> None:
    """Check model configuration and run initial tests."""
    from stimulus.cli.check_model import check_model as check_model_func

    check_model_func(
        data_path=data,
        model_path=model,
        data_config_path=data_config,
        model_config_path=model_config,
        optuna_results_dirpath=optuna_results_dirpath,
    )


@cli.command()
@click.option(
    "-y",
    "--yaml",
    type=click.Path(exists=True),
    required=True,
    help="The YAML config file that hold all transform - split - parameter info",
)
@click.option(
    "-d",
    "--out-dir",
    type=click.Path(),
    required=False,
    default="./",
    help="The output dir where all the YAMLs are written to. Output YAML will be called split-#[number].yaml transform-#[number].yaml. Default -> ./",
)
def split_split(
    yaml: str,
    out_dir: str,
) -> None:
    """Split a YAML configuration file into multiple YAML files, each containing a unique split."""
    from stimulus.cli.split_split import split_split as split_split_func

    split_split_func(config_yaml=yaml, out_dir_path=out_dir)


@cli.command()
@click.option(
    "-c",
    "--csv",
    type=click.Path(exists=True),
    required=True,
    help="The file path for the csv containing the data in csv format",
)
@click.option(
    "-y",
    "--yaml",
    type=click.Path(exists=True),
    required=True,
    help="The YAML data config",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="The output file path to write the shuffled csv",
)
def shuffle_csv(
    csv: str,
    yaml: str,
    output: str,
) -> None:
    """Shuffle rows in a CSV data file."""
    from stimulus.cli.shuffle_csv import shuffle_csv as shuffle_csv_func

    shuffle_csv_func(
        data_csv=csv,
        config_yaml=yaml,
        out_path=output,
    )


@cli.command()
@click.option(
    "-c",
    "--csv",
    type=click.Path(exists=True),
    required=True,
    help="The file path for the csv containing all data",
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
    help="The output file path to write the split csv",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite the split column if it already exists in the csv",
)
def split_csv(
    csv: str,
    yaml: str,
    output: str,
    *,
    force: bool,
) -> None:
    """Split rows in a CSV data file."""
    from stimulus.cli.split_csv import split_csv as split_csv_func

    split_csv_func(
        data_csv=csv,
        config_yaml=yaml,
        out_path=output,
        force=force,
    )


@cli.command()
@click.option(
    "-y",
    "--yaml",
    type=click.Path(exists=True),
    required=True,
    help="The YAML config file that hold all the transform per split parameter info",
)
@click.option(
    "-d",
    "--out-dir",
    type=click.Path(),
    required=False,
    default="./",
    help="The output dir where all the YAMLs are written to. Output YAML will be called split_transform-#[number].yaml. Default -> ./",
)
def split_transforms(
    yaml: str,
    out_dir: str,
) -> None:
    """Split a YAML configuration file into multiple YAML files, each containing a unique transform."""
    from stimulus.cli.split_transforms import split_transforms as split_transforms_func

    split_transforms_func(config_yaml=yaml, out_dir_path=out_dir)


@cli.command()
@click.option(
    "-c",
    "--csv",
    type=click.Path(exists=True),
    required=True,
    help="The file path for the csv containing the data to transform",
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
    help="The output file path to write the transformed csv",
)
def transform_csv(
    csv: str,
    yaml: str,
    output: str,
) -> None:
    """Transform data in a CSV file according to configuration."""
    from stimulus.cli.transform_csv import main as transform_csv_func

    transform_csv_func(
        data_csv=csv,
        config_yaml=yaml,
        out_path=output,
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
    "-e",
    "--data-config",
    type=click.Path(exists=True),
    required=True,
    help="Path to data config file",
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
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose Optuna logging output",
)
def tune(
    data: str,
    model: str,
    data_config: str,
    model_config: str,
    output: str,
    best_optimizer: str,
    optuna_results_dirpath: str,
    *,
    verbose: bool,
) -> None:
    """Run hyperparameter tuning for a model."""
    from stimulus.cli.tuning import tune as tune_func

    tune_func(
        data_path=data,
        model_path=model,
        data_config_path=data_config,
        model_config_path=model_config,
        optuna_results_dirpath=optuna_results_dirpath,
        best_model_path=output,
        best_optimizer_path=best_optimizer,
        verbose=verbose,
    )
