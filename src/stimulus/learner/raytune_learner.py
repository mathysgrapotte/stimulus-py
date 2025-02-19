"""Ray Tune wrapper and trainable model classes for hyperparameter optimization."""

from stimulus.utils.yaml_model_schema import RayTuneModel
from stimulus.utils.yaml_data import SplitTransformDict
import datetime
import logging
import os
import random
from typing import Any, Optional, TypedDict

import numpy as np
import ray
import torch
from ray import cluster_resources, train, tune
from ray.tune import Trainable
from safetensors.torch import load_model as safe_load_model
from safetensors.torch import save_model as safe_save_model
from torch import nn, optim
from torch.utils.data import DataLoader

from stimulus.data.handlertorch import TorchDataset
from stimulus.data.loaders import EncoderLoader
from stimulus.learner.predict import PredictWrapper
from stimulus.utils.generic_utils import set_general_seeds


class CheckpointDict(TypedDict):
    """Dictionary type for checkpoint data."""

    checkpoint_dir: str


class TuneWrapper:
    """Wrapper class for Ray Tune hyperparameter optimization."""

    def __init__(
        self,
        model_config: RayTuneModel,
        data_config: SplitTransformDict,
        model_class: nn.Module,
        data_path: str,
        encoder_loader: EncoderLoader,
        seed: int,
        ray_results_dir: Optional[str] = None,
        tune_run_name: Optional[str] = None,
        *,
        debug: bool = False,
        autoscaler: bool = False,
    ) -> None:
        """Initialize the TuneWrapper with the paths to the config, model, and data."""
        self.config = model_config.model_dump()

        # set all general seeds: python, numpy and torch.
        set_general_seeds(seed)

        # build the tune config:
        try:
            scheduler_class = getattr(
                tune.schedulers,
                model_config.tune.scheduler.name,
            )  # todo, do this in RayConfigLoader
        except AttributeError as err:
            raise ValueError(
                f"Invalid scheduler: {model_config.tune.scheduler.name}, check Ray Tune for documentation on available schedulers",
            ) from err

        scheduler = scheduler_class(**model_config.tune.scheduler.params)
        self.tune_config = tune.TuneConfig(
            metric=model_config.tune.tune_params.metric,
            mode=model_config.tune.tune_params.mode,
            num_samples=model_config.tune.tune_params.num_samples,
            scheduler=scheduler,
        )

        # build the run config
        self.run_config = train.RunConfig(
            name=tune_run_name
            if tune_run_name is not None
            else "TuneModel_"
            + datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                "%Y-%m-%d_%H-%M-%S",
            ),
            storage_path=ray_results_dir,
            checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True),
            stop=model_config.tune.run_params.stop,
        )

        # add the data path to the config
        if not os.path.exists(data_path):
            raise ValueError("Data path does not exist. Given path:" + data_path)
        self.config["data_path"] = os.path.abspath(data_path)

        # Set up tune_run path
        if ray_results_dir is None:
            ray_results_dir = os.environ.get("HOME", "")
        self.config["tune_run_path"] = os.path.join(
            ray_results_dir,
            tune_run_name
            if tune_run_name is not None
            else "TuneModel_"
            + datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                "%Y-%m-%d_%H-%M-%S",
            ),
        )
        self.config["_debug"] = debug
        self.config["model"] = model_class
        self.config["encoder_loader"] = encoder_loader
        self.config["ray_worker_seed"] = tune.randint(0, 1000)

        self.gpu_per_trial = model_config.tune.gpu_per_trial
        self.cpu_per_trial = model_config.tune.cpu_per_trial

        self.tuner = self.tuner_initialization(
            data_config=data_config,
            data_path=data_path,
            encoder_loader=encoder_loader,
            autoscaler=autoscaler,
        )

    def tuner_initialization(
        self,
        data_config: SplitTransformDict,
        data_path: str,
        encoder_loader: EncoderLoader,
        *,
        autoscaler: bool = False,
    ) -> tune.Tuner:
        """Prepare the tuner with the configs."""
        # Get available resources from Ray cluster
        cluster_res = cluster_resources()
        logging.info(f"CLUSTER resources   ->  {cluster_res}")

        # Check per-trial resources
        try:
            if self.gpu_per_trial > cluster_res["GPU"] and not autoscaler:
                raise ValueError(
                    "GPU per trial is more than what is available in the cluster, set autoscaler to True to allow for autoscaler to be used.",
                )
        except KeyError as err:
            logging.warning(
                f"KeyError: {err}, no GPU resources available in the cluster: {cluster_res}",
            )

        if self.cpu_per_trial > cluster_res["CPU"] and not autoscaler:
            raise ValueError(
                "CPU per trial is more than what is available in the cluster, set autoscaler to True to allow for autoscaler to be used.",
            )

        logging.info(
            f"PER_TRIAL resources ->  GPU: {self.gpu_per_trial} CPU: {self.cpu_per_trial}",
        )

        # Pre-load and encode datasets once, then put them in Ray's object store

        training = TorchDataset(
            data_config=data_config,
            csv_path=data_path,
            encoder_loader=encoder_loader,
            split=0,
        )
        validation = TorchDataset(
            data_config=data_config,
            csv_path=data_path,
            encoder_loader=encoder_loader,
            split=1,
        )

        # log to debug the names of the columns and shapes of tensors for a batch of training
        # Log shapes of encoded tensors for first batch of training data
        inputs, labels, meta = training[0:10]

        logging.debug("Training data tensor shapes:")
        for field, tensor in inputs.items():
            logging.debug(f"Input field '{field}' shape: {tensor.shape}")

        for field, tensor in labels.items():
            logging.debug(f"Label field '{field}' shape: {tensor.shape}")

        for field, values in meta.items():
            logging.debug(f"Meta field '{field}' length: {len(values)}")

        training_ref = ray.put(training)
        validation_ref = ray.put(validation)

        self.config["_training_ref"] = training_ref
        self.config["_validation_ref"] = validation_ref

        # Configure trainable with resources and dataset parameters
        trainable = tune.with_resources(
            tune.with_parameters(
                TuneModel,
            ),
            resources={"cpu": self.cpu_per_trial, "gpu": self.gpu_per_trial},
        )

        return tune.Tuner(
            trainable,
            tune_config=self.tune_config,
            param_space=self.config,
            run_config=self.run_config,
        )

    def tune(self) -> ray.tune.ResultGrid:
        """Run the tuning process."""
        return self.tuner.fit()


class TuneModel(Trainable):
    """Trainable model class for Ray Tune."""

    def setup(self, config: dict[Any, Any]) -> None:
        """Get the model, loss function(s), optimizer, train and test data from the config."""
        # set the seeds the second time, first in TuneWrapper initialization
        set_general_seeds(self.config["ray_worker_seed"])

        # Initialize model with the config params
        self.model = config["model"](**config["network_params"])

        # Get the loss function(s) from the config model params
        self.loss_dict = config["loss_params"]
        for key, loss_fn in self.loss_dict.items():
            try:
                self.loss_dict[key] = getattr(nn, loss_fn)()
            except AttributeError as err:
                raise ValueError(
                    f"Invalid loss function: {loss_fn}, check PyTorch for documentation on available loss functions",
                ) from err

        # get the optimizer parameters
        optimizer_lr = config["optimizer_params"]["lr"]
        self.optimizer = getattr(optim, config["optimizer_params"]["method"])(
            self.model.parameters(),
            lr=optimizer_lr,
        )

        # get step size from the config
        self.step_size = config["tune"]["step_size"]

        # Get datasets from Ray's object store
        training, validation = (
            ray.get(self.config["_training_ref"]),
            ray.get(self.config["_validation_ref"]),
        )

        # use dataloader on training/validation data
        self.batch_size = config["data_params"]["batch_size"]
        self.training = DataLoader(
            training,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.validation = DataLoader(
            validation,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # debug section, first create a dedicated directory for each worker inside Ray_results/<tune_model_run_specific_dir> location
        debug_dir = os.path.join(
            config["tune_run_path"],
            "debug",
            ("worker_with_seed_" + str(self.config["ray_worker_seed"])),
        )
        if config["_debug"]:
            # creating a special directory for it one that is worker/trial/experiment specific
            os.makedirs(debug_dir)
            seed_filename = os.path.join(debug_dir, "seeds.txt")

            # save the initialized model weights
            self.export_model(export_dir=debug_dir)

            # save the seeds
            with open(seed_filename, "a") as seed_f:
                # you can not retrieve the actual seed once it set, or the current seed neither for python, numpy nor torch. so we select five numbers randomly. If that is the first draw of numbers they are always the same.
                python_values = random.sample(range(100), 5)
                numpy_values = list(np.random.randint(0, 100, size=5))
                torch_values = torch.randint(0, 100, (5,)).tolist()
                seed_f.write(
                    f"python drawn numbers : {python_values}\nnumpy drawn numbers : {numpy_values}\ntorch drawn numbers : {torch_values}\n",
                )

    def step(self) -> dict:
        """For each batch in the training data, calculate the loss and update the model parameters.

        This calculation is performed based on the model's batch function.
        At the end, return the objective metric(s) for the tuning process.
        """
        for _step_size in range(self.step_size):
            for x, y, _meta in self.training:
                # the loss dict could be unpacked with ** and the function declaration handle it differently like **kwargs. to be decided, personally find this more clean and understable.
                self.model.batch(x=x, y=y, optimizer=self.optimizer, **self.loss_dict)
        return self.objective()

    def objective(self) -> dict[str, float]:
        """Compute the objective metric(s) for the tuning process."""
        metrics = [
            "loss",
            "rocauc",
            "prauc",
            "mcc",
            "f1score",
            "precision",
            "recall",
            "spearmanr",
        ]  # TODO maybe we report only a subset of metrics, given certain criteria (eg. if classification or regression)
        predict_val = PredictWrapper(
            self.model,
            self.validation,
            loss_dict=self.loss_dict,
        )
        predict_train = PredictWrapper(
            self.model,
            self.training,
            loss_dict=self.loss_dict,
        )
        return {
            **{
                "val_" + metric: value
                for metric, value in predict_val.compute_metrics(metrics).items()
            },
            **{
                "train_" + metric: value
                for metric, value in predict_train.compute_metrics(metrics).items()
            },
        }

    # type: ignore[override]
    def export_model(self, export_dir: str | None = None) -> None:
        """Export model to safetensors format."""
        if export_dir is None:
            return
        safe_save_model(self.model, os.path.join(export_dir, "model.safetensors"))

    def load_checkpoint(self, checkpoint: dict[Any, Any] | None) -> None:
        """Load model and optimizer state from checkpoint."""
        if checkpoint is None:
            return
        checkpoint_dir = checkpoint["checkpoint_dir"]
        self.model = safe_load_model(
            self.model,
            os.path.join(checkpoint_dir, "model.safetensors"),
        )
        self.optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "optimizer.pt")),
        )

    def save_checkpoint(self, checkpoint_dir: str) -> dict[Any, Any]:
        """Save model and optimizer state to checkpoint."""
        safe_save_model(self.model, os.path.join(checkpoint_dir, "model.safetensors"))
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(checkpoint_dir, "optimizer.pt"),
        )
        return {"checkpoint_dir": checkpoint_dir}
