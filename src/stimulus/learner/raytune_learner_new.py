"""Ray Tune wrapper and trainable model classes for hyperparameter optimization."""

import os
import ray
import logging
import datetime

from torch import nn
from typing import Optional

from src.stimulus.data.experiments import EncoderLoader
from src.stimulus.utils.generic_utils import (
    set_general_seeds,
    check_path,
    check_not_none,
)
from src.stimulus.utils.yaml_model_schema import YamlRayConfigLoader


class TuneWrapper:
    """
    Wrapper class for Ray Tune hyperparameter optimization

    Creates a dict:
        dict = {
            "model_params": {},
            "loss_params": {},
            "optimizer_params": {},
            "data_params": {},
            "tune": {
                "scheduler": {
                    "name": str,
                    "params": {},
               },
               run_params: {},
            }
            "seed": int,
            "model": nn.Module,
            "EncoderLoader": EncoderLoader,
            "ray_worker_seed": int,
            "data_path": str,
            "tune_run_path": str,


    Args:
        config_path (str): path to the configuration file
        model_class (nn.Module): A pytorch model.
        data_path (str): path to the data to use.
        encode_loader (EncodeLoader): An EncoderLoader object use by TorchDataset
        ray_result_dir (:obj: str, optional): Directory to store the results.
            Defaults to None
        tune_run_name (:obj: str, optional): Name to give to the ray tune runs.
            Defaults to None
    """

    def __init__(
        self,
        config_path: str,
        model_class: nn.Module,
        data_path: str,
        encode_loader: EncoderLoader,
        max_gpus: int,  # CHANGED THIS TO AN OBLIGATORY PARAMETER
        max_cpus: int,  # CHANGED THIS TO AN OBLIGATORY PARAMETER
        max_mem: int,  # CHANGED THIS TO AN OBLIGATORY PARAMETER
        max_object_store_mem: Optional[float] = None,
        ray_result_dir: Optional[str] = None,
        tune_run_name: Optional[str] = None,
    ) -> None:
        # Load the configuration
        self.config = YamlRayConfigLoader(config_path).get_config()

        # Set all general seeds: python, numpy and pytorch
        set_general_seeds(self.config["seed"])

        self.config["model"]: nn.Module = model_class
        self.config["EncoderLoader"]: EncoderLoader = encode_loader
        # add the ray method for number generation to the config so it can be passed to the trainable class, that will in turn set per worker seeds in a reproducible mnanner.
        self.config["ray_worker_seed"] = ray.tune.randint(0, 1000)
        self.config["data_path"]: str = check_path(data_path)

        # Set the tune run name and dir
        if tune_run_name is None:
            tune_run_name = "TuneModel_" + datetime.datetime.now(
                tz=datetime.timezone.utc
            ).strftime("%Y-%m-%d-%H-%M-%S")
        if ray_result_dir is None:
            ray_result_dir = os.environ.get(
                "HOME"
            )  # If none ray puts it under home so we do to
        self.config["tune_run_path"]: str = os.path.join(ray_result_dir, tune_run_name)

        # Create the tune configuration
        scheduler_params: dict = self.config["tune"]["scheduler"]
        scheduler: ray.tune.Scheduler = getattr(
            ray.tune.schedulers, scheduler_params["name"]
        )(**scheduler_params["params"])
        self.tune_config: ray.tune.TuneConfig = ray.tune.TuneConfig(
            **self.config, scheduler=scheduler
        )

        # Set the hardware ressources
        # TODO: if there's  a check for these params, check it here during init
        self.max_gpus: int = check_not_none(max_gpus, "max_gpus")
        self.max_cpus: int = check_not_none(max_cpus, "max_cpus")
        self.max_object_store_mem: int = max_object_store_mem  # this is a special subset of the total usable memory that ray need for his internal work, by default is set to 30% of total memory usable
        self.max_mem: int = max_mem

        # TODO: implement checkpoiting
        self.checkpoint_config: ray.train.CheckpointConfig = ray.train.CheckpointConfig(
            checkpoint_at_end=True
        )
        self.run_config: ray.train.RunConfig(
            name=tune_run_name,
            storage_path=ray_result_dir,
            checkpoint_config=self.checkpoint_config,
            **self.config["tune"]["run_params"],
        )
        self.tuner = self.tuner_initialization()

    def tuner_initiazilation(self) -> ray.tune.Tuner:
        """Prepare the tuner with the configs."""
        # in ray 3.0.0 the following issue is fixed. Sometimes it sees that ray is already initialized, so in that case shut it off and start anew. TODO update to ray 3.0.0
        if ray.is_initialized():
            ray.shutdown()

        ray.init(
            num_cpus=self.max_cpus,
            num_gpus=self.max_gpus,
            object_store_memory=self.max_object_store_mem,
            _memory=self.max_mem,
        )

        cluster_ressources: dict = ray.cluster_ressources()

        logging.info(f"CLUSTER ressources\t->\t{cluster_ressources}")

        self.gpu_per_trial = self._set_per_trial_ressources(cluster_ressources, "gpu")

    def _set_per_trial_ressources(self, ):
