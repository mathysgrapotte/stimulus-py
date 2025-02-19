"""This file provides the class API for handling the data in pytorch using the Dataset and Dataloader classes."""

from typing import Optional

from torch.utils.data import Dataset

from stimulus.data import data_handlers, loaders
from stimulus.utils.yaml_data import SplitTransformDict


class TorchDataset(Dataset):
    """Class for creating a torch dataset."""

    def __init__(
        self,
        data_config: SplitTransformDict,
        csv_path: str,
        encoder_loader: loaders.EncoderLoader,
        split: Optional[int] = None,
    ) -> None:
        """Initialize the TorchDataset.

        Args:
            data_config: A SplitTransformDict holding the configuration.
            csv_path: Path to the CSV data file
            encoder_loader: Encoder loader instance
            split: Optional tuple containing split information
        """
        self.loader = data_handlers.DatasetLoader(
            data_config=data_config,
            csv_path=csv_path,
            encoder_loader=encoder_loader,
            split=split,
        )

    def __len__(self) -> int:
        return len(self.loader)

    def __getitem__(self, idx: int) -> tuple[dict, dict, dict]:
        return self.loader[idx]
