"""Interface for dataset wrappers."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import datasets
import numpy as np
import torch


class StimulusDataset(ABC):
    """Abstract base class for Stimulus datasets.

    This class defines the minimal interface required for the Stimulus codebase.
    It abstracts away the underlying data storage and format (e.g. Hugging Face datasets,
    AnnData, custom formats).

    Users wishing to integrate a new dataset type should subclass this and implement
    all abstract methods.
    """

    @property
    @abstractmethod
    def split_names(self) -> list[str]:
        """Get the names of available splits (e.g. 'train', 'test', 'validation').

        Returns:
            list[str]: A list of strings representing the split names present in the dataset.
        """

    @property
    @abstractmethod
    def column_names(self) -> dict[str, list[str]]:
        """Get column names for each split.

        Returns:
            dict[str, list[str]]: A dictionary mapping split names to lists of column names.
                                  Example: {"train": ["text", "label"], "test": ["text", "label"]}
        """

    @abstractmethod
    def get_column(self, split: str, column_name: str) -> Union[list[Any], np.ndarray]:
        """Get a column from a specific split.

        This method is primarily used for inspecting data values, for example during
        splitting logic (e.g. stratified split based on labels).

        Args:
            split (str): The name of the split to access.
            column_name (str): The name of the column to retrieve.

        Returns:
            list[Any]: A list-like object containing the values of the column.
                       Should support indexing and iteration.
        """

    @abstractmethod
    def get_torch_dataset(self, split: Union[str, list[str]]) -> torch.utils.data.Dataset:
        """Get a PyTorch Dataset for training or inference.

        This method should return a standard PyTorch Dataset that yields samples
        compatible with the model's expected input.

        Args:
            split (Union[str, list[str]]): The split name(s) to retrieve.
                                           If a list is provided, the returned dataset
                                           should be the concatenation of those splits.

        Returns:
            torch.utils.data.Dataset: A PyTorch Dataset instance.
        """

    @abstractmethod
    def map(
        self,
        function: Callable,
        *,
        batched: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        remove_columns: Optional[list[str]] = None,
    ) -> "StimulusDataset":
        """Apply a transformation to all splits in the dataset.

        This method corresponds to element-wise or batch-wise processing.

        Args:
            function (Callable): The function to apply.
            batched (bool): Whether to apply the function to batches of data.
            fn_kwargs (Optional[dict]): Keyword arguments to pass to the function.
            num_proc (Optional[int]): Number of processes to use for parallel execution.
            remove_columns (Optional[list[str]]): List of columns to remove after transformation.

        Returns:
            StimulusDataset: A new dataset instance with the transformation applied.
        """

    @abstractmethod
    def apply(self, transformation: Callable) -> "StimulusDataset":
        """Apply a transformation to the dataset.

        The transformation can operate at element level or dataset level.
        If the transformation has a `scope` attribute set to 'dataset',
        it will be applied to the entire dataset at once (e.g. global normalization).
        Otherwise, it is treated as an element-wise transformation (delegated to `map`).

        Args:
            transformation (Callable): The transformation object or function.

        Returns:
            StimulusDataset: A new dataset instance with the transformation applied.
        """

    @abstractmethod
    def filter(self, function: Callable, *, batched: bool = False, **kwargs: Any) -> "StimulusDataset":
        """Filter all splits in the dataset.

        Args:
            function (Callable): A function that returns True for samples to keep.
            batched (bool): Whether to apply the function to batches.
            **kwargs: Additional arguments passed to the underlying implementation.

        Returns:
            StimulusDataset: A new filtered dataset instance.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the dataset to disk.

        Args:
            path (str): The directory path where the dataset should be saved.
        """

    @classmethod
    @abstractmethod
    def load_from_disk(cls, path: str, **kwargs: Any) -> "StimulusDataset":
        """Load a dataset from disk.

        Args:
            path (str): The path to the dataset.
            **kwargs: Additional arguments passed to the underlying dataset loader.
                      Common arguments include:
                      - keep_in_memory (bool): Whether to copy the dataset to memory.
                      - storage_options (dict): Options for the storage backend (e.g. S3).

        Returns:
            StimulusDataset: The loaded dataset.
        """

    # Methods for splitting and reconstruction

    @abstractmethod
    def select_split(self, split: str, indices: Any) -> Any:
        """Select a subset of a split.

        This method is used by the splitting logic to create train/test splits.
        It should return an object representing the subset of the data.
        This object will be passed to `create_from_splits`.

        Args:
            split (str): The name of the split to select from.
            indices (Any): The indices to select (e.g. list of integers, numpy array).

        Returns:
            Any: An opaque split object representing the subset.
        """

    @abstractmethod
    def create_from_splits(self, splits: dict[str, Any]) -> "StimulusDataset":
        """Create a new StimulusDataset from a dictionary of split objects.

        This method reconstructs a StimulusDataset from the outputs of `select_split`.

        Args:
            splits (dict[str, Any]): A dictionary mapping split names to split objects
                                     (as returned by `select_split`).

        Returns:
            StimulusDataset: A new dataset instance containing the provided splits.
        """


class TorchDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper to make HuggingFace dataset compatible with torch.utils.data.Dataset."""

    def __init__(self, dataset: Any):
        """Initialize the TorchDatasetWrapper."""
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: Any) -> Any:
        return self.dataset[idx]


class HuggingFaceDataset(StimulusDataset):
    """Wrapper for HuggingFace DatasetDict."""

    def __init__(self, dataset: datasets.DatasetDict):
        """Initialize the HuggingFaceDataset."""
        self._dataset = dataset

    @property
    def split_names(self) -> list[str]:
        """Return the list of split names."""
        return list(self._dataset.keys())

    @property
    def column_names(self) -> dict[str, list[str]]:
        """Return the column names for each split."""
        return {k: v.column_names for k, v in self._dataset.items()}

    def get_column(self, split: str, column_name: str) -> list[Any] | np.ndarray:
        """Get a column from a specific split."""
        return list(self._dataset[split][column_name])

    def get_torch_dataset(self, split: Union[str, list[str]]) -> torch.utils.data.Dataset:
        """Get a PyTorch dataset for the specified split(s)."""
        if isinstance(split, list):
            splits = [self._dataset[s] for s in split]
            ds = datasets.concatenate_datasets(splits)
        else:
            ds = self._dataset[split]

        ds = ds.with_format("torch")
        return TorchDatasetWrapper(ds)

    def map(
        self,
        function: Callable,
        *,
        batched: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        remove_columns: Optional[list[str]] = None,
    ) -> "HuggingFaceDataset":
        """Apply a function to each example in the dataset."""
        new_dataset = self._dataset.map(
            function,
            batched=batched,
            fn_kwargs=fn_kwargs,
            num_proc=num_proc,
            remove_columns=remove_columns,
        )
        return HuggingFaceDataset(new_dataset)

    def apply(self, transformation: Callable) -> "HuggingFaceDataset":
        """Apply a transformation to the dataset."""
        scope = getattr(transformation, "scope", "element")
        if scope == "dataset":
            # For dataset-level transforms, we assume the transform takes the dataset
            # and returns a new dataset (or modifies it if mutable, but HF is immutable-ish).
            # We pass the wrapper itself to the transform? Or the underlying dataset?
            # If we want to be generic, we should pass the wrapper.
            # But existing transforms might not know about the wrapper.
            # For now, let's assume the transform knows how to handle the wrapper
            # OR we unwrap it if it's a known type.
            # But to keep it simple:
            return transformation(self)
        # Element-level transform
        return self.map(transformation)

    def filter(self, function: Callable, *, batched: bool = False, **kwargs: Any) -> "HuggingFaceDataset":
        """Filter the dataset using a function."""
        new_dataset = self._dataset.filter(function, batched=batched, **kwargs)
        return HuggingFaceDataset(new_dataset)

    def save(self, path: str) -> None:
        """Save the dataset to disk."""
        self._dataset.save_to_disk(path)

    def select_split(self, split: str, indices: Any) -> Any:
        """Select a subset of examples from a split."""
        return self._dataset[split].select(indices)

    def create_from_splits(self, splits: dict[str, Any]) -> "HuggingFaceDataset":
        """Create a new dataset from a dictionary of splits."""
        return HuggingFaceDataset(datasets.DatasetDict(splits))

    @property
    def unwrap(self) -> datasets.DatasetDict:
        """Access the underlying HuggingFace dataset."""
        return self._dataset

    @classmethod
    def load_from_disk(cls, path: str, **kwargs: Any) -> "StimulusDataset":
        """Load a dataset from disk with strict format checking.

        Args:
            path: Path to the dataset file (CSV/Parquet) or directory.
            **kwargs: Additional arguments passed to the underlying dataset loader.
                      Common arguments include:
                      - keep_in_memory (bool): Whether to copy the dataset to memory.
                      - storage_options (dict): Options for the storage backend (e.g. S3).

        Returns:
            HuggingFaceDataset: The loaded dataset.

        Raises:
            ValueError: If the file extension is not supported or format mismatch occurs.
        """
        import logging
        import os

        import datasets

        logger = logging.getLogger(__name__)

        if os.path.isdir(path):
            logger.info(f"Loading dataset from directory: {path}")
            dataset = datasets.load_from_disk(path, **kwargs)
        elif path.endswith(".parquet"):
            logger.info(f"Loading as parquet: {path}")
            dataset = datasets.load_dataset("parquet", data_files=path, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file format or missing extension for path: {path}. Expected .parquet or a directory.",
            )

        return cls(dataset)


class AnnDataTorchDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for AnnData that densifies data into memory."""

    def __init__(self, adata: Any, columns: list[str]):
        """Initialize the dataset by loading everything into memory.

        Args:
            adata: The AnnData object.
            columns: List of columns to include. "X" is the main matrix.
        """
        self.columns = columns
        self.n_obs = adata.n_obs
        self.data = {}

        for col in columns:
            if col == "X":
                val = adata.X
                if hasattr(val, "toarray"):
                    val = val.toarray()
                # Create tensor once for the whole dataset
                self.data[col] = torch.tensor(val, dtype=torch.float32)
            elif col in adata.obs:
                val = adata.obs[col].values
                # Convert to tensor if numeric, otherwise keep as numpy array
                if np.issubdtype(val.dtype, np.number):
                    self.data[col] = torch.tensor(val)
                else:
                    self.data[col] = val
            else:
                raise ValueError(f"Column {col} not found in AnnData object.")

    def __len__(self) -> int:
        return self.n_obs

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {col: self.data[col][idx] for col in self.columns}


class H5adDataset(StimulusDataset):
    """Wrapper for AnnData objects loaded from .h5ad files."""

    def __init__(self, adata: Any, split_col: Optional[str] = None):
        """Initialize the H5adDataset.

        Args:
            adata: The AnnData object.
            split_col: Name of the column in adata.obs that defines the splits.
                       If None, assumes a single 'train' split.
        """
        self._adata = adata
        self._split_col = split_col

    @property
    def split_names(self) -> list[str]:
        """Get the names of available splits."""
        if self._split_col:
            return list(self._adata.obs[self._split_col].unique())
        return ["train"]

    @property
    def column_names(self) -> dict[str, list[str]]:
        """Get column names for each split."""
        cols = ["X"] + list(self._adata.obs.columns)
        return {split: cols for split in self.split_names}

    def get_column(self, split: str, column_name: str) -> Union[list[Any], np.ndarray]:
        """Get a column from a specific split."""
        if self._split_col:
            subset = self._adata[self._adata.obs[self._split_col] == split]
        else:
            subset = self._adata

        if column_name == "X":
            return subset.X
        return subset.obs[column_name].values

    def get_torch_dataset(self, split: Union[str, list[str]]) -> torch.utils.data.Dataset:
        """Get a PyTorch Dataset for training or inference."""
        if isinstance(split, list):
            # Concatenate subsets
            # For AnnData, we can just filter the original object
            if self._split_col:
                mask = self._adata.obs[self._split_col].isin(split)
                subset = self._adata[mask]
            else:
                # If no split col, and we ask for multiple splits, it's ambiguous unless it's just ['train']
                if split == ["train"]:
                    subset = self._adata
                else:
                    raise ValueError("Cannot select multiple splits without a split column.")
        else:
            if self._split_col:
                subset = self._adata[self._adata.obs[self._split_col] == split]
            else:
                if split == "train":
                    subset = self._adata
                else:
                    raise ValueError(f"Split {split} not found.")

        # By default, include all columns + X
        # In practice, users might want to select specific columns.
        # The StimulusDataset interface doesn't strictly define how to select columns for the torch dataset
        # other than what the model expects.
        # For now, let's include X and all obs columns.
        cols = ["X"] + list(subset.obs.columns)
        return AnnDataTorchDataset(subset, cols)

    def map(
        self,
        function: Callable,
        *,
        batched: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        remove_columns: Optional[list[str]] = None,
    ) -> "StimulusDataset":
        """Apply a transformation to all splits in the dataset."""
        # AnnData doesn't have a native map function like HF.
        # We might need to implement this if needed, or raise NotImplementedError.
        # For now, let's raise NotImplementedError as it's complex to map in-place or out-of-place on AnnData efficiently.
        raise NotImplementedError("map operation is not yet supported for H5adDataset.")

    def apply(self, transformation: Callable) -> "StimulusDataset":
        """Apply a transformation to the dataset."""
        # Similar to map, but for dataset-level transforms.
        # If the transformation expects an AnnData object, we can pass it.
        scope = getattr(transformation, "scope", "element")
        if scope == "dataset":
             # Assume transformation takes AnnData and returns AnnData
            new_adata = transformation(self._adata)
            return H5adDataset(new_adata, self._split_col)
        raise NotImplementedError("Element-wise apply is not yet supported for H5adDataset.")

    def filter(self, function: Callable, *, batched: bool = False, **kwargs: Any) -> "StimulusDataset":
        """Filter all splits in the dataset."""
        # We can implement filter by iterating or applying mask
        raise NotImplementedError("filter operation is not yet supported for H5adDataset.")

    def save(self, path: str) -> None:
        """Save the dataset to disk."""
        self._adata.write_h5ad(path)

    @classmethod
    def load_from_disk(cls, path: str, **kwargs: Any) -> "H5adDataset":
        """Load a dataset from disk."""
        import anndata

        # Check for split column in kwargs or default
        split_col = kwargs.pop("split_col", None)
        adata = anndata.read_h5ad(path, **kwargs)
        return cls(adata, split_col=split_col)

    def select_split(self, split: str, indices: Any) -> Any:
        """Select a subset of a split."""
        # This is used for splitting logic.
        # We need to return a subset of the data.
        if self._split_col:
            subset = self._adata[self._adata.obs[self._split_col] == split]
        else:
            subset = self._adata
        
        return subset[indices]

    def create_from_splits(self, splits: dict[str, Any]) -> "StimulusDataset":
        """Create a new StimulusDataset from a dictionary of split objects."""
        import anndata

        # splits is a dict of {split_name: AnnData_subset}
        # We need to concatenate them and create a new AnnData object
        # and potentially add a split column.
        
        adatas = []
        for split_name, adata in splits.items():
            # Add split column
            if self._split_col:
                # If split col exists, update it
                # But adata is a view or copy, so be careful
                adata = adata.copy()
                adata.obs[self._split_col] = split_name
            else:
                # If no split col, we might need to create one if we are combining multiple splits
                # For now, let's assume we just concat.
                pass
            adatas.append(adata)
            
        new_adata = anndata.concat(adatas, join="outer")
        return H5adDataset(new_adata, self._split_col)
