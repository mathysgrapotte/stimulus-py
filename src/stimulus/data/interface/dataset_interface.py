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

    @property
    def dataset_attributes(self) -> dict[str, Any]:
        """Get dataset-level attributes (optional).

        Returns:
            dict[str, Any]: Dictionary of attributes (e.g. {'gene_dim': 2000}).
                            Defaults to empty dict.
        """
        return {}

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
    def get_torch_dataset(
        self, split: Union[str, list[str]]
    ) -> Union[torch.utils.data.Dataset, dict[str, torch.utils.data.Dataset]]:
        """Get a PyTorch Dataset for training or inference.

        This method should return a standard PyTorch Dataset that yields samples
        compatible with the model's expected input.

        Args:
            split (Union[str, list[str]]): The split name(s) to retrieve.

        Returns:
            If split is a string: a single PyTorch Dataset for that split.
            If split is a list: a dictionary mapping split names to PyTorch Datasets.
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
    def filter(
        self, function: Callable, *, batched: bool = False, **kwargs: Any
    ) -> "StimulusDataset":
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

    def get_torch_dataset(
        self, split: Union[str, list[str]]
    ) -> Union[torch.utils.data.Dataset, dict[str, torch.utils.data.Dataset]]:
        """Get a PyTorch dataset for the specified split(s).

        Args:
            split: Either a single split name (str) or a list of split names.

        Returns:
            If split is a string: a single PyTorch Dataset for that split.
            If split is a list: a dictionary mapping split names to PyTorch Datasets.
        """
        if isinstance(split, list):
            result = {}
            for s in split:
                ds = self._dataset[s].with_format("torch")
                result[s] = TorchDatasetWrapper(ds)
            return result
        else:
            ds = self._dataset[split].with_format("torch")
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

    def filter(
        self, function: Callable, *, batched: bool = False, **kwargs: Any
    ) -> "HuggingFaceDataset":
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
    """PyTorch Dataset wrapper for AnnData with lazy loading and metadata support."""

    def __init__(
        self, adata: Any, columns: list[str], target_gene_col: str = "target_gene"
    ):
        """Initialize the dataset.

        Args:
            adata: The AnnData object.
            columns: List of columns to include. "X" is the main matrix.
            target_gene_col: Column name required for metadata extraction.
        """
        self.adata = adata
        self.columns = columns
        self.n_obs = adata.n_obs

        # Metadata storage for validation
        self.gene_names = list(adata.var_names)
        self.control_adata = None
        if target_gene_col in adata.obs:
            control_mask = adata.obs[target_gene_col] == "non-targeting"
            if np.any(control_mask):
                self.control_adata = adata[control_mask].copy()

        # Cache non-X columns
        self.cache = {}
        for col in columns:
            if col == "X":
                continue

            if col in adata.obs:
                val = adata.obs[col].values
                # Numeric check for tensor conversion
                is_numeric = False
                try:
                    is_numeric = np.issubdtype(val.dtype, np.number)
                except TypeError:
                    is_numeric = False

                self.cache[col] = torch.tensor(val) if is_numeric else val

            elif col in adata.obsm:
                # Always tensor for obsm
                self.cache[col] = torch.tensor(adata.obsm[col], dtype=torch.float32)
            else:
                # Warn or skip if column not found?
                # For now silent skip or keep existing behavior (error?)
                # user wants "opinionated", so maybe just skip missing?
                # But if requested explicitly, better to error or fill?
                # H5adDataset asks for Everything.
                pass

    def __len__(self) -> int:
        return self.n_obs

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {}

        # Add cached columns
        for col, data in self.cache.items():
            item[col] = data[idx]

        # Add X if requested (lazy load)
        if "X" in self.columns:
            val = self.adata.X[idx]
            if hasattr(val, "toarray"):
                val = val.toarray()
            elif hasattr(val, "todense"):
                val = np.array(val.todense())

            if isinstance(val, np.ndarray):
                val = val.flatten()

            item["X"] = torch.tensor(val, dtype=torch.float32)

        return item

    @property
    def dataset_attributes(self) -> dict[str, Any]:
        """Return dataset attributes like gene_dim, pca_dim."""
        attrs = {}
        # Gene dimension
        if self.adata is not None:
            attrs["gene_dim"] = self.adata.n_vars

            # PCA dimension if X_pca is present in obsm
            if "X_pca" in self.adata.obsm:
                attrs["pca_dim"] = self.adata.obsm["X_pca"].shape[1]
            elif "X_avg_pca" in self.adata.obsm:
                attrs["pca_dim"] = self.adata.obsm["X_avg_pca"].shape[1]

        return attrs


class H5adDataset(StimulusDataset):
    """Wrapper for AnnData objects loaded from .h5ad files.

    Data is always stored as a dictionary of {split_name: AnnData}.
    Single AnnData objects are wrapped as {"train": adata}.
    """

    def __init__(self, adata: Union[Any, dict[str, Any]]):
        """Initialize the H5adDataset.

        Args:
            adata: Either a single AnnData object or a dictionary of {split_name: AnnData}.
                   Single AnnData objects are wrapped as {"train": adata}.
        """
        if isinstance(adata, dict):
            self._adata_dict = adata
        else:
            # Wrap single AnnData as {"train": adata}
            self._adata_dict = {"train": adata}

    @property
    def split_names(self) -> list[str]:
        """Get the names of available splits."""
        return list(self._adata_dict.keys())

    @property
    def column_names(self) -> dict[str, list[str]]:
        """Get column names for each split."""
        result = {}
        for split_name, adata in self._adata_dict.items():
            cols = ["X", *list(adata.obs.columns)]
            result[split_name] = cols
        return result

    def get_column(self, split: str, column_name: str) -> Union[list[Any], np.ndarray]:
        """Get a column from a specific split."""
        if split not in self._adata_dict:
            raise ValueError(f"Split {split} not found in dataset")
        adata = self._adata_dict[split]
        if column_name == "X":
            return adata.X
        return adata.obs[column_name].values

    def get_torch_dataset(
        self, split: Union[str, list[str]]
    ) -> Union[torch.utils.data.Dataset, dict[str, torch.utils.data.Dataset]]:
        """Get a PyTorch Dataset for training or inference.

        Args:
            split: Either a single split name (str) or a list of split names.

        Returns:
            If split is a string: a single PyTorch Dataset for that split.
            If split is a list: a dictionary mapping split names to PyTorch Datasets.
        """
        if isinstance(split, list):
            # Return a dictionary of datasets
            result = {}
            for s in split:
                if s not in self._adata_dict:
                    raise ValueError(f"Split {s} not found in dataset")
                adata = self._adata_dict[s]
                cols = ["X", *list(adata.obs.columns), *list(adata.obsm.keys())]
                result[s] = AnnDataTorchDataset(adata, cols)
            return result
        else:
            # Single split - return a single dataset
            if split not in self._adata_dict:
                raise ValueError(f"Split {split} not found in dataset")
            adata = self._adata_dict[split]
            cols = ["X", *list(adata.obs.columns), *list(adata.obsm.keys())]
            return AnnDataTorchDataset(adata, cols)

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
        raise NotImplementedError("map operation is not yet supported for H5adDataset.")

    def apply(self, transformation: Callable) -> "StimulusDataset":
        """Apply a transformation to the dataset."""
        scope = getattr(transformation, "scope", "element")
        if scope == "dataset":
            new_adata_dict = {}
            for split_name, adata in self._adata_dict.items():
                new_adata_dict[split_name] = transformation(adata)
            return H5adDataset(new_adata_dict)
        raise NotImplementedError(
            "Element-wise apply is not yet supported for H5adDataset."
        )

    def filter(
        self, function: Callable, *, batched: bool = False, **kwargs: Any
    ) -> "StimulusDataset":
        """Filter all splits in the dataset."""
        raise NotImplementedError(
            "filter operation is not yet supported for H5adDataset."
        )

    def save(self, path: str) -> None:
        """Save the dataset to disk.

        Saves each split as a separate .h5ad file in the directory
        (e.g., path/train.h5ad, path/val.h5ad).
        """
        import os

        os.makedirs(path, exist_ok=True)

        for split_name, split_adata in self._adata_dict.items():
            split_path = os.path.join(path, f"{split_name}.h5ad")
            split_adata.write_h5ad(split_path)

    @classmethod
    def load_from_disk(cls, path: str, **kwargs: Any) -> "H5adDataset":
        """Load a dataset from disk.

        Args:
            path: Either a single .h5ad file or a directory containing .h5ad files.
                  - Single file: wrapped as {"train": adata}
                  - Directory with train.h5ad: loads train, optionally val.h5ad
                  - Directory with >2 .h5ad files: merges all into 'train'

        Returns:
            H5adDataset with splits stored as a dictionary.
        """
        import os

        import anndata

        if os.path.isfile(path) and path.endswith(".h5ad"):
            # Single file load - wrap as {"train": adata}
            adata = anndata.read_h5ad(path, **kwargs)
            return cls({"train": adata})

        if os.path.isdir(path):
            # Find all .h5ad files in the directory
            h5ad_files = {}
            try:
                for entry in os.listdir(path):
                    if entry.endswith(".h5ad"):
                        split_name = entry[:-5]  # Remove .h5ad extension
                        h5ad_files[split_name] = os.path.join(path, entry)
            except (OSError, PermissionError) as e:
                raise ValueError(f"Error reading directory {path}: {e}") from e

            if not h5ad_files:
                raise ValueError(
                    f"Directory {path} does not contain any .h5ad files"
                )

            # Load based on number of files found
            if len(h5ad_files) == 1:
                # Single file in directory - load as train
                split_name, file_path = next(iter(h5ad_files.items()))
                adata = anndata.read_h5ad(file_path, **kwargs)
                return cls({"train": adata})

            if len(h5ad_files) == 2 and "train" in h5ad_files and "val" in h5ad_files:
                # Exactly train and val - load both
                adata_dict = {
                    "train": anndata.read_h5ad(h5ad_files["train"], **kwargs),
                    "val": anndata.read_h5ad(h5ad_files["val"], **kwargs),
                }
                return cls(adata_dict)

            # More than 2 files or unexpected names - merge all into train
            adatas = []
            for split_name in sorted(h5ad_files.keys()):
                adatas.append(anndata.read_h5ad(h5ad_files[split_name], **kwargs))
            merged = anndata.concat(adatas, join="outer")
            merged.obs_names_make_unique()
            return cls({"train": merged})

        raise ValueError(
            f"Path {path} is neither a .h5ad file nor a directory"
        )

    def select_split(self, split: str, indices: Any) -> Any:
        """Select a subset of a split."""
        if split not in self._adata_dict:
            raise ValueError(f"Split {split} not found in dataset")
        return self._adata_dict[split][indices]

    def create_from_splits(self, splits: dict[str, Any]) -> "StimulusDataset":
        """Create a new StimulusDataset from a dictionary of split objects.

        Args:
            splits: Dictionary mapping split names to AnnData objects.

        Returns:
            H5adDataset with splits stored as a dictionary.
        """
        adata_dict = {}
        for split_name, adata in splits.items():
            adata_dict[split_name] = adata.copy()
        return H5adDataset(adata_dict)
