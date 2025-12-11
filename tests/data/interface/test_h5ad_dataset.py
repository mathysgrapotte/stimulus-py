"""Test the H5adDataset."""

import os
import tempfile

import anndata
import numpy as np
import pandas as pd
import pytest
import torch

from stimulus.data.interface.dataset_interface import H5adDataset


@pytest.fixture
def adata() -> anndata.AnnData:
    """Create a dummy AnnData object."""
    n_obs = 100
    n_vars = 50
    x = np.random.rand(n_obs, n_vars)
    obs = pd.DataFrame(
        {
            "group": pd.Series(
                [str(x) for x in np.random.choice(["A", "B"], size=n_obs)],
                dtype="object",
            ),
            "value": np.random.rand(n_obs),
        },
    )
    obs.index = [str(i) for i in range(n_obs)]
    return anndata.AnnData(X=x, obs=obs)


@pytest.fixture
def adata_dict() -> dict[str, anndata.AnnData]:
    """Create a dictionary of AnnData objects for train/val splits."""
    n_vars = 50

    # Train split
    train_obs = 70
    train_x = np.random.rand(train_obs, n_vars)
    train_obs_df = pd.DataFrame(
        {
            "group": pd.Series(
                [str(x) for x in np.random.choice(["A", "B"], size=train_obs)],
                dtype="object",
            ),
            "value": np.random.rand(train_obs),
        },
    )
    train_obs_df.index = [str(i) for i in range(train_obs)]
    train_adata = anndata.AnnData(X=train_x, obs=train_obs_df)

    # Val split
    val_obs = 30
    val_x = np.random.rand(val_obs, n_vars)
    val_obs_df = pd.DataFrame(
        {
            "group": pd.Series(
                [str(x) for x in np.random.choice(["A", "B"], size=val_obs)],
                dtype="object",
            ),
            "value": np.random.rand(val_obs),
        },
    )
    val_obs_df.index = [str(i) for i in range(val_obs)]
    val_adata = anndata.AnnData(X=val_x, obs=val_obs_df)

    return {"train": train_adata, "val": val_adata}


def test_anndata_dataset_init_single(adata: anndata.AnnData) -> None:
    """Test initialization with single AnnData (wraps as train)."""
    dataset = H5adDataset(adata)
    assert dataset.split_names == ["train"]
    assert "X" in dataset.column_names["train"]
    assert "group" in dataset.column_names["train"]


def test_anndata_dataset_init_dict(adata_dict: dict[str, anndata.AnnData]) -> None:
    """Test initialization with dictionary of AnnData objects."""
    dataset = H5adDataset(adata_dict)
    assert set(dataset.split_names) == {"train", "val"}
    assert "X" in dataset.column_names["train"]
    assert "X" in dataset.column_names["val"]


def test_anndata_dataset_get_column(adata: anndata.AnnData) -> None:
    """Test get_column."""
    dataset = H5adDataset(adata)
    x = dataset.get_column("train", "X")
    assert x.shape == (100, 50)

    group = dataset.get_column("train", "group")
    assert len(group) == 100


def test_anndata_dataset_torch_dataset(adata: anndata.AnnData) -> None:
    """Test get_torch_dataset with single split."""
    dataset = H5adDataset(adata)
    torch_ds = dataset.get_torch_dataset("train")

    item = torch_ds[0]
    assert "X" in item
    assert isinstance(item["X"], torch.Tensor)
    assert item["X"].shape == (50,)
    assert "group" in item
    assert "value" in item
    assert isinstance(item["value"], torch.Tensor)


def test_anndata_dataset_torch_dataset_multiple(
    adata_dict: dict[str, anndata.AnnData],
) -> None:
    """Test get_torch_dataset with multiple splits returns dictionary."""
    dataset = H5adDataset(adata_dict)
    result = dataset.get_torch_dataset(["train", "val"])

    assert isinstance(result, dict)
    assert "train" in result
    assert "val" in result
    assert len(result["train"]) == 70
    assert len(result["val"]) == 30


def test_save_load_h5ad_dict(adata_dict: dict[str, anndata.AnnData]) -> None:
    """Test save and load with train/val splits."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "test_split_dataset")
        dataset = H5adDataset(adata_dict)
        dataset.save(path)

        # Check that directory structure was created (flat: train.h5ad, val.h5ad)
        assert os.path.isdir(path)
        assert os.path.exists(os.path.join(path, "train.h5ad"))
        assert os.path.exists(os.path.join(path, "val.h5ad"))

        # Load back and verify
        loaded_dataset = H5adDataset.load_from_disk(path)
        assert set(loaded_dataset.split_names) == {"train", "val"}
        assert len(loaded_dataset.get_torch_dataset("train")) == 70
        assert len(loaded_dataset.get_torch_dataset("val")) == 30


def test_save_load_h5ad_single(adata: anndata.AnnData) -> None:
    """Test save and load with single AnnData (train only)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "test_single_dataset")
        dataset = H5adDataset(adata)
        dataset.save(path)

        # Check that directory structure was created with train.h5ad
        assert os.path.isdir(path)
        assert os.path.exists(os.path.join(path, "train.h5ad"))

        # Load back and verify
        loaded_dataset = H5adDataset.load_from_disk(path)
        assert loaded_dataset.split_names == ["train"]
        assert len(loaded_dataset.get_torch_dataset("train")) == 100


def test_load_single_file() -> None:
    """Test loading a single .h5ad file wraps it as train."""
    n_obs = 100
    n_vars = 50
    x = np.random.rand(n_obs, n_vars)
    obs = pd.DataFrame({"value": np.random.rand(n_obs)})
    obs.index = [str(i) for i in range(n_obs)]
    adata = anndata.AnnData(X=x, obs=obs)

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "test_single.h5ad")
        adata.write_h5ad(path)

        # Load single file - should wrap as {"train": adata}
        loaded_dataset = H5adDataset.load_from_disk(path)
        assert loaded_dataset.split_names == ["train"]
        assert len(loaded_dataset.get_torch_dataset("train")) == 100


def test_load_directory_with_extra_files() -> None:
    """Test loading directory with >2 files merges all into train."""
    n_vars = 50

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create 3 h5ad files with unique indices across all files
        total_idx = 0
        for i, name in enumerate(["part1", "part2", "part3"]):
            n_obs = 30 + i * 10  # 30, 40, 50
            x = np.random.rand(n_obs, n_vars)
            obs = pd.DataFrame({"value": np.random.rand(n_obs)})
            # Use globally unique indices to avoid duplicate warnings
            obs.index = [str(total_idx + j) for j in range(n_obs)]
            total_idx += n_obs
            adata = anndata.AnnData(X=x, obs=obs)
            adata.write_h5ad(os.path.join(temp_dir, f"{name}.h5ad"))

        # Load directory - should merge all into train
        loaded_dataset = H5adDataset.load_from_disk(temp_dir)
        assert loaded_dataset.split_names == ["train"]
        # Total observations: 30 + 40 + 50 = 120
        assert len(loaded_dataset.get_torch_dataset("train")) == 120


def test_select_split_and_create_from_splits(
    adata_dict: dict[str, anndata.AnnData],
) -> None:
    """Test select_split and create_from_splits for splitting workflow."""
    dataset = H5adDataset(adata_dict)

    # Select subset of train
    train_indices = list(range(35))  # First 35 samples
    train_subset = dataset.select_split("train", train_indices)

    # Select subset of val
    val_indices = list(range(15))  # First 15 samples
    val_subset = dataset.select_split("val", val_indices)

    # Create new dataset from subsets
    new_dataset = dataset.create_from_splits({"train": train_subset, "val": val_subset})

    assert len(new_dataset.get_torch_dataset("train")) == 35
    assert len(new_dataset.get_torch_dataset("val")) == 15
