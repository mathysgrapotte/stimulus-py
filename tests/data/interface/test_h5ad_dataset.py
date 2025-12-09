"""Test the AnnDataDataset."""

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
            "group": pd.Series([str(x) for x in np.random.choice(["A", "B"], size=n_obs)], dtype="object"),
            "split": pd.Series([str(x) for x in np.random.choice(["train", "test"], size=n_obs)], dtype="object"),
            "value": np.random.rand(n_obs),
        },
    )
    obs.index = [str(i) for i in range(n_obs)]
    # obs["group"] = pd.Categorical(obs["group"])
    # obs["split"] = pd.Categorical(obs["split"])
    return anndata.AnnData(X=x, obs=obs)


def test_anndata_dataset_init(adata: anndata.AnnData) -> None:
    """Test initialization."""
    dataset = H5adDataset(adata)
    assert dataset.split_names == ["train"]
    assert "X" in dataset.column_names["train"]
    assert "group" in dataset.column_names["train"]


def test_anndata_dataset_split(adata: anndata.AnnData) -> None:
    """Test split handling."""
    dataset = H5adDataset(adata, split_col="split")
    assert set(dataset.split_names) == {"train", "test"}

    train_ds = dataset.get_torch_dataset("train")
    test_ds = dataset.get_torch_dataset("test")

    assert len(train_ds) + len(test_ds) == 100


def test_anndata_dataset_get_column(adata: anndata.AnnData) -> None:
    """Test get_column."""
    dataset = H5adDataset(adata)
    x = dataset.get_column("train", "X")
    assert x.shape == (100, 50)

    group = dataset.get_column("train", "group")
    assert len(group) == 100


def test_anndata_dataset_torch_dataset(adata: anndata.AnnData) -> None:
    """Test get_torch_dataset."""
    dataset = H5adDataset(adata)
    torch_ds = dataset.get_torch_dataset("train")

    item = torch_ds[0]
    assert "X" in item
    assert isinstance(item["X"], torch.Tensor)
    assert item["X"].shape == (50,)
    assert "group" in item
    assert "value" in item
    assert isinstance(item["value"], torch.Tensor)


@pytest.mark.xfail(reason="AnnData persistence issue with string/categorical columns")
def test_save_load_h5ad(adata: anndata.AnnData) -> None:
    """Test save and load."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "test.h5ad")
        dataset = H5adDataset(adata, split_col="split")
        dataset.save(path)

        assert os.path.exists(path)

        loaded_dataset = H5adDataset.load_from_disk(path, split_col="split")
        assert set(loaded_dataset.split_names) == {"train", "test"}
        assert len(loaded_dataset.get_torch_dataset("train")) == len(dataset.get_torch_dataset("train"))


def test_save_load_h5ad_numeric() -> None:
    """Test save and load with numeric only."""
    n_obs = 100
    n_vars = 50
    x = np.random.rand(n_obs, n_vars)
    obs = pd.DataFrame({"value": np.random.rand(n_obs)})
    obs.index = [str(i) for i in range(n_obs)]
    adata = anndata.AnnData(X=x, obs=obs)

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "test_numeric.h5ad")
        dataset = H5adDataset(adata)
        dataset.save(path)

        assert os.path.exists(path)
        loaded_dataset = H5adDataset.load_from_disk(path)
        assert len(loaded_dataset.get_torch_dataset("train")) == 100
