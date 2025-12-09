"""Test the AnnDataDataset."""

import os
import tempfile
from typing import Generator

import anndata
import numpy as np
import pandas as pd
import pytest
import torch

from stimulus.data.interface.anndata_dataset import AnnDataDataset


@pytest.fixture
def adata() -> Generator[anndata.AnnData, None, None]:
    """Create a dummy AnnData object."""
    n_obs = 100
    n_vars = 50
    X = np.random.rand(n_obs, n_vars)
    obs = pd.DataFrame(
        {
            "group": np.random.choice(["A", "B"], size=n_obs),
            "split": np.random.choice(["train", "test"], size=n_obs),
            "value": np.random.rand(n_obs),
        }
    )
    adata = anndata.AnnData(X=X, obs=obs)
    yield adata


def test_anndata_dataset_init(adata: anndata.AnnData) -> None:
    """Test initialization."""
    dataset = AnnDataDataset(adata)
    assert dataset.split_names == ["train"]
    assert "X" in dataset.column_names["train"]
    assert "group" in dataset.column_names["train"]


def test_anndata_dataset_split(adata: anndata.AnnData) -> None:
    """Test split handling."""
    dataset = AnnDataDataset(adata, split_col="split")
    assert set(dataset.split_names) == {"train", "test"}
    
    train_ds = dataset.get_torch_dataset("train")
    test_ds = dataset.get_torch_dataset("test")
    
    assert len(train_ds) + len(test_ds) == 100


def test_anndata_dataset_get_column(adata: anndata.AnnData) -> None:
    """Test get_column."""
    dataset = AnnDataDataset(adata)
    X = dataset.get_column("train", "X")
    assert X.shape == (100, 50)
    
    group = dataset.get_column("train", "group")
    assert len(group) == 100


def test_anndata_dataset_torch_dataset(adata: anndata.AnnData) -> None:
    """Test get_torch_dataset."""
    dataset = AnnDataDataset(adata)
    torch_ds = dataset.get_torch_dataset("train")
    
    item = torch_ds[0]
    assert "X" in item
    assert isinstance(item["X"], torch.Tensor)
    assert item["X"].shape == (50,)
    assert "group" in item
    assert "value" in item
    assert isinstance(item["value"], torch.Tensor)


def test_save_load_h5ad(adata: anndata.AnnData) -> None:
    """Test save and load."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "test.h5ad")
        dataset = AnnDataDataset(adata, split_col="split")
        dataset.save(path)
        
        assert os.path.exists(path)
        
        loaded_dataset = AnnDataDataset.load_from_disk(path, split_col="split")
        assert set(loaded_dataset.split_names) == {"train", "test"}
        assert len(loaded_dataset.get_torch_dataset("train")) == len(dataset.get_torch_dataset("train"))
