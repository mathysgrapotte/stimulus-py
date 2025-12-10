"""Test the ScanpyTransform."""

import anndata
import numpy as np
import pytest

from stimulus.data.transforming.transforms import ScanpyTransform


@pytest.fixture
def adata() -> anndata.AnnData:
    """Create a dummy AnnData object."""
    n_obs = 100
    n_vars = 50
    x = np.random.rand(n_obs, n_vars)
    obs = {"group": np.random.choice(["A", "B"], size=n_obs)}
    return anndata.AnnData(X=x, obs=obs)


def test_scanpy_transform_init() -> None:
    """Test initialization."""
    transform = ScanpyTransform("pp.normalize_total", target_sum=1e4)
    assert transform.func == "pp.normalize_total"
    assert transform.kwargs == {"target_sum": 1e4}


def test_scanpy_transform_apply(adata: anndata.AnnData) -> None:
    """Test applying the transform."""
    # Using a simple function: pp.log1p (log(x+1))
    transform = ScanpyTransform("pp.log1p")

    # Calculate expected
    expected = np.log1p(adata.X.copy())

    # Apply transform
    transformed_adata = transform.transform(adata)

    # Verify modification
    # Note: Scanpy modifies in-place usually
    assert np.allclose(transformed_adata.X, expected)


def test_scanpy_transform_kwargs(adata: anndata.AnnData) -> None:
    """Test applying the transform with kwargs."""
    # Using pp.normalize_total
    target_sum = 1e4
    transform = ScanpyTransform("pp.normalize_total", target_sum=target_sum)

    transformed_adata = transform.transform(adata)

    # Verify sums approx target_sum
    # (might not be exact for all cells but sum over row should be close)
    row_sums = np.sum(transformed_adata.X, axis=1)
    assert np.allclose(row_sums, target_sum, rtol=1e-5)
