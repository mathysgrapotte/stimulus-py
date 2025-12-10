"""Tests for AveragePca transform."""

import anndata
import numpy as np
import pytest

from stimulus.data.transforming.transforms import AveragePca


@pytest.fixture
def adata_with_groups() -> anndata.AnnData:
    """Create a dummy AnnData object with groups."""
    n_obs = 200
    n_vars = 50
    # Create sufficient distinct groups (>= n_components=5)
    # 20 groups with 10 samples each
    n_groups = 20
    samples_per_group = 10

    conditions = []
    for i in range(n_groups):
        conditions.extend([f"Group_{i}"] * samples_per_group)

    x = np.random.normal(0, 1, (n_obs, n_vars))

    # Add signal to odd groups
    for i in range(1, n_groups, 2):
        start = i * samples_per_group
        end = start + samples_per_group
        x[start:end, :10] += 5

    # Add non-targeting group to test filtering
    x_nt = np.random.normal(0, 1, (20, n_vars))
    obs_nt = {"condition": ["non-targeting"] * 20}

    x = np.vstack([x, x_nt])
    obs = {"condition": np.hstack([conditions, obs_nt["condition"]])}

    return anndata.AnnData(X=x, obs=obs)


def test_average_pca_init() -> None:
    """Test initialization."""
    transform = AveragePca(
        n_components=5,
        field="condition",
        store_field="X_avg_pca",
    )
    assert transform.n_components == 5
    assert transform.field == "condition"
    assert transform.store_field == "X_avg_pca"
    assert transform.scope == "dataset"
    assert transform.loadings is None
    assert transform.pca_mean is None


def test_average_pca_transform(adata_with_groups: anndata.AnnData) -> None:
    """Test applying the transform."""
    n_components = 5
    store_field = "X_avg_pca"
    transform = AveragePca(
        n_components=n_components,
        field="condition",
        store_field=store_field,
        seed=42,
    )

    # Apply transform
    transformed_adata = transform.transform(adata_with_groups)

    # Check if store_field exists
    assert store_field in transformed_adata.obsm

    # Check shape
    assert transformed_adata.obsm[store_field].shape == (adata_with_groups.n_obs, n_components)

    # Check if loadings and mean are stored in the object
    assert transform.loadings is not None
    assert transform.pca_mean is not None
    assert transform.loadings.shape == (adata_with_groups.n_vars, n_components)

    # Check if loadings are stored in adata
    assert "PCs" in transformed_adata.varm
    assert np.allclose(transformed_adata.varm["PCs"], transform.loadings)

    # Check consistency
    # (X - mean) @ loadings = projected
    # Note: We need to respect the removal of non-targeting if it influenced the mean/loadings
    # In this test, we kept default removal of 'non-targeting'

    # Manually project
    expected_proj = (adata_with_groups.X - transform.pca_mean) @ transform.loadings
    assert np.allclose(transformed_adata.obsm[store_field], expected_proj)


def test_average_pca_filtering(adata_with_groups: anndata.AnnData) -> None:
    """Test that filtering works (affects the model fit)."""
    # Fit with filtering non-targeting
    transform_filtered = AveragePca(
        n_components=5,
        field="condition",
        store_field="X_pca_filtered",
        remove_target_field="condition",
        remove_target_values=["non-targeting"],
        seed=42,
    )
    _adata_filtered = transform_filtered.transform(adata_with_groups.copy())

    # Fit without filtering
    # Note: Using fewer components here to avoid min(samples, features) error if non-targeting adds only 1 group vs 20
    # Actually we have 20 groups already so it's fine.
    transform_full = AveragePca(
        n_components=5,
        field="condition",
        store_field="X_pca_full",
        remove_target_values=[],  # Don't remove anything
        seed=42,
    )
    _adata_full = transform_full.transform(adata_with_groups.copy())

    # Loadings should be different because the "non-targeting" group
    # affects the condition averages if included
    # (The non-targeting group has different mean)
    assert not np.allclose(transform_filtered.loadings, transform_full.loadings)


def test_average_pca_pytorch_integration(adata_with_groups: anndata.AnnData) -> None:
    """Test that the transformed data is accessible via H5adDataset.get_torch_dataset."""
    from stimulus.data.interface.dataset_interface import H5adDataset

    store_field = "X_avg_pca"
    transform = AveragePca(
        n_components=5,
        field="condition",
        store_field=store_field,
        seed=42,
    )

    # Transform
    transformed_adata = transform.transform(adata_with_groups)

    # Wrap in H5adDataset
    dataset = H5adDataset(transformed_adata)

    # Get torch dataset
    torch_dataset = dataset.get_torch_dataset(split="train")

    # Get a sample
    sample = torch_dataset[0]

    # Verify the PCA field is present and correct
    assert store_field in sample
    assert sample[store_field].shape == (5,)  # Single sample, 5 components
    assert np.allclose(sample[store_field].numpy(), transformed_adata.obsm[store_field][0])
