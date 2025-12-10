"""Tests for the transform CLI command with Scanpy transforms."""

import os
import tempfile

import anndata
import numpy as np
import pytest
import yaml

from stimulus.cli.transform import transform
from stimulus.data.interface.dataset_interface import H5adDataset


@pytest.fixture
def h5ad_data() -> anndata.AnnData:
    """Create a dummy AnnData object."""
    n_obs = 100
    n_vars = 50
    x = np.random.rand(n_obs, n_vars)
    obs = {"group": np.random.choice(["A", "B"], size=n_obs)}
    return anndata.AnnData(X=x, obs=obs)


def test_scanpy_transform_cli(h5ad_data: anndata.AnnData) -> None:
    """Test the transform CLI with a Scanpy transform."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save dummy h5ad
        input_path = os.path.join(tmp_dir, "input.h5ad")
        h5ad_data.write_h5ad(input_path)

        # Create config
        config = {
            "global_params": {"seed": 42},
            "transforms": {
                "transformation_name": "log_transform",
                "columns": [
                    {
                        "column_name": "X",  # Column name is required by schema but ignored for dataset scope
                        "transformations": [
                            {
                                "name": "ScanpyTransform",
                                "params": {"func": "pp.log1p"},
                            },
                        ],
                    },
                ],
            },
        }

        config_path = os.path.join(tmp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        output_path = os.path.join(tmp_dir, "output.h5ad")

        # Run transform
        transform(
            data_path=input_path,
            config_yaml=config_path,
            out_path=output_path,
            dataset_cls=H5adDataset,
        )

        # Verify output
        assert os.path.exists(output_path)

        # Load and check transformation
        new_adata = anndata.read_h5ad(output_path)

        # Check if values are logged: log1p(x) should be < x for x > 0 (and x was rand[0,1])
        # Or simple exact check vs manual
        expected = np.log1p(h5ad_data.X)
        assert np.allclose(new_adata.X, expected)


def test_scanpy_transform_cli_chaining(h5ad_data: anndata.AnnData) -> None:
    """Test chaining multiple Scanpy transforms."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, "input.h5ad")
        h5ad_data.write_h5ad(input_path)

        # Normalize then Log
        config = {
            "global_params": {"seed": 42},
            "transforms": {
                "transformation_name": "norm_and_log",
                "columns": [
                    {
                        "column_name": "X",
                        "transformations": [
                            {
                                "name": "ScanpyTransform",
                                "params": {"func": "pp.normalize_total", "target_sum": 1e4},
                            },
                            {
                                "name": "ScanpyTransform",
                                "params": {"func": "pp.log1p"},
                            },
                        ],
                    },
                ],
            },
        }

        config_path = os.path.join(tmp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        output_path = os.path.join(tmp_dir, "output.h5ad")

        transform(
            data_path=input_path,
            config_yaml=config_path,
            out_path=output_path,
            dataset_cls=H5adDataset,
        )

        new_adata = anndata.read_h5ad(output_path)

        # Check manual calculation
        import scanpy as sc

        expected_adata = h5ad_data.copy()
        sc.pp.normalize_total(expected_adata, target_sum=1e4)
        sc.pp.log1p(expected_adata)

        assert np.allclose(new_adata.X, expected_adata.X)
