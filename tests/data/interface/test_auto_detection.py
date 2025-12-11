"""Tests for auto-detection of datasets."""

import os
import tempfile

import pytest

from stimulus.data.interface.dataset_interface import (
    H5adDataset,
    HuggingFaceDataset,
    auto_detect_dataset,
)


class TestAutoDetection:
    """Tests for dataset auto-detection."""

    def test_auto_detect_h5ad(self) -> None:
        """Test detection of .h5ad files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test.h5ad")
            with open(file_path, "w") as f:
                f.write("dummy content")

            cls = auto_detect_dataset(file_path)
            assert cls == H5adDataset

    def test_auto_detect_parquet(self) -> None:
        """Test detection of .parquet files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test.parquet")
            with open(file_path, "w") as f:
                f.write("dummy content")

            cls = auto_detect_dataset(file_path)
            assert cls == HuggingFaceDataset

    def test_auto_detect_arrow(self) -> None:
        """Test detection of .arrow files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test.arrow")
            with open(file_path, "w") as f:
                f.write("dummy content")

            cls = auto_detect_dataset(file_path)
            assert cls == HuggingFaceDataset

    def test_auto_detect_directory(self) -> None:
        """Test detection of directory containing supported files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a nested directory structure
            nested_dir = os.path.join(tmp_dir, "data")
            os.makedirs(nested_dir)
            file_path = os.path.join(nested_dir, "data.arrow")
            with open(file_path, "w") as f:
                f.write("dummy content")

            cls = auto_detect_dataset(tmp_dir)
            assert cls == HuggingFaceDataset

    def test_auto_detect_fail(self) -> None:
        """Test failure for unsupported extension."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test.txt")
            with open(file_path, "w") as f:
                f.write("dummy content")

            with pytest.raises(ValueError, match="Unsupported file extension"):
                auto_detect_dataset(file_path)

    def test_auto_detect_empty_dir(self) -> None:
        """Test failure for empty directory."""
        with tempfile.TemporaryDirectory() as tmp_dir, pytest.raises(ValueError, match="Could not find any files"):
            auto_detect_dataset(tmp_dir)
