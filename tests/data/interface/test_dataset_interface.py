"""Tests for the StimulusDataset interface and HuggingFaceDataset implementation."""

import os
import tempfile
from typing import Any

import datasets
import numpy as np
import pytest
import torch

from stimulus.data.interface.dataset_interface import HuggingFaceDataset, StimulusDataset


class TestHuggingFaceDataset:
    """Tests for the HuggingFaceDataset class."""

    @pytest.fixture
    def sample_dataset(self) -> datasets.DatasetDict:
        """Create a sample Hugging Face dataset."""
        data = {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
        dataset = datasets.Dataset.from_dict(data)
        return datasets.DatasetDict({"train": dataset, "val": dataset})

    @pytest.fixture
    def stimulus_dataset(self, sample_dataset: datasets.DatasetDict) -> HuggingFaceDataset:
        """Create a HuggingFaceDataset instance."""
        return HuggingFaceDataset(sample_dataset)

    def test_split_names(self, stimulus_dataset: HuggingFaceDataset) -> None:
        """Test split_names property."""
        assert set(stimulus_dataset.split_names) == {"train", "val"}

    def test_column_names(self, stimulus_dataset: HuggingFaceDataset) -> None:
        """Test column_names property."""
        assert set(stimulus_dataset.column_names["train"]) == {"col1", "col2", "col3"}
        assert set(stimulus_dataset.column_names["val"]) == {"col1", "col2", "col3"}

    def test_get_column(self, stimulus_dataset: HuggingFaceDataset) -> None:
        """Test get_column method."""
        col1 = stimulus_dataset.get_column("train", "col1")
        assert len(col1) == 5
        assert col1[0] == 1
        assert isinstance(col1, (list, np.ndarray))

    def test_get_torch_dataset(self, stimulus_dataset: HuggingFaceDataset) -> None:
        """Test get_torch_dataset method."""
        torch_dataset = stimulus_dataset.get_torch_dataset("train")
        assert isinstance(torch_dataset, torch.utils.data.Dataset)
        assert len(torch_dataset) == 5  # type: ignore[arg-type]
        item = torch_dataset[0]
        assert "col1" in item
        assert "col3" in item
        # col2 is string, might not be in default torch format unless handled,
        # but HF handles it by keeping it as string or excluding it if set_format is strict.
        # HuggingFaceDataset uses set_format("torch"), so strings might be kept as is or issues might arise if not handled.
        # Let's check what actually happens.

    def test_map(self, stimulus_dataset: HuggingFaceDataset) -> None:
        """Test map method."""

        def add_one(example: dict[str, Any]) -> dict[str, Any]:
            example["col1"] += 1
            return example

        mapped_dataset = stimulus_dataset.map(add_one, batched=False)
        col1 = mapped_dataset.get_column("train", "col1")
        assert col1[0] == 2

    def test_filter(self, stimulus_dataset: HuggingFaceDataset) -> None:
        """Test filter method."""
        filtered_dataset = stimulus_dataset.filter(lambda x: x["col1"] > 2, batched=False)
        col1 = filtered_dataset.get_column("train", "col1")
        assert len(col1) == 3
        assert col1[0] == 3

    def test_select_split(self, stimulus_dataset: HuggingFaceDataset) -> None:
        """Test select_split method."""
        indices = [0, 2, 4]
        split_obj = stimulus_dataset.select_split("train", indices)
        # split_obj is opaque, but for HF it's a Dataset
        assert len(split_obj) == 3

    def test_create_from_splits(self, stimulus_dataset: HuggingFaceDataset) -> None:
        """Test create_from_splits method."""
        indices = [0, 1]
        train_split = stimulus_dataset.select_split("train", indices)
        val_split = stimulus_dataset.select_split("val", indices)

        new_dataset = stimulus_dataset.create_from_splits({"train": train_split, "val": val_split})
        assert isinstance(new_dataset, HuggingFaceDataset)
        assert len(new_dataset.get_column("train", "col1")) == 2

    def test_save(self, stimulus_dataset: HuggingFaceDataset) -> None:
        """Test save method."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "saved_dataset")
            stimulus_dataset.save(save_path)
            assert os.path.exists(save_path)

            # Verify we can load it back (using HF load_from_disk for verification)
            loaded = datasets.load_from_disk(save_path)
            assert "train" in loaded

    def test_apply_element_scope(self, stimulus_dataset: HuggingFaceDataset) -> None:
        """Test apply method with element scope."""

        class AddOneTransform:
            scope = "element"

            def __call__(self, example: Any):
                example["col1"] += 1
                return example

        transform = AddOneTransform()
        # We need to wrap the transform to match the expected signature if it's not a simple callable
        # But apply expects a callable that might be a class with scope.
        # Let's mock a transform object.

        # Actually, apply takes a transformation callable.
        # If it has a scope attribute, it uses it.

        transformed_dataset = stimulus_dataset.apply(transform)
        col1 = transformed_dataset.get_column("train", "col1")
        assert col1[0] == 2

    def test_apply_dataset_scope(self, stimulus_dataset: HuggingFaceDataset) -> None:
        """Test apply method with dataset scope."""

        class DatasetTransform:
            scope = "dataset"

            def __call__(self, dataset: StimulusDataset) -> StimulusDataset:
                # Return a new dataset or modify in place (but we expect return)
                # Let's just return the dataset for this test, maybe check it was called
                return dataset

        transform = DatasetTransform()
        transformed_dataset = stimulus_dataset.apply(transform)
        assert isinstance(transformed_dataset, StimulusDataset)
