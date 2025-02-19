"""Test for the split_transforms CLI command"""

import hashlib
import os
from typing import Any, Callable

import pytest

from src.stimulus.cli import split_transforms


# Fixtures
@pytest.fixture
def correct_yaml_path() -> str:
    """Fixture that returns the path to a correct YAML file with one split only"""
    return "tests/test_data/titanic/titanic_unique_split.yaml"


@pytest.fixture
def wrong_yaml_path() -> str:
    """Fixture that returns the path to a wrong YAML file"""
    return "tests/test_data/yaml_files/wrong_field_type.yaml"


# Test cases
test_cases = [("correct_yaml_path", None), ("wrong_yaml_path", ValueError)]


# Tests
@pytest.mark.parametrize(("yaml_type", "error"), test_cases)
def test_split_transforms(
    request: pytest.FixtureRequest,
    snapshot: Callable[[], Any],
    yaml_type: str,
    error: Exception | None,
    tmp_path,  # Pytest tmp file system
) -> None:
    """Tests the CLI command with correct and wrong YAML files."""
    yaml_path: str = request.getfixturevalue(yaml_type)
    tmpdir = tmp_path
    if error:
        with pytest.raises(error):
            split_transforms.main(yaml_path, tmpdir)
    else:
        split_transforms.main(yaml_path, tmpdir)
        files = os.listdir(tmpdir)
        test_out = [f for f in files if f.startswith("test_")]
        hashes = []
        for f in test_out:
            with open(os.path.join(tmpdir, f)) as file:
                hashes.append(hashlib.md5(file.read().encode()).hexdigest())
        assert sorted(hashes) == snapshot  # Sorted ensures that the order of the hashes does not matter
