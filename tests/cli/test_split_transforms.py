"""Test for the split_transforms CLI command."""

import hashlib
import os
from pathlib import Path
from typing import Any, Callable

import pytest

from src.stimulus.cli.split_transforms import main


# Fixtures
@pytest.fixture
def correct_yaml_path() -> str:
    """Fixture that returns the path to a correct YAML file with one split only."""
    return "tests/test_data/titanic/titanic_unique_split.yaml"


@pytest.fixture
def wrong_yaml_path() -> str:
    """Fixture that returns the path to a wrong YAML file."""
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
    tmp_path: Path,  # Pytest tmp file system
) -> None:
    """Tests the CLI command with correct and wrong YAML files."""
    yaml_path: str = request.getfixturevalue(yaml_type)
    tmpdir = str(tmp_path)
    if error is not None:
        with pytest.raises(error):  # type: ignore[call-overload]
            main(yaml_path, tmpdir)
    else:
        # convert tmpdir to str
        main(yaml_path, tmpdir)
        files = os.listdir(tmpdir)
        test_out = [f for f in files if f.startswith("test_")]
        hashes = []
        for f in test_out:
            with open(os.path.join(tmpdir, f)) as file:
                hashes.append(hashlib.sha256(file.read().encode()).hexdigest())
        # Sorted ensures that the order of the hashes does not matter
        assert sorted(hashes) == snapshot
