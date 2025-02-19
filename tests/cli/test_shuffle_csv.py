"""Tests for the shuffle_csv CLI command."""

import hashlib
import pathlib
import tempfile
from typing import Any, Callable

import pytest

from src.stimulus.cli.shuffle_csv import main


# Fixtures
@pytest.fixture
def correct_yaml_path() -> str:
    """Fixture that returns the path to a correct YAML file."""
    return "tests/test_data/titanic/titanic_sub_config.yaml"


@pytest.fixture
def correct_csv_path() -> str:
    """Fixture that returns the path to a correct CSV file."""
    return "tests/test_data/titanic/titanic_stimulus.csv"


# Test cases
test_cases = [
    ("correct_csv_path", "correct_yaml_path", None),
]


# Tests
@pytest.mark.parametrize(("csv_type", "yaml_type", "error"), test_cases)
def test_shuffle_csv(
    request: pytest.FixtureRequest,
    snapshot: Callable[[], Any],
    csv_type: str,
    yaml_type: str,
    error: Exception | None,
) -> None:
    """Tests the CLI command with correct and wrong YAML files."""
    csv_path = request.getfixturevalue(csv_type)
    yaml_path = request.getfixturevalue(yaml_type)
    tmpdir = pathlib.Path(tempfile.gettempdir())
    if error:
        with pytest.raises(error):  # type: ignore[call-overload]
            main(csv_path, yaml_path, str(tmpdir / "test.csv"))
    else:
        main(csv_path, yaml_path, str(tmpdir / "test.csv"))
        with open(tmpdir / "test.csv") as file:
            hash = hashlib.md5(file.read().encode()).hexdigest()  # noqa: S324
        assert hash == snapshot
