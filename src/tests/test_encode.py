import sys

sys.path.insert(0, 'src')

from components.encode import ExperienceEncoder
import pytest


@pytest.fixture
def encoder():
    return ExperienceEncoder(
        base_url="http://localhost:11434/v1"
    )


def test_encoder_initialization(encoder):
    assert encoder.base_url == "http://localhost:11434/v1"
    assert encoder.api_key == "dummy-key"
    assert encoder.client is None


def test_encoder_client_not_initialized(encoder):
    with pytest.raises(RuntimeError, match="LLM client not initialized"):
        encoder.encode_experience({"id": "test"})


def test_save_and_load_units(encoder, tmp_path):
    units = [
        {
            "type": "lesson",
            "content": "Test content",
            "metadata": {"key": "value"},
            "tags": ["test"]
        }
    ]
    filename = tmp_path / "test_units.json"
    encoder.save_units(units, str(filename))
    loaded = encoder.load_units(str(filename))
    assert len(loaded) == 1
    assert loaded[0]["type"] == "lesson"


def test_load_nonexistent_file(encoder):
    result = encoder.load_units("nonexistent.json")
    assert result == []
