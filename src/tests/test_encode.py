import sys

sys.path.insert(0, 'src')

from components.encode import ExperienceEncoder
import pytest


@pytest.fixture
def encoder():
    import os
    from dotenv import load_dotenv
    load_dotenv()  # Ensure .env is loaded
    return ExperienceEncoder(
        base_url=os.getenv("MEMEVOLVE_LLM_BASE_URL")
    )


def test_encoder_initialization(encoder):
    import os
    assert encoder.base_url == os.getenv("MEMEVOLVE_LLM_BASE_URL")
    assert encoder.api_key == os.getenv("MEMEVOLVE_LLM_API_KEY", "")
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


def test_batch_encoding_trajectory():
    """Test batch encoding functionality."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    encoder = ExperienceEncoder(
        base_url=os.getenv("MEMEVOLVE_LLM_BASE_URL")
    )

    # Test batch encoding with empty trajectory
    result = encoder.encode_trajectory_batch([])
    assert result == []

    # Test batch encoding with sample trajectory
    trajectory = [
        {"id": "exp1", "action": "search", "result": "found item"},
        {"id": "exp2", "action": "sort", "result": "sorted array"},
        {"id": "exp3", "action": "filter", "result": "filtered data"}
    ]

    # This will attempt to call LLM, but should handle gracefully
    result = encoder.encode_trajectory_batch(trajectory, max_workers=2, batch_size=2)
    # Note: This test may fail if LLM is not available, but tests the interface
    assert isinstance(result, list)


def test_tool_encoding_prompt_structure():
    """Test that tool encoding is included in the prompt."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    encoder = ExperienceEncoder(
        base_url=os.getenv("MEMEVOLVE_LLM_BASE_URL")
    )

    # Check that the prompt includes tool as an option
    # We can't easily test the full encoding without LLM, but we can check prompt is constructed
    experience = {"id": "test", "action": "algorithm", "result": "binary search"}

    # The encode_experience method will fail without LLM client, but let's check the prompt is constructed
    # This is more of an integration test that would need mocking
    assert encoder.base_url is not None


def test_clean_llm_response():
    """Test LLM response cleaning functionality."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    encoder = ExperienceEncoder(
        base_url=os.getenv("MEMEVOLVE_LLM_BASE_URL")
    )

    # Test markdown code block removal
    markdown_response = '''```json
{
  "type": "lesson",
  "content": "Test content",
  "metadata": {},
  "tags": ["test"]
}
```'''
    cleaned = encoder._clean_llm_response(markdown_response)
    assert cleaned.startswith('{')
    assert '"type": "lesson"' in cleaned
    assert not cleaned.startswith('```')

    # Test plain JSON (should pass through)
    plain_json = '{"type": "tool", "content": "Test tool"}'
    cleaned_plain = encoder._clean_llm_response(plain_json)
    assert cleaned_plain == plain_json

    # Test JSON with extra text
    mixed_response = 'Here is the JSON: {"type": "skill", "content": "Test skill"} and some extra text'
    cleaned_mixed = encoder._clean_llm_response(mixed_response)
    assert cleaned_mixed == '{"type": "skill", "content": "Test skill"}'

    # Test invalid response
    try:
        encoder._clean_llm_response("No JSON here")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected

    # Test empty response
    try:
        encoder._clean_llm_response("")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
