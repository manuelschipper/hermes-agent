"""Tests for context-compression gateway notification callback."""

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Fixture (same pattern as tests/test_413_compression.py)
# ---------------------------------------------------------------------------

def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = False
        a.save_trajectories = False
        return a


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_callback_fires(agent):
    """tool_progress_callback is called with _compression event after compress."""
    cb = MagicMock()
    agent.tool_progress_callback = cb
    agent.context_compressor = MagicMock()
    agent.context_compressor.compress.return_value = [{"role": "user", "content": "summary"}]

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "bye"},
    ]
    agent._compress_context(messages, "system prompt", approx_tokens=50000)

    cb.assert_called_once()
    call_args = cb.call_args[0]
    assert call_args[0] == "_compression"
    assert "3 → 1 messages" in call_args[1]


def test_callback_absent(agent):
    """No error when tool_progress_callback is None."""
    agent.tool_progress_callback = None
    agent.context_compressor = MagicMock()
    agent.context_compressor.compress.return_value = [{"role": "user", "content": "summary"}]

    messages = [{"role": "user", "content": "hello"}]
    # Should not raise
    agent._compress_context(messages, "system prompt")


def test_callback_error_logged(agent):
    """Callback errors are caught and logged, compression still completes."""
    cb = MagicMock(side_effect=Exception("boom"))
    agent.tool_progress_callback = cb
    agent.context_compressor = MagicMock()
    agent.context_compressor.compress.return_value = [{"role": "user", "content": "summary"}]

    messages = [{"role": "user", "content": "hello"}]

    with patch("run_agent.logger") as mock_logger:
        result = agent._compress_context(messages, "system prompt", approx_tokens=5000)
        mock_logger.debug.assert_called()
        # Compression still returned a result
        assert result is not None


def test_preview_with_tokens(agent):
    """When approx_tokens is provided, preview includes formatted token count."""
    cb = MagicMock()
    agent.tool_progress_callback = cb
    agent.context_compressor = MagicMock()
    agent.context_compressor.compress.return_value = [{"role": "user", "content": "s"}]

    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    agent._compress_context(messages, "system prompt", approx_tokens=100000)

    preview = cb.call_args[0][1]
    assert "~100,000 tokens" in preview
    assert "2 → 1 messages" in preview


def test_preview_without_tokens(agent):
    """When approx_tokens is None, preview omits token info."""
    cb = MagicMock()
    agent.tool_progress_callback = cb
    agent.context_compressor = MagicMock()
    agent.context_compressor.compress.return_value = [{"role": "user", "content": "s"}]

    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    agent._compress_context(messages, "system prompt", approx_tokens=None)

    preview = cb.call_args[0][1]
    assert "tokens" not in preview
    assert "2 → 1 messages" in preview
