"""Tests for gateway progress_callback rendering — pseudo-event handling."""

import queue


def _make_progress_callback(progress_mode="all"):
    """Build a progress_callback closure matching gateway/run.py logic."""
    progress_queue = queue.Queue()
    last_tool = [None]

    def progress_callback(tool_name: str, preview: str = None, args: dict = None):
        if not progress_queue:
            return

        # Dedup: always for pseudo-events, configurable for regular tools
        if tool_name == last_tool[0] and (progress_mode == "new" or tool_name.startswith("_")):
            return
        last_tool[0] = tool_name

        tool_emojis = {
            "terminal": "\U0001f4bb",
            "web_search": "\U0001f50d",
            "_compression": "\U0001f5dc\ufe0f",
        }
        emoji = tool_emojis.get(tool_name, "\u2699\ufe0f")

        if preview:
            if len(preview) > 80:
                preview = preview[:77] + "..."
            if tool_name.startswith("_"):
                if progress_mode == "verbose":
                    msg = f"{emoji} {preview}"
                else:
                    _friendly = {
                        "_compression": "Session too long. Compressing context\u2026",
                    }
                    msg = f"{emoji} {_friendly.get(tool_name, preview)}"
            else:
                msg = f"{emoji} {tool_name}: \"{preview}\""
        else:
            msg = f"{emoji} {tool_name}..."

        progress_queue.put(msg)

    return progress_callback, progress_queue


def test_pseudo_event_label_hidden():
    """Non-verbose pseudo-events show friendly message, not raw details."""
    cb, q = _make_progress_callback()
    cb("_compression", "9 \u2192 8 messages (~4,406 tokens)")

    msg = q.get_nowait()
    assert "_compression" not in msg
    assert "Session too long. Compressing context\u2026" in msg
    assert "9 \u2192 8 messages" not in msg
    assert msg.startswith("\U0001f5dc\ufe0f")


def test_pseudo_event_verbose_shows_details():
    """Verbose mode shows raw technical preview for pseudo-events."""
    cb, q = _make_progress_callback(progress_mode="verbose")
    cb("_compression", "9 \u2192 8 messages (~4,406 tokens)")

    msg = q.get_nowait()
    assert "9 \u2192 8 messages (~4,406 tokens)" in msg
    assert msg.startswith("\U0001f5dc\ufe0f")


def test_regular_tool_keeps_label():
    """Regular tools still render with tool name and quotes."""
    cb, q = _make_progress_callback()
    cb("web_search", "python tutorials")

    msg = q.get_nowait()
    assert 'web_search: "python tutorials"' in msg


def test_consecutive_pseudo_events_deduped():
    """Consecutive _compression calls collapse into one (all modes)."""
    cb, q = _make_progress_callback(progress_mode="all")
    cb("_compression", "9 \u2192 8 messages (~4,406 tokens)")
    cb("_compression", "8 \u2192 8 messages (~4,302 tokens)")

    assert q.qsize() == 1


def test_pseudo_event_dedup_resets_after_other_tool():
    """Dedup resets when a different tool fires in between."""
    cb, q = _make_progress_callback()
    cb("_compression", "first pass")
    cb("terminal", "ls -la")
    cb("_compression", "second pass")

    assert q.qsize() == 3
