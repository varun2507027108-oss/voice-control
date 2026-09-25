"""
Unit tests for mid-sentence stream dispatcher and token deduplication.
"""

from unittest.mock import MagicMock
from src.core.stream_dispatcher import StreamDispatcher
from src.engine.laya_router import LayaRouter


def test_mid_sentence_execution():
    """Verify that a command triggers mid-sentence and doesn't re-fire on sentence completion."""
    mock_action_cb = MagicMock()
    mock_transcript_cb = MagicMock()

    dispatcher = StreamDispatcher(
        router=LayaRouter(),
        action_callback=mock_action_cb,
        transcript_callback=mock_transcript_cb,
        debounce_seconds=0.2,
    )

    # Step 1: User starts speaking (passive intro)
    dispatcher.process_partial("hey I think")
    assert mock_action_cb.call_count == 0

    # Step 2: User says command mid-sentence
    dispatcher.process_partial("hey I think turn it up")
    # Command "turn it up" should trigger!
    assert mock_action_cb.call_count == 1
    badge, urgency, intent = mock_action_cb.call_args[0]
    assert intent == "volume_up"
    assert "Volume" in badge

    # Step 3: User continues speaking in same sentence
    # "hey I think turn it up because it is quiet"
    dispatcher.process_partial("hey I think turn it up because it is quiet")
    # The consumed index was updated to after "turn it up", so the remaining phrase
    # is "because it is quiet" which is passive -> NO duplicate trigger!
    assert mock_action_cb.call_count == 1


def test_consecutive_different_commands():
    """Verify multiple distinct commands in streaming speech."""
    actions = []
    dispatcher = StreamDispatcher(
        router=LayaRouter(),
        action_callback=lambda badge, urg, intent: actions.append(intent),
        debounce_seconds=0.1,
    )

    # Trigger first command
    dispatcher.process_partial("open spotify")
    assert len(actions) == 1
    assert actions[0] == "open_app"

    # User immediately gives another command
    dispatcher.process_partial("open spotify and now volume up")
    assert len(actions) == 2
    assert actions[1] == "volume_up"
