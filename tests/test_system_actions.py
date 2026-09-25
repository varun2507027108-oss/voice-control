"""
Unit tests for system action controller.
"""

from src.actions.system_actions import SystemActions


def test_volume_formatting_and_bounds():
    # Test setting volume bounds
    success, msg = SystemActions.volume_set(75)
    assert "75%" in msg

    # Test clamping
    success_clamp, msg_clamp = SystemActions.volume_set(150)
    assert "100%" in msg_clamp

    success_low, msg_low = SystemActions.volume_set(-20)
    assert "0%" in msg_low


def test_browser_url_formatting():
    # URL parsing
    success, msg = SystemActions.open_browser_tab("github.com")
    assert "Opened" in msg or "github.com" in msg

    # Query search parsing
    success_q, msg_q = SystemActions.open_browser_tab("weather forecast")
    assert "Searched" in msg_q or "weather" in msg_q


def test_app_aliases():
    assert "code" == SystemActions.APP_ALIASES["vscode"]
    assert "spotify" == SystemActions.APP_ALIASES["spotify"]
    assert "chrome" == SystemActions.APP_ALIASES["google chrome"]


def test_execute_dispatcher():
    success, msg = SystemActions.execute("volume_set", {"percentage": 42})
    assert "42%" in msg

    success_noop, msg_noop = SystemActions.execute("noop", {})
    assert success_noop is False
