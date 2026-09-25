"""
Unit tests for Laya AI decision engine router and typed schema primitives.
"""

import pytest
from src.engine.laya_router import LayaRouter, LayaDecision


@pytest.fixture
def router():
    return LayaRouter()


def test_volume_up_intent(router):
    decision = router.predict("turn it up")
    assert decision.intent == "volume_up"
    assert decision.is_actionable >= 0.82
    assert decision.urgency == 1
    assert decision.should_execute is True


def test_volume_down_intent(router):
    decision = router.predict("lower the volume please")
    assert decision.intent == "volume_down"
    assert decision.is_actionable >= 0.82
    assert decision.urgency == 1


def test_volume_set_percentage(router):
    decision = router.predict("set volume to 65 percent")
    assert decision.intent == "volume_set"
    assert decision.is_actionable >= 0.82
    assert decision.slots.get("percentage") == 65.0


def test_volume_mute_unmute(router):
    d1 = router.predict("mute audio")
    assert d1.intent == "volume_mute"
    assert d1.is_actionable >= 0.82

    d2 = router.predict("unmute sound")
    assert d2.intent == "volume_mute"
    assert d2.is_actionable >= 0.82


def test_open_app_intent(router):
    d1 = router.predict("open Spotify")
    assert d1.intent == "open_app"
    assert d1.slots.get("app_name", "").lower() == "spotify"
    assert d1.should_execute is True

    d2 = router.predict("launch VS Code")
    assert d2.intent == "open_app"
    assert "vs code" in d2.slots.get("app_name", "").lower()


def test_open_browser_tab(router):
    decision = router.predict("open github.com in a new tab")
    assert decision.intent == "open_browser_tab"
    assert "github.com" in decision.slots.get("target", "").lower()

    decision2 = router.predict("search for weather today")
    assert decision2.intent == "open_browser_tab"
    assert "weather" in decision2.slots.get("target", "").lower()


def test_tab_navigation(router):
    assert router.predict("close tab").intent == "close_tab"
    assert router.predict("next tab").intent == "next_tab"
    assert router.predict("previous tab").intent == "prev_tab"


def test_media_controls(router):
    d_pause = router.predict("pause music")
    assert d_pause.intent == "media_play_pause"
    assert d_pause.should_execute is True

    d_skip = router.predict("skip song")
    assert d_skip.intent == "media_next"
    assert d_skip.should_execute is True


def test_take_screenshot(router):
    decision = router.predict("take a screenshot")
    assert decision.intent == "take_screenshot"
    assert decision.is_actionable >= 0.82


def test_lock_workstation_high_urgency(router):
    decision = router.predict("lock workstation")
    assert decision.intent == "lock_workstation"
    assert decision.urgency == 2  # critical urgency score


def test_passive_speech_noop(router):
    decision = router.predict("I was talking to my friend about the movie yesterday")
    assert decision.intent == "noop"
    assert decision.is_actionable < 0.50
    assert decision.should_execute is False
