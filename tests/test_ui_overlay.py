"""
Unit tests for PyQt6 HUD Overlay and HUDBridge components.
"""

import sys
import pytest
from PyQt6.QtWidgets import QApplication
from src.ui.overlay import GlassmorphismOverlay, WaveformVisualizer, ActionBadge


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_waveform_visualizer(qapp):
    vis = WaveformVisualizer()
    vis.set_level(0.75)
    assert vis.target_level == 0.75
    # Trigger animation frame update
    vis._animate_step()
    assert len(vis.current_levels) == 5


def test_action_badge(qapp):
    badge = ActionBadge("⚡ Opened Spotify", urgency=1)
    assert badge.label.text() == "⚡ Opened Spotify"

    crit_badge = ActionBadge("🔒 Locked", urgency=2)
    assert crit_badge.label.text() == "🔒 Locked"


def test_overlay_creation_and_bridge(qapp):
    overlay = GlassmorphismOverlay()
    assert overlay.windowOpacity() >= 0.90

    # Test bridge signals
    overlay.bridge.transcript_signal.emit("Full sentence active phrase", "active phrase")
    assert "active phrase" in overlay.transcript_label.text()

    overlay.bridge.audio_level_signal.emit(0.85)
    assert overlay.visualizer.target_level == 0.85

    overlay.bridge.action_signal.emit("🔊 Vol +10%", 1, "volume_up")
    assert overlay.badge_container.count() > 0

    # Test clear
    overlay._clear_badges()
    assert overlay.badge_container.count() == 0
    overlay.close()
