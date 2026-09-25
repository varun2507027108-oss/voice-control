"""
PyQt6 Glassmorphism Floating HUD Overlay for EchoFlux.
Features:
- Frameless, translucent, always-on-top window
- Dynamic animated waveform/visualizer bars driven by RMS audio energy
- Real-time rolling transcription display
- Animated transient action chips (e.g., ⚡ Opened Chrome, 🔊 Vol 60%)
- Auto-fadeout after 3 seconds of silence to 20% opacity; springs to 95% on speech resume.
"""

import sys
import math
from typing import Optional, List
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, QPropertyAnimation, QEasingCurve, pyqtSignal, QObject
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient, QPainterPath
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QFrame,
    QGraphicsDropShadowEffect,
)


class HUDBridge(QObject):
    """Thread-safe signal bridge for updating HUD from background worker threads."""
    transcript_signal = pyqtSignal(str, str)     # full_transcript, active_phrase
    audio_level_signal = pyqtSignal(float)       # normalized rms [0.0, 1.0]
    action_signal = pyqtSignal(str, int, str)    # badge_text, urgency, intent


class WaveformVisualizer(QWidget):
    """Animated 5-bar audio waveform pill reflecting real-time microphone energy."""

    def __init__(self, parent=None, num_bars=5):
        super().__init__(parent)
        self.num_bars = num_bars
        self.setFixedSize(48, 28)
        self.target_level = 0.0
        self.current_levels = [0.1] * self.num_bars
        self.phase = 0.0

        # Animation timer running at 40 FPS
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate_step)
        self.anim_timer.start(25)

    def set_level(self, level: float):
        """Update target RMS audio energy."""
        self.target_level = max(0.05, min(1.0, level))

    def _animate_step(self):
        """Smoothly interpolate bars toward target energy."""
        self.phase += 0.2
        for i in range(self.num_bars):
            # Staggered wave motion
            mod = 0.3 * math.sin(self.phase + i * 0.9)
            target = max(0.12, min(1.0, self.target_level + mod * self.target_level))
            self.current_levels[i] += (target - self.current_levels[i]) * 0.35
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bar_width = 3.5
        spacing = 4.5
        total_width = self.num_bars * bar_width + (self.num_bars - 1) * spacing
        start_x = (self.width() - total_width) / 2.0
        center_y = self.height() / 2.0

        for i in range(self.num_bars):
            val = self.current_levels[i]
            bar_height = max(4.0, val * (self.height() - 6.0))
            x = start_x + i * (bar_width + spacing)
            y = center_y - bar_height / 2.0

            # Gradient from electric cyan to vibrant violet
            grad = QLinearGradient(x, y, x, y + bar_height)
            grad.setColorAt(0.0, QColor(0, 229, 255, 230))
            grad.setColorAt(1.0, QColor(147, 51, 234, 230))

            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(x, y, bar_width, bar_height), 2.0, 2.0)


class ActionBadge(QFrame):
    """Accented transient pill chip displayed upon executing an action."""

    def __init__(self, text: str, urgency: int = 1, parent=None):
        super().__init__(parent)
        self.setObjectName("ActionBadge")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(6)

        self.label = QLabel(text, self)
        font = QFont("Segoe UI", 10, QFont.Weight.DemiBold)
        self.label.setFont(font)
        layout.addWidget(self.label)

        # Style based on urgency
        if urgency >= 2:
            # Critical / Red / Amber
            bg_color = "rgba(239, 68, 68, 0.35)"
            border_color = "rgba(248, 113, 113, 0.75)"
            text_color = "#FECACA"
        else:
            # Normal action: Cyan / Emerald
            bg_color = "rgba(16, 185, 129, 0.30)"
            border_color = "rgba(52, 211, 153, 0.70)"
            text_color = "#A7F3D0"

        self.setStyleSheet(f"""
            QFrame#ActionBadge {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 12px;
            }}
            QLabel {{
                color: {text_color};
                background: transparent;
            }}
        """)


class GlassmorphismOverlay(QWidget):
    """
    Always-on-top, frameless, translucent HUD bar.
    Fades out to 20% on silence and springs back to 95% on voice.
    """

    def __init__(self):
        super().__init__()
        self.bridge = HUDBridge()

        # Connect signals
        self.bridge.transcript_signal.connect(self.update_transcript)
        self.bridge.audio_level_signal.connect(self.update_audio_level)
        self.bridge.action_signal.connect(self.display_action_badge)

        self._init_window()
        self._init_ui()
        self._init_fade_animation()

    def _init_window(self):
        """Configure frameless, translucent, always-on-top window flags."""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.active_opacity = 0.95
        self.idle_opacity = 0.20
        self.setWindowOpacity(self.active_opacity)

    def _init_ui(self):
        """Construct glassmorphism container and widgets."""
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(15, 15, 15, 15)

        # Glass Container Frame
        self.glass_frame = QFrame(self)
        self.glass_frame.setObjectName("GlassFrame")
        self.glass_frame.setStyleSheet("""
            QFrame#GlassFrame {
                background-color: rgba(15, 18, 28, 0.85);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 20px;
            }
        """)

        # Drop shadow for float aesthetic
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(0, 6)
        self.glass_frame.setGraphicsEffect(shadow)

        frame_layout = QHBoxLayout(self.glass_frame)
        frame_layout.setContentsMargins(16, 8, 16, 8)
        frame_layout.setSpacing(12)

        # Left: Audio Visualizer
        self.visualizer = WaveformVisualizer(self.glass_frame)
        frame_layout.addWidget(self.visualizer)

        # Center: Live Transcription Text
        self.transcript_label = QLabel("EchoFlux Listening...", self.glass_frame)
        font = QFont("Segoe UI", 11)
        self.transcript_label.setFont(font)
        self.transcript_label.setStyleSheet("color: rgba(240, 246, 252, 0.92); background: transparent;")
        self.transcript_label.setMinimumWidth(260)
        frame_layout.addWidget(self.transcript_label, stretch=1)

        # Right: Action Badge Container
        self.badge_container = QHBoxLayout()
        self.badge_container.setSpacing(8)
        frame_layout.addLayout(self.badge_container)

        root_layout.addWidget(self.glass_frame)
        self.adjustSize()
        self._position_on_screen()

        # Timer to clear action badges
        self.badge_timer = QTimer(self)
        self.badge_timer.setSingleShot(True)
        self.badge_timer.timeout.connect(self._clear_badges)

    def _position_on_screen(self):
        """Center the HUD at the bottom of the primary display."""
        screen = QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        w = max(520, self.sizeHint().width())
        h = max(70, self.sizeHint().height())
        x = int(geo.x() + (geo.width() - w) / 2)
        y = int(geo.y() + geo.height() - h - 45)  # 45px above taskbar
        self.setGeometry(x, y, w, h)

    def _init_fade_animation(self):
        """Timer and smooth property animation for auto-fadeout on silence."""
        self.fade_timer = QTimer(self)
        self.fade_timer.setInterval(3000)  # 3 seconds silence
        self.fade_timer.timeout.connect(self._start_fade_out)
        self.fade_timer.start()

        self.fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self.fade_anim.setDuration(400)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

    def _start_fade_out(self):
        """Animate down to idle opacity."""
        if self.windowOpacity() > self.idle_opacity:
            self.fade_anim.stop()
            self.fade_anim.setStartValue(self.windowOpacity())
            self.fade_anim.setEndValue(self.idle_opacity)
            self.fade_anim.start()

    def _wake_up(self):
        """Immediately wake up to active opacity on speech resume."""
        self.fade_timer.start()
        if self.windowOpacity() < self.active_opacity:
            self.fade_anim.stop()
            self.fade_anim.setStartValue(self.windowOpacity())
            self.fade_anim.setEndValue(self.active_opacity)
            self.fade_anim.start()

    # -------------------------------------------------------------
    # Slots for background thread updates
    # -------------------------------------------------------------
    def update_transcript(self, full_text: str, active_phrase: str):
        """Update live speech caption."""
        if not full_text:
            self.transcript_label.setText("EchoFlux Listening...")
            return

        self._wake_up()
        # Highlight active unconsumed phrase in cyan
        if active_phrase and active_phrase in full_text:
            idx = full_text.rfind(active_phrase)
            consumed = full_text[:idx]
            formatted = f"<span style='color: rgba(160, 174, 192, 0.7);'>{consumed}</span> <b style='color: #00E5FF;'>{active_phrase}</b>"
            self.transcript_label.setText(formatted)
        else:
            self.transcript_label.setText(full_text)

        self._position_on_screen()

    def update_audio_level(self, level: float):
        """Drive audio waveform bars and trigger wake-up if speech detected."""
        self.visualizer.set_level(level)
        if level > 0.15:
            self._wake_up()

    def display_action_badge(self, text: str, urgency: int = 1, intent: str = ""):
        """Render transient action pill next to caption."""
        self._wake_up()
        self._clear_badges()

        badge = ActionBadge(text, urgency=urgency, parent=self.glass_frame)
        self.badge_container.addWidget(badge)
        self._position_on_screen()

        # Auto-remove after 2.6 seconds
        self.badge_timer.start(2600)

    def _clear_badges(self):
        """Remove existing action badges."""
        while self.badge_container.count():
            item = self.badge_container.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._position_on_screen()
