"""
EchoFlux — Real-Time Mid-Sentence Desktop Voice Assistant Powered by Laya AI.
Main application entry point orchestrating threads, signal-slot bindings, and graceful shutdown.
"""

import sys
import os
import signal
import logging
import argparse
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QPixmap, QColor, QPainter

# Add current workspace directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.engine.laya_router import LayaRouter
from src.core.stream_dispatcher import StreamDispatcher
from src.audio.stt_stream import StreamingSTT
from src.ui.overlay import GlassmorphismOverlay

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EchoFlux.Main")


def create_tray_icon(app: QApplication, overlay: GlassmorphismOverlay) -> QSystemTrayIcon:
    """Create minimalist neon tray icon and context menu."""
    pixmap = QPixmap(32, 32)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(0, 229, 255))
    painter.setPen(QColor(255, 255, 255))
    painter.drawEllipse(4, 4, 24, 24)
    painter.end()

    tray = QSystemTrayIcon(QIcon(pixmap), app)
    tray.setToolTip("EchoFlux Voice Assistant")

    menu = QMenu()
    toggle_action = menu.addAction("Toggle HUD")
    toggle_action.triggered.connect(lambda: overlay.setVisible(not overlay.isVisible()))
    menu.addSeparator()
    quit_action = menu.addAction("Exit EchoFlux")
    quit_action.triggered.connect(app.quit)

    tray.setContextMenu(menu)
    tray.show()
    return tray


def main():
    parser = argparse.ArgumentParser(description="EchoFlux Real-Time Voice Assistant")
    parser.add_argument("--model-size", default="tiny.en", help="faster-whisper model size (default: tiny.en)")
    parser.add_argument("--device", default="auto", help="Compute device for STT ('cpu', 'cuda', 'auto')")
    parser.add_argument("--laya-model", default=None, help="Path to Laya ONNX model")
    parser.add_argument("--demo", action="store_true", help="Run simulated interactive test commands")
    args = parser.parse_args()

    logger.info("Initializing EchoFlux Voice Assistant...")

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # 1. Initialize UI Overlay
    overlay = GlassmorphismOverlay()
    overlay.show()

    # 2. System Tray Icon
    tray = None
    if QSystemTrayIcon.isSystemTrayAvailable():
        try:
            tray = create_tray_icon(app, overlay)
        except Exception as e:
            logger.debug("System tray setup skipped: %s", e)

    # 3. Decision Router (Laya AI)
    router = LayaRouter(model_path=args.laya_model)

    # 4. Stream Dispatcher (mid-sentence consumer & action dispatcher)
    dispatcher = StreamDispatcher(
        router=router,
        action_callback=lambda badge, urgency, intent: overlay.bridge.action_signal.emit(badge, urgency, intent),
        transcript_callback=lambda full, active: overlay.bridge.transcript_signal.emit(full, active),
        level_callback=lambda level: overlay.bridge.audio_level_signal.emit(level),
    )

    # 5. Streaming STT Engine
    stt = StreamingSTT(
        model_size=args.model_size,
        device=args.device,
        on_partial=dispatcher.process_partial,
        on_final=lambda final_text: dispatcher.reset(),
    )

    # 6. Graceful shutdown handler
    def handle_exit(signum=None, frame=None):
        logger.info("EchoFlux shutting down...")
        stt.stop()
        overlay.close()
        app.quit()

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Periodic timer to allow Python signal handling while Qt event loop is active
    sig_timer = QTimer()
    sig_timer.timeout.connect(lambda: None)
    sig_timer.start(200)

    # Start audio worker threads
    stt.start()
    logger.info("EchoFlux is live! Listening for voice commands...")

    # Optional simulated demo for testing mid-sentence triggers
    if args.demo:
        logger.info("Demo mode active: scheduling test voice sequences...")
        demo_sequences = [
            ("I was thinking of listening to some music so please open spotify right now", 1500),
            ("Hey can you turn it up a bit thanks", 4000),
            ("Also open github.com in a new tab for me", 6500),
            ("Now pause music please", 9000),
            ("Take a screenshot", 11500),
        ]
        for phrase, delay_ms in demo_sequences:
            def trigger_sim(p=phrase):
                logger.info("[Simulated Speech Input]: '%s'", p)
                dispatcher.process_partial(p, rms_level=0.65)
            QTimer.singleShot(delay_ms, trigger_sim)

    # Execute Qt event loop
    ret = app.exec()
    stt.stop()
    sys.exit(ret)


if __name__ == "__main__":
    main()
