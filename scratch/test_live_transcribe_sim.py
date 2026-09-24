import time
import numpy as np
import wave
from unittest.mock import MagicMock
from voice_controller.main import VoiceControllerPipeline
from voice_controller.overlay import HUDOverlay
from voice_controller.config import SAMPLE_RATE, CHUNK_SIZE

def test_live_sim():
    print("[TEST] Setting up mock HUDOverlay...")
    mock_hud = MagicMock(spec=HUDOverlay)
    recorded_updates = []
    def record_update(**kwargs):
        recorded_updates.append(kwargs)
    mock_hud.update_state.side_effect = record_update

    pipeline = VoiceControllerPipeline(mock_hud)
    pipeline.is_running = True
    pipeline.stt_engine = MagicMock()
    pipeline.stt_engine.transcribe_partial.return_value = "open chrome"

    # Start live worker
    import threading
    pipeline.live_worker_thread = threading.Thread(target=pipeline._live_worker_loop, daemon=True)
    pipeline.live_worker_thread.start()

    # Generate synthetic speech chunks (amplitude 0.15 > noise floor)
    print("[TEST] Feeding synthetic speech audio chunks...")
    for i in range(12):
        chunk = (np.sin(np.linspace(0, 100, CHUNK_SIZE)) * 0.15).astype(np.float32)
        # Mock VAD to report speech
        pipeline.vad.is_speech = MagicMock(return_value=True)
        pipeline._handle_audio_block(chunk)
        time.sleep(0.06)

    # Wait for live worker to process snapshot
    time.sleep(0.4)

    # Check that live transcription updates were sent to HUD
    live_partials = [u.get("live_transcription") for u in recorded_updates if u.get("live_transcription")]
    print(f"[TEST] Live partial updates recorded: {live_partials}")
    assert len(live_partials) > 0, "No live partial transcription updates were received by HUD!"
    assert "open chrome" in live_partials, "Expected partial transcript 'open chrome' not found!"

    # Feed silence to simulate finish of utterance
    print("[TEST] Feeding silence chunks to finalize utterance...")
    for i in range(15):
        chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
        pipeline.vad.is_speech = MagicMock(return_value=False)
        pipeline._handle_audio_block(chunk)
        time.sleep(0.04)

    # Clean shutdown
    pipeline.stop()
    print("[TEST] Verification SUCCESS: Live partial transcription was dispatched and verified!")

if __name__ == "__main__":
    test_live_sim()
