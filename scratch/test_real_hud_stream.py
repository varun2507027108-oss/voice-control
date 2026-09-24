"""Test real HUD + real Tk mainloop + real audio stream survival for 8 seconds.
Counts audio callbacks across time to verify whether Tk updates kill the PortAudio stream.
"""
import sys
import time
import threading
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from voice_controller.overlay import HUDOverlay
from voice_controller.main import VoiceControllerPipeline

def test_real_hud_stream():
    print("=" * 60)
    print("STEP 2: Real HUD Stream Survival Test (8 seconds)")
    print("=" * 60)

    hud = HUDOverlay()
    pipeline = VoiceControllerPipeline(hud=hud)
    hud.on_close_callback = pipeline.stop

    callback_counts = []
    stream_active_history = []

    started_event = threading.Event()

    def monitor_thread():
        print("[Monitor] Waiting for pipeline to start...")
        if not started_event.wait(timeout=30.0):
            print("[Monitor] ERROR: Pipeline did not start within 30s!")
            hud.root.after(0, hud.close)
            return

        print("[Monitor] Pipeline is active! Monitoring stream for 8 seconds with HUD updates...")
        # Monitor for 8 seconds in 1s intervals
        for sec in range(1, 9):
            time.sleep(1.0)
            cb_cnt = pipeline._callback_count
            active = pipeline.stream.active if pipeline.stream else False
            callback_counts.append(cb_cnt)
            stream_active_history.append(active)
            print(f"  [T+{sec}s] Callbacks: {cb_cnt}, Stream active: {active}")
            
            # Send state updates to simulate HUD interaction from background thread
            if sec == 2:
                hud.update_state(status="PROCESSING", feedback="Testing HUD update...")
            elif sec == 4:
                hud.update_state(status="EXECUTED", feedback="Action confirmed")
            elif sec == 6:
                hud.update_state(status="LISTENING")

        print("\n[Monitor] 8 seconds completed. Closing HUD...")
        # Schedule closing on Tk main thread
        hud.root.after(0, hud.close)

    def startup_worker():
        from unittest.mock import MagicMock
        pipeline.stt_engine = MagicMock()
        pipeline.intent_router = MagicMock()
        pipeline.start()
        print("[Startup] Real audio pipeline stream started.")
        started_event.set()

    t_start = threading.Thread(target=startup_worker, daemon=True)
    t_start.start()

    t_mon = threading.Thread(target=monitor_thread, daemon=True)
    t_mon.start()

    print("[Main] Entering HUD Tkinter mainloop...")
    hud.start()
    print("[Main] Exited HUD Tkinter mainloop.")

    # Verification
    pipeline.stop()
    print("\nTest Summary:")
    print(f"  Callback counts across 8 seconds: {callback_counts}")
    print(f"  Stream active across 8 seconds: {stream_active_history}")

    if len(callback_counts) >= 6 and callback_counts[-1] > callback_counts[0]:
        rate = (callback_counts[-1] - callback_counts[0]) / 7.0
        print(f"  SUCCESS: Stream continuously delivered callbacks (~{rate:.1f} callbacks/sec) through all HUD updates!")
    else:
        print("  FAILURE: Audio callbacks stopped or stream died!")

if __name__ == "__main__":
    test_real_hud_stream()
