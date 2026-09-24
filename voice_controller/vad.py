"""Silero Neural Voice Activity Detection (VAD) module.

Provides low-latency, neural-network-backed speech boundary detection running
fully offline via ONNX runtime on CPU (~1ms per 32ms frame).
"""

import logging
import collections
import time
from typing import Optional
import numpy as np
import torch

from voice_controller.config import (
    vad_tracker,
    VAD_ONSET_THRESHOLD,
    VAD_HANGOVER_THRESHOLD,
)

logger = logging.getLogger("VoiceController.VAD")


class SileroVAD:
    """Offline neural Voice Activity Detector using Silero VAD ONNX.
    
    Processes 16kHz mono audio in 512-sample (32ms) frames:
      - Speech onset: 2 consecutive frames with probability > onset_threshold (0.30)
      - Speech continuation / hangover: frames with probability >= hangover_threshold (0.20)
    """

    def __init__(
        self,
        onset_threshold: float = VAD_ONSET_THRESHOLD,
        hangover_threshold: float = VAD_HANGOVER_THRESHOLD,
        hud=None,
    ):
        self.onset_threshold = onset_threshold
        self.hangover_threshold = hangover_threshold
        self.hud = hud
        self.model = None
        self.buffer = np.array([], dtype=np.float32)
        self.consecutive_speech_frames = 0
        self.is_speaking = False
        
        # Telemetry & Error Tracking
        self._err_count = 0
        self._frame_probs = collections.deque(maxlen=64)
        self._last_telemetry_time = time.time()
        self.self_test_passed = False
        
        self._init_model()

    def _init_model(self):
        """Load the pre-trained Silero VAD ONNX model and run boot self-test."""
        try:
            from silero_vad import load_silero_vad
            self.model = load_silero_vad(onnx=True)
            logger.info("Silero VAD ONNX model loaded successfully.")
            self.run_self_test()
            self.self_test_passed = True
        except Exception as err:
            logger.critical("Silero VAD initialization or self-test failed: %s", err)
            self.self_test_passed = False
            self.model = None
            if self.hud:
                self.hud.update_state(status="IGNORED", feedback="Voice detector erroring")
            raise

    def run_self_test(self) -> bool:
        """Execute boot self-test using real speech fixture and zero frames.
        
        Raises RuntimeError on failure so system refuses to enter Listening silently.
        """
        if self.model is None:
            raise RuntimeError("Silero VAD model is not loaded.")

        import wave
        from pathlib import Path

        # 1. Test 1s of zeros (16000 samples -> ~31 frames of 512)
        zero_frames = [np.zeros(512, dtype=np.float32) for _ in range(31)]
        zero_probs = []
        try:
            for zf in zero_frames:
                p = float(self.model(torch.from_numpy(zf), 16000)[0, 0])
                zero_probs.append(p)
        except Exception as err:
            logger.critical("Silero VAD self-test zero-frame call failed with exception: %s", err)
            raise RuntimeError(f"Silero VAD self-test failed: zero frame raised {err}") from err

        if any(p >= 0.2 for p in zero_probs):
            max_z = max(zero_probs) if zero_probs else 0.0
            logger.critical("Silero VAD self-test failed on silence: max zero prob=%.4f (expected < 0.2)", max_z)
            raise RuntimeError(f"Silero VAD self-test failed on silence (max prob={max_z:.4f} >= 0.2)")

        # 2. Test committed real-speech fixture
        fixture_paths = [
            Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "open_chrome_16k.wav",
            Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "open_chrome.wav",
        ]
        fixture_path = None
        for fp in fixture_paths:
            if fp.exists():
                fixture_path = fp
                break

        if fixture_path is None:
            logger.critical("Silero VAD self-test speech fixture not found at %s", fixture_paths[0])
            raise RuntimeError(f"Silero VAD self-test fixture missing at {fixture_paths[0]}")

        speech_probs = []
        try:
            with wave.open(str(fixture_path), "rb") as w:
                nframes = w.getnframes()
                audio_bytes = w.readframes(nframes)
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            for i in range(0, len(audio_np) - 512, 512):
                frame = audio_np[i : i + 512]
                p = float(self.model(torch.from_numpy(frame), 16000)[0, 0])
                speech_probs.append(p)
        except Exception as err:
            logger.critical("Silero VAD self-test speech fixture call failed with exception: %s", err)
            raise RuntimeError(f"Silero VAD self-test failed: speech frame raised {err}") from err

        max_p = max(speech_probs) if speech_probs else 0.0
        if max_p <= 0.6:
            logger.critical("Silero VAD self-test failed on speech: max speech prob=%.4f (expected > 0.6)", max_p)
            raise RuntimeError(f"Silero VAD self-test failed on speech fixture (max prob={max_p:.4f} <= 0.6)")

        self.reset()
        logger.info("Silero VAD boot self-test PASS (zero max=%.4f < 0.2, speech max=%.4f > 0.6).", max(zero_probs), max_p)
        return True

    def reset(self):
        """Reset internal streaming state, buffer, and RNN hidden state."""
        self.buffer = np.array([], dtype=np.float32)
        self.consecutive_speech_frames = 0
        self.is_speaking = False
        if self.model is not None and hasattr(self.model, "reset_states"):
            try:
                self.model.reset_states()
            except Exception:
                pass

    def is_speech(self, chunk: np.ndarray, is_speaking: bool = False, hud=None) -> bool:
        """Determine if audio chunk contains speech.
        
        Args:
            chunk: 1D numpy array of float32 samples at 16kHz (typically 800 samples / 50ms).
            is_speaking: Pipeline's current speaking state.
            hud: Optional HUD overlay to push notifications to.
            
        Returns:
            bool: True if chunk contains active speech, False otherwise.
        """
        if hud is not None:
            self.hud = hud

        if self.model is None:
            # Fall back to adaptive energy VAD if model failed to initialize
            return vad_tracker.is_speech(chunk, is_speaking=is_speaking)

        if chunk is None or len(chunk) == 0:
            return False

        # Accumulate samples into buffer
        self.buffer = np.concatenate([self.buffer, chunk.astype(np.float32)])
        speech_frame_detected = False

        # Dual-check with adaptive energy tracker for quiet mic / soft speech assistance
        energy_speech = vad_tracker.is_speech(chunk, is_speaking=is_speaking)

        # Evaluate all available 512-sample (32ms) frames
        while len(self.buffer) >= 512:
            frame = self.buffer[:512]
            self.buffer = self.buffer[512:]

            try:
                prob = float(self.model(torch.from_numpy(frame), 16000)[0, 0])
            except Exception as err:
                self._err_count += 1
                if self._err_count == 1 or self._err_count % 50 == 0:
                    logger.warning(
                        "Silero frame evaluation error #%d: %s (frame shape=%s, dtype=%s)",
                        self._err_count,
                        err,
                        getattr(frame, "shape", None),
                        getattr(frame, "dtype", None),
                    )
                if self._err_count > 100:
                    logger.critical("Silero VAD error count exceeded 100! Voice detector erroring.")
                    if self.hud:
                        self.hud.update_state(status="IGNORED", feedback="Voice detector erroring")
                prob = 0.0

            # Telemetry tracking
            self._frame_probs.append(prob)
            now = time.time()
            if (now - self._last_telemetry_time) >= 2.0 and len(self._frame_probs) > 0:
                probs_list = list(self._frame_probs)
                p50 = float(np.percentile(probs_list, 50))
                p95 = float(np.percentile(probs_list, 95))
                pmax = float(np.max(probs_list))
                logger.info(
                    "VAD telemetry (2s): p50=%.3f p95=%.3f max=%.3f n_frames=%d",
                    p50,
                    p95,
                    pmax,
                    len(probs_list),
                )
                self._last_telemetry_time = now

            if is_speaking:
                # During ongoing speech, hangover threshold prevents clipping intra-sentence pauses
                if prob >= self.hangover_threshold or (energy_speech and prob >= 0.12):
                    speech_frame_detected = True
            else:
                # Trigger onset when consecutive frames exceed onset threshold or 1 frame + energy speech
                if prob > self.onset_threshold or (energy_speech and prob >= 0.18):
                    self.consecutive_speech_frames += 1
                    if self.consecutive_speech_frames >= 2 or (energy_speech and prob >= 0.25):
                        speech_frame_detected = True
                        self.is_speaking = True
                else:
                    self.consecutive_speech_frames = 0

        if is_speaking:
            return speech_frame_detected

        return speech_frame_detected


_global_silero_vad: Optional[SileroVAD] = None


def get_silero_vad(hud=None) -> SileroVAD:
    """Singleton getter for global Silero VAD detector."""
    global _global_silero_vad
    if _global_silero_vad is None:
        _global_silero_vad = SileroVAD(hud=hud)
    elif hud is not None and _global_silero_vad.hud is None:
        _global_silero_vad.hud = hud
    return _global_silero_vad
