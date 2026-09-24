"""Silero Neural Voice Activity Detection (VAD) module.

Provides low-latency, neural-network-backed speech boundary detection running
fully offline via ONNX runtime on CPU (~1ms per 32ms frame).
"""

import logging
import collections
import time
from typing import Optional, Tuple
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
        self.model_ch1 = None
        self.buffer = np.array([], dtype=np.float32)
        self.buffer_ch1 = np.array([], dtype=np.float32)
        self.consecutive_speech_frames = 0
        self.consecutive_speech_frames_ch1 = 0
        self.is_speaking = False

        # Telemetry & Error Tracking
        self._err_count = 0
        self._frame_probs = collections.deque(maxlen=64)
        self._last_telemetry_time = time.time()
        self.self_test_passed = False

        self._init_model()

    def _init_model(self):
        """Load the pre-trained Silero VAD ONNX models for both channels and run boot self-test."""
        try:
            from silero_vad import load_silero_vad
            self.model = load_silero_vad(onnx=True)
            try:
                self.model_ch1 = load_silero_vad(onnx=True)
            except Exception as e:
                logger.warning("Could not instantiate secondary Silero channel model: %s", e)
                self.model_ch1 = None
            logger.info("Silero VAD ONNX model(s) loaded successfully.")
            self.run_self_test()
            self.self_test_passed = True
        except Exception as err:
            logger.critical("Silero VAD initialization or self-test failed: %s", err)
            self.self_test_passed = False
            self.model = None
            self.model_ch1 = None
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
        """Reset internal streaming state, buffers, and RNN hidden state for both channels."""
        self.buffer = np.array([], dtype=np.float32)
        self.buffer_ch1 = np.array([], dtype=np.float32)
        self.consecutive_speech_frames = 0
        self.consecutive_speech_frames_ch1 = 0
        self.is_speaking = False
        if self.model is not None and hasattr(self.model, "reset_states"):
            try:
                self.model.reset_states()
            except Exception:
                pass
        if self.model_ch1 is not None and hasattr(self.model_ch1, "reset_states"):
            try:
                self.model_ch1.reset_states()
            except Exception:
                pass

    def get_speech_probability(self, frame: np.ndarray, samplerate: int = 16000) -> float:
        """Evaluate raw 512-sample float32 frame and return speech probability."""
        if self.model is None or frame is None or len(frame) < 512:
            return 0.0
        try:
            tensor = torch.from_numpy(np.asarray(frame[:512], dtype=np.float32))
            return float(self.model(tensor, samplerate)[0, 0])
        except Exception:
            return 0.0

    def evaluate_dual(
        self,
        ch0_chunk: np.ndarray,
        ch1_chunk: Optional[np.ndarray] = None,
        is_speaking: bool = False,
        locked_channel: Optional[int] = None,
        hud=None,
    ) -> Tuple[bool, int, float, float]:
        """Independently evaluate both channels for speech activity.

        Returns:
            (is_speech_detected, active_channel, ch0_max_prob, ch1_max_prob)
        """
        if hud is not None:
            self.hud = hud

        if self.model is None:
            sp0 = vad_tracker.is_speech(ch0_chunk, is_speaking=is_speaking)
            return sp0, 0, (0.8 if sp0 else 0.0), 0.0

        # Evaluate Channel 0
        sp0, p0 = self._evaluate_channel_stream(
            ch0_chunk,
            channel=0,
            model=self.model,
            buffer_attr="buffer",
            consec_attr="consecutive_speech_frames",
            is_speaking=is_speaking,
        )

        # Evaluate Channel 1 if provided
        sp1, p1 = False, 0.0
        if ch1_chunk is not None and len(ch1_chunk) > 0:
            m1 = self.model_ch1 if self.model_ch1 is not None else self.model
            sp1, p1 = self._evaluate_channel_stream(
                ch1_chunk,
                channel=1,
                model=m1,
                buffer_attr="buffer_ch1",
                consec_attr="consecutive_speech_frames_ch1",
                is_speaking=is_speaking,
            )

        if locked_channel is not None:
            # Utterance channel lock active: channel decision is frozen
            active_channel = locked_channel
            speech_active = sp0 if locked_channel == 0 else sp1
            return speech_active, active_channel, p0, p1

        # Channel selection: pick channel containing speech with higher confidence
        if sp0 and not sp1:
            return True, 0, p0, p1
        elif sp1 and not sp0:
            return True, 1, p0, p1
        elif sp0 and sp1:
            best_ch = 0 if p0 >= p1 else 1
            return True, best_ch, p0, p1
        else:
            best_ch = 0 if p0 >= p1 else 1
            return False, best_ch, p0, p1

    def _evaluate_channel_stream(
        self,
        chunk: np.ndarray,
        channel: int,
        model,
        buffer_attr: str,
        consec_attr: str,
        is_speaking: bool,
    ) -> Tuple[bool, float]:
        """Process 512-sample frames for a single channel stream."""
        if chunk is None or len(chunk) == 0:
            return False, 0.0

        buf = getattr(self, buffer_attr)
        buf = np.concatenate([buf, chunk.astype(np.float32)])

        speech_detected = False
        max_prob = 0.0
        consec = getattr(self, consec_attr)

        energy_speech = vad_tracker.is_speech(chunk, is_speaking=is_speaking)

        while len(buf) >= 512:
            frame = buf[:512]
            buf = buf[512:]

            try:
                prob = float(model(torch.from_numpy(frame), 16000)[0, 0])
            except Exception as err:
                self._err_count += 1
                if self._err_count == 1 or self._err_count % 50 == 0:
                    logger.warning("Silero frame evaluation error #%d: %s", self._err_count, err)
                prob = 0.0

            if prob > max_prob:
                max_prob = prob

            if channel == 0:
                self._frame_probs.append(prob)

            if is_speaking:
                if prob >= self.hangover_threshold or (energy_speech and prob >= 0.08):
                    speech_detected = True
            else:
                if prob >= self.onset_threshold or (energy_speech and prob >= 0.12):
                    consec += 1
                    if consec >= 2 or prob >= 0.40 or (energy_speech and prob >= 0.18):
                        speech_detected = True
                else:
                    consec = 0

        setattr(self, buffer_attr, buf)
        setattr(self, consec_attr, consec)

        return speech_detected, max_prob

    def is_speech(self, chunk: np.ndarray, is_speaking: bool = False, hud=None) -> bool:
        """Determine if audio chunk contains speech (Channel 0 compatibility wrapper)."""
        sp, _, _, _ = self.evaluate_dual(chunk, None, is_speaking=is_speaking, hud=hud)
        return sp


_global_silero_vad: Optional[SileroVAD] = None


def get_silero_vad(hud=None) -> SileroVAD:
    """Singleton getter for global Silero VAD detector."""
    global _global_silero_vad
    if _global_silero_vad is None:
        _global_silero_vad = SileroVAD(hud=hud)
    elif hud is not None and _global_silero_vad.hud is None:
        _global_silero_vad.hud = hud
    return _global_silero_vad
