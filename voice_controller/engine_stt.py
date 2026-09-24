"""faster-whisper ASR loader and inference handler using CTranslate2 with confidence gates."""

from dataclasses import dataclass
import logging
import os
import re
import sys
import time
from typing import Optional, Tuple
import numpy as np

# Ensure nvidia cuBLAS and cuDNN DLL directories are registered on Windows
if sys.platform == "win32":
    for subpkg in ["cublas", "cudnn"]:
        try:
            import importlib.util
            spec = importlib.util.find_spec(f"nvidia.{subpkg}")
            if spec and spec.submodule_search_locations:
                pkg_path = list(spec.submodule_search_locations)[0]
                bin_path = os.path.join(pkg_path, "bin")
                if os.path.isdir(bin_path):
                    os.add_dll_directory(bin_path)
        except Exception:
            pass

from faster_whisper import WhisperModel

from voice_controller.audio_utils import normalize_level
from voice_controller.config import (
    STT_MODEL_ID,
    DEVICE,
    SAMPLE_RATE,
    STT_TARGET_RMS,
    STT_MAX_GAIN,
    STT_NO_SPEECH_MAX,
    STT_AVG_LOGPROB_MIN,
    STT_COMPRESSION_MAX,
)

logger = logging.getLogger("STTEngine")

_FILLERS = {"you", "uh", "um", "hmm", "the", "a", "an", "i", "me", "it", "ok", "okay", "yeah", "so"}


def is_plausible_speech(text: str) -> bool:
    """Determine whether decoded text represents plausible command speech rather than hallucination/filler."""
    t = (text or "").strip().lower()
    if not t or "♪" in t or "♫" in t:
        return False
    words = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", t)
    return bool(words) and not (len(words) == 1 and words[0] in _FILLERS)


@dataclass(repr=False)
class SttResult(str):
    """Result object from STT transcription containing decoded text and quality metrics."""
    text: str
    trusted: bool
    no_speech_prob: float
    avg_logprob: float
    compression_ratio: float

    def __new__(
        cls,
        text: str,
        trusted: bool,
        no_speech_prob: float,
        avg_logprob: float,
        compression_ratio: float,
    ):
        obj = str.__new__(cls, text)
        obj.text = text
        obj.trusted = bool(trusted)
        obj.no_speech_prob = float(no_speech_prob)
        obj.avg_logprob = float(avg_logprob)
        obj.compression_ratio = float(compression_ratio)
        return obj

    def __repr__(self) -> str:
        return (
            f"SttResult(text='{self.text}', trusted={self.trusted}, "
            f"no_speech_prob={self.no_speech_prob:.4f}, avg_logprob={self.avg_logprob:.4f}, "
            f"compression_ratio={self.compression_ratio:.4f})"
        )


class WhisperSTTEngine:
    """ASR / STT engine utilizing faster-whisper (CTranslate2) on CUDA/CPU."""

    def __init__(
        self,
        model_id: str = STT_MODEL_ID,
        device: str = DEVICE,
        compute_type: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.sr = SAMPLE_RATE
        # Diagnostics for the most recent utterance (used for HUD hinting / logs)
        self.last_input_rms: float = 0.0
        self.last_gain: float = 1.0

        if compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        logger.info(
            "Initializing faster-whisper STT [%s] on %s (%s)...",
            self.model_id,
            self.device,
            self.compute_type,
        )

        try:
            self.model = WhisperModel(
                self.model_id,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as err:
            if self.device == "cuda":
                logger.warning(
                    "CUDA loading failed (%s), falling back to CPU with int8...",
                    err,
                )
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(
                    self.model_id,
                    device="cpu",
                    compute_type="int8",
                )
            else:
                raise

        # Enforce offline mode after initial load to eliminate network requests
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Warm up model to compile CUDA kernels and eliminate first-inference latency
        self._warmup()
        logger.info("faster-whisper STT successfully loaded, warmed up, and ready.")

    def _warmup(self):
        """Execute a quick dummy inference pass to eliminate first-command CUDA kernel compilation lag."""
        try:
            t0 = time.time()
            # 250ms synthetic tone
            t = np.linspace(0, 0.25, self.sr // 4, endpoint=False)
            dummy_audio = (np.sin(2 * np.pi * 440 * t) * 0.2).astype(np.float32)
            _ = self.transcribe(dummy_audio)
            warmup_ms = (time.time() - t0) * 1000
            logger.info("CUDA kernel warmup completed in %.1fms.", warmup_ms)
        except Exception as err:
            logger.warning("Warmup inference pass failed (non-critical): %s", err)

    def transcribe_with_latency(
        self,
        audio_np: np.ndarray,
        noise_floor: Optional[float] = None,
    ) -> Tuple[SttResult, float]:
        """Transcribe audio and return SttResult along with inference latency in ms."""
        t0 = time.time()
        result = self.transcribe(audio_np, noise_floor=noise_floor)
        latency_ms = (time.time() - t0) * 1000.0
        return result, latency_ms

    def transcribe(
        self,
        audio_np: np.ndarray,
        noise_floor: Optional[float] = None,
    ) -> SttResult:
        """Transcribe raw float32 audio sampled at 16kHz with quality metric gating.

        Args:
            audio_np: 1D numpy array of float32 samples at 16,000 Hz.
            noise_floor: Optional ambient noise floor override.

        Returns:
            SttResult dataclass (also subclass of str).
        """
        if audio_np is None or len(audio_np) == 0:
            return SttResult(
                text="",
                trusted=False,
                no_speech_prob=1.0,
                avg_logprob=-99.0,
                compression_ratio=1.0,
            )

        # Ensure single-dimension float32 array
        audio_1d = np.squeeze(audio_np).astype(np.float32)
        if audio_1d.ndim != 1 or len(audio_1d) < int(self.sr * 0.1):
            return SttResult(
                text="",
                trusted=False,
                no_speech_prob=1.0,
                avg_logprob=-99.0,
                compression_ratio=1.0,
            )

        # AC-couple: subtract DC offset from the utterance before feeding STT
        audio_1d = audio_1d - float(np.mean(audio_1d))

        # Demeaned utterance RMS
        demeaned_utt_rms = float(np.sqrt(np.mean(np.square(audio_1d))))
        if demeaned_utt_rms < 1e-5:
            return SttResult(
                text="",
                trusted=False,
                no_speech_prob=1.0,
                avg_logprob=-99.0,
                compression_ratio=1.0,
            )

        # Condition the utterance level: quiet microphone captures (low Windows gain,
        # distant talker, attenuated array capsule) are the single most common cause of
        # Whisper returning empty/low-confidence transcripts, i.e. "it can't hear me".
        audio_1d, applied_gain = normalize_level(
            audio_1d,
            target_rms=STT_TARGET_RMS,
            max_gain=STT_MAX_GAIN,
        )
        self.last_input_rms = demeaned_utt_rms
        self.last_gain = applied_gain
        if applied_gain > 1.5:
            logger.info(
                "Quiet capture boosted: input rms=%.4f -> gain x%.1f (target rms=%.3f)",
                demeaned_utt_rms,
                applied_gain,
                STT_TARGET_RMS,
            )

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            segments, info = self.model.transcribe(
                audio_1d,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                without_timestamps=True,
                word_timestamps=False,
                vad_filter=False,
                no_speech_threshold=STT_NO_SPEECH_MAX,
                log_prob_threshold=STT_AVG_LOGPROB_MIN,
                compression_ratio_threshold=STT_COMPRESSION_MAX,
            )
            seg_list = list(segments)
        except Exception as err:
            logger.error("Error during faster-whisper transcription: %s", err, exc_info=True)
            return SttResult(
                text="",
                trusted=False,
                no_speech_prob=1.0,
                avg_logprob=-99.0,
                compression_ratio=1.0,
            )

        if not seg_list:
            return SttResult(
                text="",
                trusted=False,
                no_speech_prob=1.0,
                avg_logprob=-99.0,
                compression_ratio=1.0,
            )

        # Aggregate decoded text
        raw_text = " ".join(s.text.strip() for s in seg_list if s.text.strip()).strip()
        cleaned_text = re.sub(r"^[^\w]+|[^\w]+$", "", raw_text).strip().lower()

        # Worst-case segment metric aggregation
        no_speech_prob = max(s.no_speech_prob for s in seg_list)
        avg_logprob = min(s.avg_logprob for s in seg_list)
        compression_ratio = max(s.compression_ratio for s in seg_list)

        # Trust formula (thresholds configurable in config.py):
        # trusted = bool(text) and is_plausible_speech(text) and no_speech_prob < STT_NO_SPEECH_MAX
        #           and avg_logprob > STT_AVG_LOGPROB_MIN and compression_ratio < STT_COMPRESSION_MAX
        # avg_logprob was relaxed from -0.8 to -1.0: soft/quiet speech is transcribed
        # correctly by Whisper but scores below the old gate, causing false "can't hear you" rejections.
        trusted = (
            bool(cleaned_text)
            and is_plausible_speech(cleaned_text)
            and no_speech_prob < STT_NO_SPEECH_MAX
            and avg_logprob > STT_AVG_LOGPROB_MIN
            and compression_ratio < STT_COMPRESSION_MAX
        )

        return SttResult(
            text=cleaned_text,
            trusted=trusted,
            no_speech_prob=no_speech_prob,
            avg_logprob=avg_logprob,
            compression_ratio=compression_ratio,
        )

    def transcribe_partial(self, audio_np: np.ndarray) -> str:
        """Fast partial transcription for real-time live draft display on HUD."""
        if audio_np is None or len(audio_np) < int(self.sr * 0.25):
            return ""

        audio_1d = np.squeeze(audio_np).astype(np.float32)
        if audio_1d.ndim != 1 or len(audio_1d) < int(self.sr * 0.25):
            return ""

        audio_1d = audio_1d - float(np.mean(audio_1d))
        demeaned_rms = float(np.sqrt(np.mean(np.square(audio_1d))))
        if demeaned_rms < 1e-4:
            return ""

        audio_1d, _ = normalize_level(audio_1d, target_rms=STT_TARGET_RMS, max_gain=STT_MAX_GAIN)

        try:
            segments, _ = self.model.transcribe(
                audio_1d,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                without_timestamps=True,
                word_timestamps=False,
                vad_filter=False,
            )
            raw = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
            return re.sub(r"^[^\w]+|[^\w]+$", "", raw).strip().lower()
        except Exception as err:
            logger.debug("Partial transcription exception (non-critical): %s", err)
            return ""



# Backwards compatibility alias
MoonshineSTTEngine = WhisperSTTEngine

# Singleton instance helper
_stt_instance: Optional[WhisperSTTEngine] = None


def get_stt_engine() -> WhisperSTTEngine:
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = WhisperSTTEngine()
    return _stt_instance


def transcribe(audio_np: np.ndarray) -> SttResult:
    """Convenience function to transcribe audio using global engine instance."""
    return get_stt_engine().transcribe(audio_np)
