"""Configuration settings for Voice Controller pipeline and HUD overlay."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

# Audio stream settings
SAMPLE_RATE: int = 16000
CHANNELS: int = 1
CHUNK_DURATION_MS: int = 50  # Audio buffer step in milliseconds
CHUNK_SIZE: int = int(SAMPLE_RATE * (CHUNK_DURATION_MS / 1000.0))  # 800 samples

# Pre-roll ring buffer & deduplication
PREROLL_CHUNKS: int = 6  # 300ms pre-speech audio buffer (6 * 50ms)
DEDUP_INTERVAL_S: float = 0.7  # Reduced to 0.7s so deliberate rapid repeats are not swallowed
MAX_SPEECH_DURATION_S: float = 6.0  # Force-slice safety watchdog for continuous noise

# VAD timing parameters
SILENCE_DURATION: float = 0.8     # Seconds of continuous silence required to finalize utterance
MIN_SPEECH_DURATION: float = 0.35  # Minimum seconds of speech needed to proceed to STT
VAD_ONSET_THRESHOLD: float = 0.30  # Silero VAD probability threshold for speech start
VAD_HANGOVER_THRESHOLD: float = 0.20  # Silero VAD probability threshold for speech continuation
LIVE_TRANSCRIBE_INTERVAL_S: float = 0.25  # Real-time partial transcription update cadence

# Audio input device selection
INPUT_API_PREFERENCE: Tuple[str, ...] = ("WASAPI", "MME", "DIRECTSOUND")  # WASAPI first: real endpoint, avoids DirectSound pseudo-device silence
DEVICE_OVERRIDE_ENV: str = "VOICE_CONTROL_DEVICE"   # Optional env override, e.g. VOICE_CONTROL_DEVICE=12 or VOICE_CONTROL_DEVICE="Realtek"
CHANNEL_HYSTERESIS: float = 1.25                    # Switch mic-array channel only when another channel is 25% louder
DEVICE_PROBE_SECONDS: float = 0.4                   # Per-candidate liveness sample during boot
DEVICE_PROBE_MIN_RMS: float = 1e-4                  # Below this the endpoint is treated as digital silence

# STT input conditioning & transcript trust gates
STT_TARGET_RMS: float = 0.06        # Quiet-mic normalization target level
STT_MAX_GAIN: float = 12.0          # Hard gain ceiling so noise is not amplified into hallucinations
STT_NO_SPEECH_MAX: float = 0.60     # Reject transcripts above this no-speech probability
STT_AVG_LOGPROB_MIN: float = -1.00  # Relaxed from -0.8: soft/quiet speech was being rejected here
STT_COMPRESSION_MAX: float = 2.40   # Reject repetitive/hallucinated transcripts

# Boot-time microphone-level auto-fix (percent).
# Automatically raises microphone input level to 100% on boot if below 80% to ensure speech recognition works reliably.
AUTO_FIX_MIC_ENV: str = "VOICE_CONTROL_AUTO_FIX_MIC"
AUTO_FIX_MIC_LEVEL: bool = os.environ.get(AUTO_FIX_MIC_ENV, "1").strip().lower() not in ("0", "false", "no")
MIC_LEVEL_TARGET_PCT: float = 100.0
MIC_LEVEL_ENV: str = "VOICE_CONTROL_MIC_LEVEL"
MIC_LEVEL_LOW_PCT: float = 80.0  # Below this the boot check auto-raises or warns

# Microphone calibration persistence
CALIBRATION_FILE_PATH: Path = Path.home() / ".voice_control_mic.json"


# Calibration schema version — bump whenever the RMS or VAD math changes
CALIB_SCHEMA: str = "ac-coupled-v1"


class AdaptiveVAD:
    """Adaptive noise floor tracker and Voice Activity Detector.
    
    Dynamically tracks ambient room noise bidirectionally (fast downward, slow upward)
    and persists calibration profiles keyed by microphone device name.
    """

    def __init__(self, floor_init: float = 0.015, k: float = 2.2, alpha: float = 0.02):
        self.device_name: str = "default"
        self.noise_floor: float = floor_init
        self.k: float = k               # speech = k * noise_floor
        self.alpha_down: float = alpha  # Adaptation rate when environment is quieter
        self.alpha_up: float = 0.002    # Slow drift upward for gradual noise changes (HVAC/fans)
        self.floor_min: float = 0.005   # Lower bound clamp
        self.floor_max: float = 0.080   # Upper bound clamp
        self._load_calibrated_floor()

    def set_device_name(self, device_name: str):
        """Set current audio device and load its specific calibration profile."""
        self.device_name = (device_name or "default").strip()
        self._load_calibrated_floor()

    def _load_calibrated_floor(self):
        """Load persisted calibrated noise floor for current device name with schema validation."""
        if CALIBRATION_FILE_PATH.exists():
            try:
                with open(CALIBRATION_FILE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # Check device-specific record first
                    dev_data = data.get(self.device_name)
                    if isinstance(dev_data, dict):
                        # Schema check: reject stale calibration from older RMS math
                        if dev_data.get("schema") != CALIB_SCHEMA:
                            return
                        saved_floor = dev_data.get("noise_floor")
                        if saved_floor and isinstance(saved_floor, (int, float)):
                            self.noise_floor = float(saved_floor)
                        if "k" in dev_data and isinstance(dev_data["k"], (int, float)):
                            self.k = float(dev_data["k"])
            except Exception:
                pass

    def save_calibrated_floor(self, floor_val: float, device_name: Optional[str] = None):
        """Persist measured noise floor keyed by device name to local user config."""
        dev = (device_name or self.device_name).strip()
        self.noise_floor = max(self.floor_min, min(self.floor_max, float(floor_val)))

        data: Dict[str, Any] = {}
        if CALIBRATION_FILE_PATH.exists():
            try:
                with open(CALIBRATION_FILE_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        data = loaded
            except Exception:
                data = {}

        data[dev] = {
            "schema": CALIB_SCHEMA,
            "noise_floor": round(self.noise_floor, 4),
            "k": round(self.k, 2),
        }

        try:
            with open(CALIBRATION_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def compute_rms(self, chunk: np.ndarray) -> float:
        """Compute AC-coupled (demeaned) root-mean-square amplitude of 1D float32 audio chunk."""
        if chunk is None or len(chunk) == 0:
            return 0.0
        # AC-couple: remove DC bias so hardware DC offset does not inflate energy calculations
        demeaned = chunk.astype(np.float32) - float(np.mean(chunk))
        return float(np.sqrt(np.mean(np.square(demeaned, dtype=np.float32))))

    def get_thresholds(self) -> Tuple[float, float]:
        """Compute speech onset and hangover (continuation) thresholds with dynamic additive margin.

        Returns (onset_threshold, hangover_threshold).
        """
        # Dynamic margin: scales with noise floor, bounded to [0.015, 0.035]
        margin_onset = max(0.015, min(0.035, (self.k - 1.0) * self.noise_floor))
        onset_threshold = self.noise_floor + margin_onset
        # Hangover threshold uses half the margin to prevent clipping during brief intra-phrase pauses
        margin_hangover = margin_onset * 0.50
        hangover_threshold = self.noise_floor + margin_hangover
        return onset_threshold, hangover_threshold

    def is_speech(self, chunk: np.ndarray, is_speaking: bool = False) -> bool:
        """Determine if audio chunk contains speech using dual-threshold hysteresis and bidirectional adaptation."""
        r = self.compute_rms(chunk)
        onset_th, hangover_th = self.get_thresholds()

        if is_speaking:
            # During ongoing speech, use lower hangover threshold to avoid premature cutoff
            return r > hangover_th
        else:
            # When in silence, adapt baseline noise floor
            if r < self.noise_floor:
                # Environment became quieter: adapt down quickly
                self.noise_floor += self.alpha_down * (r - self.noise_floor)
            elif r < onset_th:
                # Ambient noise slowly increased (fan, AC): adapt up slowly
                # Guaranteed not to adapt during speech bursts because r < onset_th
                self.noise_floor += self.alpha_up * (r - self.noise_floor)

            self.noise_floor = max(self.floor_min, min(self.floor_max, self.noise_floor))
            return r > onset_th

    def adapt_floor(self, r: float):
        """Adapt noise floor on silence chunk."""
        onset_th, _ = self.get_thresholds()
        if r < self.noise_floor:
            self.noise_floor += self.alpha_down * (r - self.noise_floor)
        elif r < onset_th:
            self.noise_floor += self.alpha_up * (r - self.noise_floor)
        self.noise_floor = max(self.floor_min, min(self.floor_max, self.noise_floor))


# Global VAD instance
vad_tracker = AdaptiveVAD()

# AI Model Configuration
STT_MODEL_ID: str = "small.en"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Intent routing confidence threshold
CONFIDENCE_THRESHOLD: float = 0.70

# HUD Overlay Aesthetics (macOS Notepad Theme)
HUD_WIDTH: int = 380
HUD_HEIGHT: int = 295
HUD_ALPHA: float = 0.92
HUD_PADDING_X: int = 24
HUD_PADDING_Y: int = 24

# macOS Dark Palette
COLOR_BG: str = "#1C1C1E"            # macOS Dark System Background
COLOR_PANEL: str = "#2C2C2E"         # macOS Card / Header Toolbar
COLOR_BORDER: str = "#38383A"        # macOS Subtle Border / Divider
COLOR_TEXT_MAIN: str = "#FFFFFF"     # High-contrast clean white
COLOR_TEXT_MUTED: str = "#8E8E93"    # macOS System Gray
COLOR_TEXT_NOTE: str = "#E5E5EA"     # macOS Note Body Text
COLOR_ACCENT: str = "#0A84FF"        # macOS System Blue
COLOR_ACCENT_GOLD: str = "#FFD60A"   # Classic Apple Notes Accent

# Traffic Light Colors (macOS standard)
COLOR_TL_RED: str = "#FF5F56"
COLOR_TL_YELLOW: str = "#FFBD2E"
COLOR_TL_GREEN: str = "#27C93F"
COLOR_TL_BORDER_RED: str = "#E0443E"
COLOR_TL_BORDER_YELLOW: str = "#DEA123"
COLOR_TL_BORDER_GREEN: str = "#1AAB29"

# Status Colors
COLOR_STATUS_LISTENING: str = "#30D158"   # macOS Green
COLOR_STATUS_PROCESSING: str = "#FF9F0A"  # macOS Orange
COLOR_STATUS_EXECUTED: str = "#0A84FF"    # macOS Blue
COLOR_STATUS_IGNORED: str = "#FF453A"     # macOS Red
COLOR_STATUS_OFFLINE: str = "#636366"     # macOS Gray
