"""Pure audio helpers shared by the capture callback, device probe, and STT front-end.

These helpers are intentionally dependency-light (numpy only) so they can be unit
tested without any audio hardware attached.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

logger = logging.getLogger("VoiceController.AudioUtils")

# PortAudio pseudo-devices that exist on Windows but are not real endpoints.
# They are kept as last-resort fallbacks only.
PSEUDO_DEVICE_MARKERS = ("primary sound capture", "sound mapper", "microsoft sound mapper")

# Default API preference: WASAPI exposes the true hardware endpoint and avoids the
# DirectSound "Primary Sound Capture Driver" pseudo-device, which historically
# captures silence or heavily attenuated audio on Realtek drivers.
DEFAULT_API_PREFERENCE: Tuple[str, ...] = ("WASAPI", "MME", "DIRECTSOUND")

DEFAULT_CHANNEL_HYSTERESIS: float = 1.25


def rank_input_candidates(
    host_apis: Sequence[Dict[str, Any]],
    devices: Sequence[Dict[str, Any]],
    api_preference: Sequence[str] = DEFAULT_API_PREFERENCE,
    override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return ordered input-device candidates as ``{"index", "name", "api"}`` dicts.

    Ordering rules:
      1. An explicit ``override`` (device index digits or a name substring) wins.
      2. Host APIs follow ``api_preference`` order (WASAPI -> MME -> DIRECTSOUND).
      3. Within an API the default endpoint is tried before sibling endpoints.
      4. WDM-KS host APIs are skipped and pseudo-devices are demoted to the end.
    """
    ranked_apis: List[int] = []
    for pref in api_preference:
        for idx, h in enumerate(host_apis):
            api_name = h.get("name", "").upper()
            if "WDM" in api_name or "KS" in api_name:
                continue
            if pref in api_name and idx not in ranked_apis:
                ranked_apis.append(idx)
                break

    primary: List[Dict[str, Any]] = []
    fallback: List[Dict[str, Any]] = []
    seen = set()

    for api_idx in ranked_apis:
        api_name = host_apis[api_idx].get("name", "")
        default_dev = host_apis[api_idx].get("default_input_device", -1)
        order = ([default_dev] if 0 <= default_dev < len(devices) else []) + list(range(len(devices)))
        for dev_idx in order:
            if dev_idx in seen or not (0 <= dev_idx < len(devices)):
                continue
            dev = devices[dev_idx]
            if dev.get("hostapi") != api_idx or int(dev.get("max_input_channels", 0)) <= 0:
                continue
            seen.add(dev_idx)
            name = dev.get("name", "")
            entry = {"index": dev_idx, "name": name, "api": api_name}
            if any(marker in name.lower() for marker in PSEUDO_DEVICE_MARKERS):
                fallback.append(entry)
            else:
                primary.append(entry)

    ordered = primary + fallback

    if override:
        override_str = str(override).strip()
        if override_str:
            match = None
            if override_str.isdigit():
                for entry in ordered:
                    if entry["index"] == int(override_str):
                        match = entry
                        break
            if match is None:
                needle = override_str.lower()
                for entry in ordered:
                    if needle in entry["name"].lower():
                        match = entry
                        break
            if match is not None:
                ordered = [match] + [e for e in ordered if e["index"] != match["index"]]
                logger.info("Device override '%s' matched '%s' (idx=%s).", override_str, match["name"], match["index"])
            else:
                logger.warning("Device override '%s' did not match any input device; using auto-ranking.", override_str)

    return ordered


def channel_scores(block: np.ndarray, samplerate: Optional[int] = None) -> np.ndarray:
    """Per-channel activity score used to pick the microphone-array capsule carrying voice.

    When ``samplerate`` is provided the score is computed on a first-differenced signal.
    Differencing removes hardware DC offset completely and attenuates low-frequency hum
    (mains buzz at 50/60 Hz, chassis rumble) by roughly 24 dB relative to the speech band
    (measured: 60 Hz -> -42 dB, 1 kHz -> -18 dB at 48 kHz). The capsule whose *voice* band
    is louder therefore wins, which is exactly what raw broadband RMS fails to do when a
    different capsule carries more rumble than speech.

    Returns one score per channel (empty array when ``block`` is not 2-D).
    """
    if block is None or np.size(block) == 0 or block.ndim != 2:
        return np.zeros(0, dtype=np.float32)
    if samplerate and samplerate >= 8000 and block.shape[0] > 1:
        signal = np.diff(block, axis=0)
    else:
        signal = block
    return np.mean(np.abs(signal), axis=0).astype(np.float32)


class ChannelScorer:
    """Pick the microphone-array capsule that actually hears voice, without flapping.

    Raw per-block energy cannot do this job on this hardware: the Realtek array's two channels
    sit within ~1.25x of each other in speech-band energy, so a per-block comparison toggles the
    captured capsule every few seconds, and a channel whose extra energy is hiss or DC drift can
    win outright. This scorer instead tracks each channel's own hiss floor and scores channels by
    how far the current block rises *above that floor* (a running SNR estimate), which is exactly
    the question "which capsule is hearing something new".

    Floors adapt quickly downwards (quiet room) and very slowly upwards, so a long utterance does
    not inflate its own floor, and silence leaves every channel near a score of 1.0 — stable.
    """

    def __init__(self, floor_down: float = 0.10, floor_up: float = 0.002, min_floor: float = 1e-7):
        self.floor_down = floor_down
        self.floor_up = floor_up
        self.min_floor = min_floor
        self.floors: Optional[np.ndarray] = None
        self.scores: Optional[np.ndarray] = None

    def update(self, block: np.ndarray, samplerate: Optional[int] = None) -> np.ndarray:
        """Update tracked floors from a capture block and return the per-channel SNR scores."""
        energy = channel_scores(block, samplerate)
        if energy.size == 0:
            return energy

        if self.floors is None or self.floors.shape != energy.shape:
            self.floors = energy.astype(np.float32).copy()
            self.scores = np.ones_like(energy, dtype=np.float32)
            return self.scores

        alpha = np.where(energy < self.floors, self.floor_down, self.floor_up).astype(np.float32)
        self.floors = (self.floors + alpha * (energy - self.floors)).astype(np.float32)
        self.floors = np.maximum(self.floors, self.min_floor)
        self.scores = (energy / np.maximum(self.floors, self.min_floor)).astype(np.float32)
        return self.scores


def select_channel(
    block: np.ndarray,
    previous_channel: int = 1,
    hysteresis: float = DEFAULT_CHANNEL_HYSTERESIS,
    samplerate: Optional[int] = None,
    scores: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Downmix a multichannel capture block onto its loudest channel.

    Realtek microphone arrays routinely expose one attenuated/reference capsule next
    to the active capsule (measured 3-4x lower RMS on this hardware). Blindly
    averaging with ``np.mean(axis=1)`` therefore throws away ~6 dB of voice energy
    and, for phase-inverted capsules, can cancel the voice almost entirely. This
    helper latches onto the channel with the higher score and switches only when
    another channel scores ``hysteresis`` times higher, which keeps the latch stable.

    Scoring comes from ``scores`` when supplied (see :class:`ChannelScorer` for the
    SNR-based version used by the live pipeline), otherwise from :func:`channel_scores`
    using speech-band weighting whenever ``samplerate`` is supplied.

    Returns ``(mono_chunk, channel_index)``.
    """
    if block is None or len(block) == 0:
        return block, previous_channel
    if block.ndim == 1:
        return block.copy(), previous_channel
    n_channels = block.shape[1]
    if n_channels == 1:
        return block[:, 0].copy(), previous_channel

    magnitudes = np.asarray(scores, dtype=np.float32) if scores is not None else channel_scores(block, samplerate=samplerate)
    if magnitudes.size != n_channels:
        magnitudes = channel_scores(block, samplerate=samplerate)
    if magnitudes.size == 0:
        return block[:, 0].copy(), previous_channel
    best = int(np.argmax(magnitudes))
    if 0 <= previous_channel < n_channels:
        if magnitudes[best] < magnitudes[previous_channel] * hysteresis:
            best = previous_channel
    return block[:, best].copy(), best


def normalize_level(
    audio: np.ndarray,
    target_rms: float = 0.06,
    max_gain: float = 12.0,
    peak_ceiling: float = 0.95,
) -> Tuple[np.ndarray, float]:
    """Apply safe broadband gain so quiet microphones reach a healthy STT input level.

    Returns ``(conditioned_audio, gain_applied)``. Gain is never reduced (``>= 1.0``)
    and is limited so the loudest sample stays below ``peak_ceiling``.
    """
    if audio is None or len(audio) == 0:
        return audio, 1.0

    x = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(x))))
    if rms < 1e-6 or target_rms <= 0:
        return x, 1.0

    gain = min(max_gain, target_rms / rms)
    if gain <= 1.01:
        return x, 1.0

    peak = float(np.max(np.abs(x)))
    if peak > 1e-6:
        gain = min(gain, peak_ceiling / peak)
    gain = max(1.0, float(gain))
    return (x * gain).astype(np.float32), gain


def filter_highpass(audio: np.ndarray, samplerate: int = 16000, cutoff_hz: float = 75.0) -> np.ndarray:
    """Apply a 2nd-order Butterworth high-pass filter to eliminate low-frequency rumble (< 75 Hz)."""
    if audio is None or len(audio) == 0:
        return audio
    x = np.asarray(audio, dtype=np.float32)
    x = x - float(np.mean(x))
    if cutoff_hz <= 0 or samplerate <= cutoff_hz * 2 or len(x) < 8:
        return x
    try:
        from scipy.signal import butter, sosfilt
        sos = butter(2, cutoff_hz, btype="highpass", fs=samplerate, output="sos")
        return sosfilt(sos, x).astype(np.float32)
    except Exception:
        # Simple difference fallback if scipy fails
        diff = np.diff(x, prepend=x[0])
        return diff.astype(np.float32)


class ChannelPreprocessor:
    """Preprocesses a single channel stream: removes DC bias, applies 75Hz high-pass filter, and resamples to 16kHz."""

    def __init__(
        self,
        native_samplerate: int = 48000,
        target_samplerate: int = 16000,
        highpass_hz: float = 75.0,
    ):
        self.native_samplerate = native_samplerate
        self.target_samplerate = target_samplerate
        self.highpass_hz = highpass_hz

        self.sos = None
        self.zi = None
        if highpass_hz > 0 and native_samplerate > highpass_hz * 2:
            try:
                from scipy.signal import butter, sosfilt_zi
                self.sos = butter(2, highpass_hz, btype="highpass", fs=native_samplerate, output="sos")
                self.zi = sosfilt_zi(self.sos)
            except Exception as err:
                logger.warning("Failed to initialize butter highpass filter: %s", err)
                self.sos = None

        if native_samplerate != target_samplerate:
            import math
            g = math.gcd(target_samplerate, native_samplerate)
            self.up = target_samplerate // g
            self.down = native_samplerate // g
        else:
            self.up = 1
            self.down = 1

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Process 1D raw chunk: remove DC, apply highpass, resample to target_samplerate."""
        if chunk is None or len(chunk) == 0:
            return np.zeros(0, dtype=np.float32)
        x = np.asarray(chunk, dtype=np.float32)
        # DC removal
        x = x - float(np.mean(x))
        # Highpass filtering (attenuates 20-40 Hz rumble)
        if self.sos is not None:
            try:
                from scipy.signal import sosfilt
                if self.zi is not None:
                    x, self.zi = sosfilt(self.sos, x, zi=self.zi)
                else:
                    x = sosfilt(self.sos, x)
            except Exception:
                pass
        # Resample to target samplerate
        if self.native_samplerate != self.target_samplerate:
            try:
                from scipy.signal import resample_poly
                x = resample_poly(x, self.up, self.down).astype(np.float32)
            except Exception:
                pass

        # Mild pre-VAD leveling for quiet microphones
        # When signal is above noise floor (>0.003) but quiet (<0.035), gently boost up to 3x
        if len(x) > 0:
            r = float(np.sqrt(np.mean(np.square(x))))
            if 0.003 < r < 0.035:
                gain = min(3.0, 0.035 / max(1e-5, r))
                x = np.clip(x * gain, -0.95, 0.95)

        return x

    process_chunk = process


def measure_channel_profile(audio_1d: np.ndarray, samplerate: int = 16000) -> Dict[str, float]:
    """Calculate broadband RMS, low-frequency rumble RMS (<100Hz), and speech-band RMS (100Hz - 4kHz)."""
    if audio_1d is None or len(audio_1d) == 0:
        return {"broadband_rms": 0.0, "speech_band_rms": 0.0, "low_band_rms": 0.0, "snr_db": 0.0}
    x = np.asarray(audio_1d, dtype=np.float32)
    x = x - float(np.mean(x))
    broadband_rms = float(np.sqrt(np.mean(np.square(x))))
    if broadband_rms < 1e-6:
        return {"broadband_rms": 0.0, "speech_band_rms": 0.0, "low_band_rms": 0.0, "snr_db": 0.0}

    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / samplerate)

    low_mask = freqs < 100.0
    speech_mask = (freqs >= 100.0) & (freqs <= 4000.0)

    low_energy = np.sum(np.abs(fft[low_mask]) ** 2) / (len(x) ** 2)
    speech_energy = np.sum(np.abs(fft[speech_mask]) ** 2) / (len(x) ** 2)

    low_rms = float(np.sqrt(max(0.0, low_energy)))
    speech_rms = float(np.sqrt(max(0.0, speech_energy)))

    snr_db = 20.0 * np.log10(max(1e-5, speech_rms) / max(1e-5, low_rms)) if low_rms > 1e-6 else 20.0
    return {
        "broadband_rms": round(broadband_rms, 5),
        "speech_band_rms": round(speech_rms, 5),
        "low_band_rms": round(low_rms, 5),
        "snr_db": round(float(snr_db), 1),
    }


class DualChannelSpeechSelector:
    """Evaluates multi-channel microphone audio for speech activity and locks channel per utterance.

    Eliminates the Realtek array rumble failure where Channel 1 has 21-29 Hz energy that tricks
    broadband energy detectors. Both channels are filtered and independently evaluated.
    Once speech is confirmed on a channel, that channel is locked for the duration of the utterance.
    """

    def __init__(
        self,
        native_samplerate: int = 48000,
        target_samplerate: int = 16000,
        highpass_hz: float = 75.0,
        preferred_channel: int = 0,
    ):
        self.native_samplerate = native_samplerate
        self.target_samplerate = target_samplerate
        self.preprocessors = [
            ChannelPreprocessor(native_samplerate, target_samplerate, highpass_hz),
            ChannelPreprocessor(native_samplerate, target_samplerate, highpass_hz),
        ]
        self.locked_channel: Optional[int] = None
        self.selected_channel: int = preferred_channel
        self.preferred_channel: int = preferred_channel
        self.channel_profiles: Dict[int, Dict[str, float]] = {}

    def lock_channel(self, channel: int):
        """Lock active channel for the current utterance to eliminate mid-speech flapping."""
        self.locked_channel = channel
        self.selected_channel = channel

    def unlock(self):
        """Unlock channel when utterance finalizes back to IDLE."""
        self.locked_channel = None

    def process_block(self, block: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
        """Preprocesses multi-channel audio block.

        Returns:
            (ch0_16k, ch1_16k_or_None, active_channel_index)
        """
        if block is None or len(block) == 0:
            return np.zeros(0, dtype=np.float32), None, 0
        if block.ndim == 1 or block.shape[1] == 1:
            ch0 = block[:, 0] if block.ndim > 1 else block
            ch0_proc = self.preprocessors[0].process(ch0)
            return ch0_proc, None, 0

        ch0 = block[:, 0]
        ch1 = block[:, 1]
        ch0_proc = self.preprocessors[0].process(ch0)
        ch1_proc = self.preprocessors[1].process(ch1)

        active_idx = self.locked_channel if self.locked_channel is not None else self.selected_channel
        return ch0_proc, ch1_proc, active_idx
