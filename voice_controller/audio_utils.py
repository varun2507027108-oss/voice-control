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
