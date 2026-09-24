"""Comprehensive verification test suite for Voice Controller pipeline with mocked hardware actions."""

import os
import sys
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from voice_controller.config import SAMPLE_RATE, AdaptiveVAD
from voice_controller.audio_utils import (ChannelScorer, channel_scores, rank_input_candidates,
                                         select_channel, normalize_level)
from voice_controller.engine_stt import get_stt_engine
from voice_controller.engine_intent import get_intent_router
import voice_controller.actions as actions
from voice_controller.overlay import HUDOverlay


def test_intent_router_classifications():
    """Verify intent classification accuracy and latency across core voice commands using perf_counter."""
    router = get_intent_router()

    test_cases = [
        ("open google chrome", "open_browser"),
        ("launch terminal", "open_terminal"),
        ("open vs code editor", "open_editor"),
        ("turn up the volume", "volume_up"),
        ("louder sound please", "volume_up"),
        ("quieter audio", "volume_down"),
        ("turn down the volume", "volume_down"),
        ("mute audio", "volume_mute"),
        ("silence the sound", "volume_mute"),
        ("unmute audio", "volume_unmute"),
        ("dim the screen", "brightness_down"),
        ("make screen darker", "brightness_down"),
        ("brighten the display", "brightness_up"),
        ("turn up brightness", "brightness_up"),
        ("skip track", "media_next"),
        ("next song", "media_next"),
        ("previous song", "media_prev"),
        ("pause music", "media_play_pause"),
        ("lock my computer", "lock_workstation"),
        ("calibrate the microphone", "calibrate_mic"),
    ]

    # Warmup calls (both fast path and Laya semantic router)
    _ = router.predict("volume up")
    _ = router.predict("louder sound please", bypass_fast_path=True)

    latencies = []
    for utterance, expected_action in test_cases:
        t0 = time.perf_counter()
        res = router.predict(utterance)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(duration_ms)

        action = res["action"]
        conf = res["confidence"]

        assert action == expected_action, f"Utterance '{utterance}' expected '{expected_action}', got '{action}'"
        assert conf >= 0.70, f"Utterance '{utterance}' confidence {conf} below 0.70 threshold"
        print(f"PASS: '{utterance}' -> {action} ({conf*100:.0f}%, {duration_ms:.2f}ms, source={res.get('source')})")

    p50_latency = float(np.percentile(latencies, 50))
    p95_latency = float(np.percentile(latencies, 95))
    print(f"Fast-path Routing Latency — P50: {p50_latency:.2f}ms, P95: {p95_latency:.2f}ms")
    assert p95_latency < 50.0, f"Routing P95 latency ({p95_latency:.2f}ms) exceeded 50ms budget"


def test_laya_pure_latency():
    """Benchmark true Laya semantic inference latency (P50 & P95) with fast-path explicitly bypassed."""
    router = get_intent_router()

    benchmark_cases = [
        ("boost sound output", "volume_up"),
        ("make things quieter", "volume_down"),
        ("darken display screen", "brightness_down"),
        ("increase monitor brightness", "brightness_up"),
        ("advance to next track", "media_next"),
        ("pause the sound track", "media_play_pause"),
        ("secure the desktop workstation", "lock_workstation"),
        ("browse the web online", "open_browser"),
    ]

    latencies = []
    print("\n=== LAYA PURE SEMANTIC BENCHMARK (Fast-path bypassed) ===")
    for utterance, expected_action in benchmark_cases:
        t0 = time.perf_counter()
        res = router.predict(utterance, bypass_fast_path=True)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(duration_ms)

        assert res["source"] == "laya", f"Expected source 'laya', got {res['source']}"
        assert res["action"] == expected_action, f"'{utterance}' expected '{expected_action}', got '{res['action']}'"
        print(f"Laya route: '{utterance}' -> {res['action']} in {duration_ms:.2f}ms (conf={res['confidence']:.2f})")

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    print(f"Laya Semantic Latency Benchmark: P50={p50:.2f}ms, P95={p95:.2f}ms")
    assert p95 < 120.0, f"Laya P95 ({p95:.2f}ms) exceeded 120ms threshold"


def test_slot_extraction():
    """Verify numeric and word-based slot extraction for volume and brightness commands."""
    router = get_intent_router()

    slot_cases = [
        ("set volume to 60%", "volume_set", 60),
        ("volume to 40 percent", "volume_set", 40),
        ("set volume to fifty percent", "volume_set", 50),
        ("set brightness to 80%", "brightness_set", 80),
        ("brightness to twenty", "brightness_set", 20),
        ("open spotify", "open_app", None),
        ("launch calculator", "open_app", None),
    ]

    for utterance, expected_action, expected_val in slot_cases:
        res = router.predict(utterance)
        assert res["action"] == expected_action, f"Expected {expected_action} for '{utterance}', got {res['action']}"
        if expected_val is not None:
            assert res["slot_value"] == expected_val, f"Expected slot {expected_val} for '{utterance}', got {res['slot_value']}"
        print(f"PASS Slot: '{utterance}' -> {res['action']} (slot={res['slot_value']}, target={res.get('target_app')})")


def test_negative_off_domain_cases():
    """Verify false-accept resistance across 30+ off-domain utterances."""
    router = get_intent_router()

    negative_cases = [
        "what is the meaning of life",
        "tell me a funny joke",
        "who is the president of france",
        "what time is it right now",
        "how does quantum computing work",
        "can you write a poem for me",
        "book a flight to new york",
        "order a pizza with pepperoni",
        "where is the closest grocery store",
        "how do i fix a leaky pipe",
        "play chess against the computer",
        "translate hello to japanese",
        "what is the stock price of apple",
        "set a timer for ten minutes",
        "remind me to call mom tomorrow",
        "how many calories in an apple",
        "what is the capital of australia",
        "tell me the latest sports scores",
        "can dogs eat chocolate",
        "write an email to my manager",
        "what is the boiling point of water",
        "solve this math problem for me",
        "how old is the universe",
        "what movies are playing tonight",
        "read my unread text messages",
        "what is the distance to the moon",
        "summarize the news headlines",
        "who invented the lightbulb",
        "sing me a birthday song",
        "how do i bake a chocolate cake",
        "recommend a good book to read",
        "what is photosynthesis",
    ]

    for utterance in negative_cases:
        res = router.predict(utterance)
        assert res["action"] == "unrecognized", f"Off-domain utterance '{utterance}' misclassified as '{res['action']}'"
    print(f"PASS: All {len(negative_cases)} negative off-domain cases successfully rejected as 'unrecognized'.")


def test_adaptive_vad_synthetic_audio():
    """Verify AdaptiveVAD adapts to background noise and detects speech onsets."""
    vad = AdaptiveVAD(floor_init=0.010, k=2.2, alpha=0.05)

    # 1. Background noise chunk (RMS ~0.012)
    noise_chunk = np.random.normal(0, 0.012, 800).astype(np.float32)
    for _ in range(5):
        is_speech = vad.is_speech(noise_chunk)
    assert not is_speech, "Noise chunk should not be detected as speech"
    assert vad.noise_floor > 0.008, "Noise floor should adapt towards noise level"

    # 2. Speech burst chunk (RMS ~0.08)
    speech_chunk = np.random.normal(0, 0.08, 800).astype(np.float32)
    is_speech = vad.is_speech(speech_chunk)
    assert is_speech, "Speech chunk must trigger speech onset detection"
    print("PASS: AdaptiveVAD synthetic noise & speech burst test passed.")


def test_stt_synthetic_audio():
    """Verify Moonshine-tiny ASR processes numpy audio on CUDA FP16 without errors."""
    stt = get_stt_engine()
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    result = stt.transcribe(silence)
    assert isinstance(result, str)
    print(f"STT test on silence passed: output='{result}'")


def test_actions_safe_execution_mocked():
    """Verify action dispatchers execute cleanly WITH HARDWARE ACTIONS FULLY MOCKED.
    
    Guarantees test suite NEVER changes user's physical volume, locks screen, or launches apps.
    """
    mock_action_ok = actions.ActionResult(True, "Success", {})
    with patch.object(actions.VolumeController, "set_volume_percent", return_value=mock_action_ok) as mock_set_vol, \
         patch.object(actions.VolumeController, "set_volume", return_value=mock_action_ok), \
         patch.object(actions.VolumeController, "get_volume", return_value=0.50), \
         patch.object(actions.VolumeController, "adjust_volume", return_value=mock_action_ok), \
         patch.object(actions.VolumeController, "set_mute", return_value=mock_action_ok) as mock_mute, \
         patch("ctypes.windll.user32.keybd_event") as mock_keybd, \
         patch("ctypes.windll.user32.LockWorkStation", return_value=1) as mock_lock, \
         patch("screen_brightness_control.get_brightness", return_value=[65]), \
         patch("screen_brightness_control.set_brightness", return_value=[65]), \
         patch("subprocess.Popen") as mock_popen, \
         patch("webbrowser.open") as mock_browser:

        # 1. Volume tests
        vol_res = actions.action_volume_set(45)
        assert isinstance(vol_res, actions.ActionResult)
        assert vol_res.success
        assert mock_set_vol.called

        mute_res = actions.action_volume_mute()
        assert mute_res.success
        assert mock_mute.called

        unmute_res = actions.action_volume_unmute()
        assert unmute_res.success

        vol_up_res = actions.action_volume_up(10)
        assert vol_up_res.success

        # 2. Media key Win32 injection test
        media_res = actions.action_media_play_pause()
        assert media_res.success
        assert mock_keybd.called

        # 3. Brightness test
        bright_res = actions.action_brightness_up(10)
        assert bright_res.success

        # 4. Lock workstation test (mocked)
        lock_res = actions.action_lock_workstation()
        assert lock_res.success
        assert mock_lock.called

        # 5. App launch test (mocked)
        app_res = actions.action_open_terminal()
        assert app_res.success
        assert mock_popen.called

        print("PASS: All action dispatchers validated safely with hardware mocks.")


def test_hud_overlay_lifecycle():
    """Verify HUD initializes, renders traffic lights, processes state changes, and closes cleanly."""
    hud = HUDOverlay()

    hud.update_state(
        status="LISTENING",
        transcription="Turn up the volume",
        action="volume_up",
        confidence=0.98,
        feedback="Volume set to 60%",
        audio_level=0.75,
        is_loading=False,
    )

    def auto_close():
        for _ in range(50):
            time.sleep(0.1)
            if hud.root and hud._is_running:
                # Verify traffic light buttons exist
                assert hasattr(hud, "btn_close"), "Red traffic light missing"
                assert hasattr(hud, "btn_min"), "Yellow traffic light missing"
                assert hasattr(hud, "btn_reset"), "Green traffic light missing"
                hud.root.after(200, hud.close)
                return

    import threading
    t = threading.Thread(target=auto_close, daemon=True)
    t.start()

    hud.start()
    assert hud.current_status == "LISTENING"
    print("HUD overlay lifecycle and traffic lights test passed.")


def test_transcript_plausibility_gate():
    """Verify that is_plausible_speech rejects fillers, single words, and music tokens."""
    from voice_controller.engine_stt import is_plausible_speech

    # Must be rejected (hallucinations, non-speech, single-word fillers)
    rejected_inputs = [
        "",
        "   ",
        "you",
        "You",
        "YOU",
        "uh",
        "um",
        "hmm",
        "the",
        "a",
        "an",
        "♪",
        "♪♪♪♪♪",
        "♫",
        "♫♫♫",
        "♪♪♪... ",
    ]
    for text in rejected_inputs:
        assert not is_plausible_speech(text), f"Expected rejection for: '{text}'"

    # Must be accepted (valid commands, multi-word or actionable single words)
    accepted_inputs = [
        "open chrome",
        "set volume to 50",
        "volume up",
        "volume down",
        "mute",
        "take screenshot",
        "unmute",
        "50",
    ]
    for text in accepted_inputs:
        assert is_plausible_speech(text), f"Expected acceptance for: '{text}'"

    print("Transcript plausibility gate test passed.")


def test_circuit_breaker_tripwire():
    """Verify action circuit breaker trips after >3 rapid executions in 30 seconds."""
    from voice_controller.actions import reset_circuit_breaker, check_circuit_breaker, execute
    from unittest.mock import patch

    reset_circuit_breaker()

    # 1. Direct check_circuit_breaker
    tripped, _ = check_circuit_breaker("volume_up")
    assert not tripped

    # 2. Mock execution to avoid actual Win32 keystrokes during test
    with patch("voice_controller.actions._send_vk"):
        # Executes 1, 2, 3 should succeed
        res1 = execute("volume_up")
        assert res1.success
        res2 = execute("volume_up")
        assert res2.success
        res3 = execute("volume_up")
        assert res3.success

        # 4th execution should be blocked by circuit breaker
        res4 = execute("volume_up")
        assert not res4.success
        assert "repeated too fast — pausing" in res4.message
        assert res4.data.get("breaker_tripped") is True

    reset_circuit_breaker()
    print("Action circuit breaker test passed.")


def test_ac_coupled_vad_rms():
    """Verify that DC bias is subtracted and does not falsely elevate RMS."""
    from voice_controller.config import vad_tracker
    import numpy as np

    # Synthetic silence with a massive DC offset (+0.05)
    dc_offset = 0.05
    pure_noise = np.random.normal(0, 0.002, 800).astype(np.float32)
    biased_chunk = pure_noise + dc_offset

    raw_rms = float(np.sqrt(np.mean(biased_chunk**2)))
    ac_coupled_rms = vad_tracker.compute_rms(biased_chunk)

    assert raw_rms > 0.045, "Raw RMS should have reflected the high DC bias"
    assert ac_coupled_rms < 0.008, f"AC-coupled RMS should be down near actual noise floor: {ac_coupled_rms}"
    print("AC-coupled VAD RMS test passed.")


def test_real_speech_fixture_end_to_end():
    """Verify STT + intent router end-to-end with the committed speech fixture."""
    import wave
    from pathlib import Path
    from voice_controller.engine_stt import get_stt_engine, is_plausible_speech
    from voice_controller.engine_intent import get_intent_router

    fixture_path = Path(__file__).parent / "fixtures" / "open_chrome.wav"
    assert fixture_path.exists(), f"Missing fixture at {fixture_path}"

    with wave.open(str(fixture_path), "rb") as wf:
        sr = wf.getframerate()
        assert sr == 16000, f"Expected 16000Hz fixture, got {sr}"
        raw_bytes = wf.readframes(wf.getnframes())
        audio_16k = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    stt = get_stt_engine()
    text, latency_ms = stt.transcribe_with_latency(audio_16k, noise_floor=0.005)
    print(f"Fixture STT result in {latency_ms:.1f}ms: '{text}'")

    assert text != "", "STT returned empty transcript for real speech fixture"
    assert is_plausible_speech(text), f"Transcript '{text}' failed plausibility check"

    router = get_intent_router()
    pred = router.predict(text)
    print(f"Fixture Intent result: action='{pred.get('action')}', conf={pred.get('confidence')}")

    assert pred.get("action") == "open_browser", f"Expected open_browser, got: {pred.get('action')}"
    assert pred.get("confidence", 0) >= 0.70, f"Confidence too low: {pred.get('confidence')}"
    print("Real-speech fixture end-to-end test passed.")


def test_silero_vad_offline_detection():
    """Verify Silero VAD detects real speech and rejects ambient noise."""
    import wave
    from pathlib import Path
    from voice_controller.vad import get_silero_vad

    vad = get_silero_vad()
    vad.reset()

    # 1. Reject Gaussian noise
    noise = np.random.normal(0, 0.035, 16000).astype(np.float32)
    chunk_size = 800
    noise_speech_flags = []
    for i in range(0, len(noise), chunk_size):
        chunk = noise[i : i + chunk_size]
        noise_speech_flags.append(vad.is_speech(chunk))
    assert not any(noise_speech_flags), "Silero VAD triggered on ambient Gaussian noise!"

    # 2. Detect real speech fixture
    vad.reset()
    fixture_path = Path(__file__).parent / "fixtures" / "open_chrome.wav"
    with wave.open(str(fixture_path), "rb") as wf:
        raw_bytes = wf.readframes(wf.getnframes())
        audio_16k = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    speech_flags = []
    is_speaking = False
    for i in range(0, len(audio_16k) - chunk_size, chunk_size):
        chunk = audio_16k[i : i + chunk_size]
        detected = vad.is_speech(chunk, is_speaking=is_speaking)
        speech_flags.append(detected)
        if detected:
            is_speaking = True

    assert any(speech_flags), "Silero VAD failed to trigger on real speech fixture!"
    print("Silero VAD offline detection test passed.")


def test_calibration_schema_versioning():
    """Verify that calibration files enforce the current CALIB_SCHEMA and reject stale schemas."""
    from voice_controller.config import CALIB_SCHEMA, CALIBRATION_FILE_PATH, vad_tracker
    import json

    test_dev = "TestDevice_SchemaCheck"
    vad_tracker.save_calibrated_floor(0.0123, device_name=test_dev)

    assert CALIBRATION_FILE_PATH.exists()
    with open(CALIBRATION_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert test_dev in data
    assert data[test_dev].get("schema") == CALIB_SCHEMA
    assert data[test_dev].get("noise_floor") == 0.0123
    print("Calibration schema versioning test passed.")


def test_whisper_real_speech_fixture_e2e():
    """Verify Whisper STT + intent router end-to-end with the committed speech fixture."""
    import wave
    from pathlib import Path
    from voice_controller.engine_stt import get_stt_engine
    from voice_controller.engine_intent import get_intent_router

    fixture_path = Path(__file__).parent / "fixtures" / "open_chrome.wav"
    assert fixture_path.exists(), f"Missing fixture at {fixture_path}"

    with wave.open(str(fixture_path), "rb") as wf:
        raw_bytes = wf.readframes(wf.getnframes())
        audio_16k = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    stt = get_stt_engine()
    result, latency_ms = stt.transcribe_with_latency(audio_16k)
    print(f"Whisper fixture STT result in {latency_ms:.1f}ms: {result}")

    assert result.trusted is True, f"Expected trusted=True, got {result.trusted}"
    assert result.no_speech_prob < 0.1, f"Expected no_speech_prob < 0.1, got {result.no_speech_prob}"
    assert result.avg_logprob > -0.5, f"Expected avg_logprob > -0.5, got {result.avg_logprob}"
    assert result.compression_ratio < 1.5, f"Expected compression_ratio < 1.5, got {result.compression_ratio}"
    assert "chrome" in result.text, f"Expected 'chrome' in text, got '{result.text}'"

    router = get_intent_router()
    pred = router.predict(result.text)
    assert pred.get("action") == "open_browser", f"Expected open_browser, got: {pred.get('action')}"
    print("Whisper real speech fixture E2E test passed.")


def test_whisper_garbage_rejection():
    """Verify that zeros, Gaussian noise, and clicks all produce trusted == False."""
    stt = get_stt_engine()

    # 1. Zeros
    zeros = np.zeros(16000, dtype=np.float32)
    res_zeros = stt.transcribe(zeros)
    assert not res_zeros.trusted, f"Expected trusted=False for zeros, got {res_zeros}"

    # 2. Gaussian noise at ambient level
    noise = np.random.normal(0, 0.035, 16000).astype(np.float32)
    res_noise = stt.transcribe(noise)
    assert not res_noise.trusted, f"Expected trusted=False for noise, got {res_noise}"

    # 3. 100ms click
    click = np.zeros(16000, dtype=np.float32)
    click[8000:8100] = 0.8
    res_click = stt.transcribe(click)
    assert not res_click.trusted, f"Expected trusted=False for click, got {res_click}"

    print("Whisper garbage rejection test passed.")


def test_whisper_cuda_latency_benchmark():
    """Benchmark Whisper transcription latency across 10 iterations on GPU."""
    import wave
    from pathlib import Path
    from voice_controller.engine_stt import get_stt_engine

    fixture_path = Path(__file__).parent / "fixtures" / "open_chrome.wav"
    assert fixture_path.exists(), f"Missing fixture at {fixture_path}"

    with wave.open(str(fixture_path), "rb") as wf:
        raw_bytes = wf.readframes(wf.getnframes())
        audio_16k = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    stt = get_stt_engine()
    print(f"STT instance in benchmark: device={stt.device}, compute_type={stt.compute_type}")
    # Warmup
    _ = stt.transcribe(audio_16k)

    latencies = []
    for idx in range(10):
        t0 = time.perf_counter()
        _ = stt.transcribe(audio_16k)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        print(f"Whisper bench [{idx}]: {dt:.1f}ms")

    p95 = float(np.percentile(latencies, 95))
    print(f"Whisper CUDA Latency P95: {p95:.1f}ms (threshold: 400ms)")
    assert p95 < 400.0, f"Whisper CUDA P95 latency ({p95:.1f}ms) exceeded 400ms budget"


def test_microphone_channel_selection():
    """Verify loudest-capsule channel selection instead of blind stereo averaging.

    Realtek mic arrays expose one attenuated capsule (measured 3-4x lower RMS on the
    target machine); np.mean(axis=1) would discard ~6 dB of voice energy and can
    fully cancel phase-inverted capsules.
    """
    sr = SAMPLE_RATE
    t = np.arange(int(0.5 * sr)) / sr
    voice = (np.sin(2 * np.pi * 220 * t) * 0.2).astype(np.float32)

    # Capsule 1 is 4x louder than capsule 0 -> the loud capsule must be selected
    stereo = np.stack([voice * 0.25, voice], axis=1).astype(np.float32)
    mono, channel = select_channel(stereo, previous_channel=0)
    assert channel == 1, f"Expected loudest channel 1, got {channel}"
    assert abs(np.sqrt(np.mean(np.square(mono))) - np.sqrt(np.mean(np.square(voice)))) < 1e-3

    # Blind mean() would have destroyed ~6dB; prove the helper keeps the energy
    mean_downmix_rms = float(np.sqrt(np.mean(np.square(np.mean(stereo, axis=1)))))
    assert np.sqrt(np.mean(np.square(mono))) > 1.3 * mean_downmix_rms

    # Phase-inverted capsules: mean() would cancel to near-silence, helper must not
    inverted = np.stack([voice, -voice], axis=1).astype(np.float32)
    mono_inv, _ = select_channel(inverted, previous_channel=0)
    assert float(np.max(np.abs(mono_inv))) > 0.15, "Phase-inverted array must not cancel out"

    # Hysteresis: a marginally louder channel must not steal the latch
    latched, channel_keep = select_channel(np.stack([voice, voice * 1.05], axis=1).astype(np.float32), previous_channel=0)
    assert channel_keep == 0, "Channel must only switch when another capsule is clearly louder"

    # Mono input is passed through untouched
    mono_only, ch_mono = select_channel(voice.reshape(-1, 1), previous_channel=0)
    assert ch_mono == 0 and len(mono_only) == len(voice)


def test_speech_band_channel_scoring():
    """Verify capsule scoring prefers the voice-bearing channel over a humming/DC-offset one.

    Broadband RMS alone is fooled by a capsule that carries mains hum or a hardware DC
    offset: that capsule can be 10x "louder" while carrying no usable voice at all.
    Speech-band scoring must pick the voice capsule instead.
    """
    sr = 48000
    t = np.arange(int(0.5 * sr)) / sr
    voice = (np.sin(2 * np.pi * 1000 * t) * 0.05).astype(np.float32)
    hum = (np.sin(2 * np.pi * 60 * t) * 0.30).astype(np.float32)

    hum_block = np.stack([voice, hum], axis=1).astype(np.float32)

    # Broadband scoring is fooled: the 60Hz hum capsule looks 6x louder
    _mono_bb, channel_bb = select_channel(hum_block, previous_channel=0)
    assert channel_bb == 1, "Broadband scoring is expected to be fooled by the hum (documents the bug)"

    # Speech-band scoring picks the channel that actually carries voice
    mono_sb, channel_sb = select_channel(hum_block, previous_channel=1, samplerate=sr)
    assert channel_sb == 0, f"Speech-band scoring must reject the hum capsule, got ch{channel_sb}"
    assert float(np.max(np.abs(mono_sb))) > 0.03, "Selected channel must contain the voice signal"

    # A pure DC offset on the reference capsule must never win either
    dc_block = np.stack([voice, np.full_like(voice, 0.3)], axis=1).astype(np.float32)
    _mono_dc, channel_dc = select_channel(dc_block, previous_channel=1, samplerate=sr)
    assert channel_dc == 0, "A DC-offset capsule must not win speech-band scoring"

    # Scores are only defined for multichannel blocks
    assert channel_scores(voice).size == 0, "1-D input has no per-channel scores"
    scores = channel_scores(np.stack([voice, voice * 3.0], axis=1).astype(np.float32), sr)
    assert len(scores) == 2 and scores[1] > scores[0], "Louder capsule must score higher"


def test_channel_scorer_picks_voice_capsule_without_flapping():
    """Verify SNR-based capsule scoring selects the voice channel and stays latched.

    Measured on the target Realtek array: both channels sit within ~1.25x of each other in
    speech-band energy, so a raw per-block comparison flapped between capsules every few
    seconds. The scorer must instead follow whichever capsule rises above its own hiss floor,
    and must not flap while only ambient noise is present.
    """
    rng = np.random.default_rng(11)
    sr = 48000
    block_len = int(0.05 * sr)
    t = np.arange(block_len) / sr

    def noise(scale):
        return (rng.standard_normal(block_len) * scale).astype(np.float32)

    # ch0 = quiet capsule (little hiss), ch1 = hissy reference capsule with 8x the noise
    scorer = ChannelScorer()
    latched = 0
    switches = 0
    for _ in range(40):
        block = np.stack([noise(0.002), noise(0.016)], axis=1)
        scores = scorer.update(block, sr)
        _mono, channel = select_channel(block, latched, samplerate=sr, scores=scores)
        if channel != latched:
            switches += 1
        latched = channel
    assert switches == 0, f"Latch must not flap during ambient noise (saw {switches} switches)"

    # Voice now arrives ONLY on ch0 and is clearly above its floor
    voice = (np.sin(2 * np.pi * 900 * t) * 0.06).astype(np.float32)
    for _ in range(20):
        block = np.stack([voice + noise(0.002), noise(0.016)], axis=1)
        scores = scorer.update(block, sr)
        _mono, channel = select_channel(block, latched, samplerate=sr, scores=scores)
        latched = channel
    assert latched == 0, "Scorer must latch onto the capsule that carries voice above its hiss"

    # Scores are relative to each channel's own floor: the hissy capsule stays near 1.0
    assert scores[0] > scores[1], f"Voice capsule must score higher (got {scores})"
    assert scorer.floors is not None and scorer.floors[1] > scorer.floors[0], "Hissy channel floor must be higher"


def test_microphone_permissions_reader():
    """Verify Windows privacy-consent reader returns a well-formed dict and never raises."""
    perms = actions.get_microphone_permissions()
    for key in ("available", "global_access", "desktop_apps", "this_app", "machine_policy", "error"):
        assert key in perms, f"Missing key '{key}' in microphone permission report"
    assert isinstance(perms["available"], bool)
    if perms["available"]:
        assert perms["global_access"] in ("Allow", "Deny", "unknown")
        assert perms["desktop_apps"] in ("Allow", "Deny", "unknown")


def test_quiet_audio_level_normalization():
    """Verify safe gain conditioning for quiet captures and no attenuation of loud ones."""
    sr = SAMPLE_RATE
    t = np.arange(sr) / sr

    quiet = (np.sin(2 * np.pi * 300 * t) * 0.004).astype(np.float32)
    boosted, gain = normalize_level(quiet, target_rms=0.06, max_gain=12.0)
    assert gain > 1.0, "Quiet capture should receive gain"
    assert gain <= 12.0, "Gain must respect the configured ceiling"
    assert np.sqrt(np.mean(np.square(boosted))) > np.sqrt(np.mean(np.square(quiet)))
    assert float(np.max(np.abs(boosted))) <= 0.95 + 1e-6, "Boosted audio must not clip"

    loud = (np.sin(2 * np.pi * 300 * t) * 0.5).astype(np.float32)
    unchanged, gain_loud = normalize_level(loud, target_rms=0.06, max_gain=12.0)
    assert gain_loud == 1.0, "Loud captures must never be attenuated"
    assert np.allclose(loud, unchanged)

    silent = np.zeros(sr, dtype=np.float32)
    silent_out, gain_silent = normalize_level(silent, target_rms=0.06, max_gain=12.0)
    assert gain_silent == 1.0 and float(np.max(np.abs(silent_out))) == 0.0, "Silence must stay silent (no noise amplification)"

    # Peak safety: a quiet but spiky signal must be gain-limited below the ceiling
    spiky = quiet.copy()
    spiky[100] = 0.35
    spiky_out, _ = normalize_level(spiky, target_rms=0.06, max_gain=12.0)
    assert float(np.max(np.abs(spiky_out))) <= 0.95 + 1e-6


def test_input_device_ranking_avoids_pseudo_devices():
    """Verify WASAPI-first ranking, pseudo-device demotion, WDM-KS exclusion, and overrides.

    Uses a synthetic device table mirroring the target Windows machine (Realtek array
    exposed through WASAPI, DirectSound, MME plus DirectSound/MME pseudo-devices).
    """
    host_apis = [
        {"name": "MME", "default_input_device": 0},
        {"name": "Windows DirectSound", "default_input_device": 1},
        {"name": "Windows WASAPI", "default_input_device": 2},
        {"name": "Windows WDM-KS", "default_input_device": 3},
    ]
    devices = [
        {"name": "Microphone (2- Realtek(R) Audio)", "hostapi": 0, "max_input_channels": 2},
        {"name": "Primary Sound Capture Driver", "hostapi": 1, "max_input_channels": 2},
        {"name": "Microphone (2- Realtek(R) Audio)", "hostapi": 2, "max_input_channels": 2},
        {"name": "Microphone (Realtek HD Audio Mic input)", "hostapi": 3, "max_input_channels": 2},
        {"name": "Stereo Mix (Realtek HD Audio Stereo input)", "hostapi": 3, "max_input_channels": 2},
    ]

    ranked = rank_input_candidates(host_apis, devices)
    indices = [entry["index"] for entry in ranked]

    assert 2 in indices, "Real WASAPI device must be a candidate"
    assert ranked[0]["index"] == 2, f"WASAPI endpoint must rank first, got {ranked[0]}"
    assert 3 not in indices and 4 not in indices, "WDM-KS endpoints must be excluded"
    assert indices[-1] == 1, f"Pseudo-device must be demoted to last, got {indices}"

    # Explicit override by index
    by_index = rank_input_candidates(host_apis, devices, override="0")
    assert by_index[0]["index"] == 0, "Numeric override must force the requested device"

    # Explicit override by name substring
    by_name = rank_input_candidates(host_apis, devices, override="Realtek(R)")
    assert by_name[0]["index"] == 2, "Name override must match the first eligible endpoint"

    # Unmatched override must degrade gracefully to auto-ranking
    unmatched = rank_input_candidates(host_apis, devices, override="Nonexistent Mic")
    assert unmatched[0]["index"] == 2, "Unmatched override must fall back to auto-ranking"


if __name__ == "__main__":
    print("--- 1. Testing Intent Router (Standard Commands) ---")
    test_intent_router_classifications()

    print("\n--- 2. Testing Laya Pure Latency (Fast-path Bypassed) ---")
    test_laya_pure_latency()

    print("\n--- 3. Testing Slot Extraction (Volume & Brightness) ---")
    test_slot_extraction()

    print("\n--- 4. Testing Negative Off-Domain Utterances (30+ Cases) ---")
    test_negative_off_domain_cases()

    print("\n--- 5. Testing Adaptive VAD ---")
    test_adaptive_vad_synthetic_audio()

    print("\n--- 6. Testing AC-coupled VAD RMS (DC bias immunity) ---")
    test_ac_coupled_vad_rms()

    print("\n--- 7. Testing Transcript Plausibility Gate ---")
    test_transcript_plausibility_gate()

    print("\n--- 8. Testing Action Circuit Breaker ---")
    test_circuit_breaker_tripwire()

    print("\n--- 9. Testing STT Engine ---")
    test_stt_synthetic_audio()

    print("\n--- 10. Testing Action Executors (Safely Mocked) ---")
    test_actions_safe_execution_mocked()

    print("\n--- 11. Testing HUD Overlay with Traffic Lights ---")
    test_hud_overlay_lifecycle()

    print("\n--- 12. Testing Calibration Schema Versioning ---")
    test_calibration_schema_versioning()

    print("\n--- 13. Testing Silero VAD Offline Detection ---")
    test_silero_vad_offline_detection()

    print("\n--- 14. Testing Real Speech Fixture End-to-End ---")
    test_real_speech_fixture_end_to_end()

    print("\n--- 15. Testing Whisper Real Speech Fixture E2E ---")
    test_whisper_real_speech_fixture_e2e()

    print("\n--- 16. Testing Whisper Garbage Rejection ---")
    test_whisper_garbage_rejection()

    print("\n--- 17. Testing Whisper CUDA Latency Benchmark ---")
    test_whisper_cuda_latency_benchmark()

    print("\n--- 18. Testing Microphone Channel Selection ---")
    test_microphone_channel_selection()

    print("\n--- 19. Testing Quiet Audio Level Normalization ---")
    test_quiet_audio_level_normalization()

    print("\n--- 20. Testing Input Device Ranking (WASAPI first, pseudo-devices demoted) ---")
    test_input_device_ranking_avoids_pseudo_devices()

    print("\n>>> ALL PIPELINE TESTS PASSED SUCCESSFULLY! <<<")
