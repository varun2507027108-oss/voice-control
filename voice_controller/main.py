"""Main entry point for Voice Control pipeline connecting Mic -> Adaptive VAD -> STT -> Laya -> Actions -> HUD."""

import collections
import logging
import os
import queue
import sys
import threading
import time
from typing import Deque, List, Optional
import numpy as np
import sounddevice as sd
import torch

try:
    from scipy.signal import resample_poly  # imported once; never inside the realtime callback
except Exception:  # pragma: no cover - scipy is a declared dependency
    resample_poly = None

from voice_controller.config import (
    SAMPLE_RATE,
    CHANNELS,
    CHUNK_SIZE,
    PREROLL_CHUNKS,
    DEDUP_INTERVAL_S,
    MAX_SPEECH_DURATION_S,
    SILENCE_DURATION,
    MIN_SPEECH_DURATION,
    CONFIDENCE_THRESHOLD,
    CALIBRATION_FILE_PATH,
    CALIB_SCHEMA,
    INPUT_API_PREFERENCE,
    DEVICE_OVERRIDE_ENV,
    MIC_LEVEL_ENV,
    MIC_LEVEL_LOW_PCT,
    MIC_LEVEL_TARGET_PCT,
    AUTO_FIX_MIC_LEVEL,
    LIVE_TRANSCRIBE_INTERVAL_S,
    CHANNEL_HYSTERESIS,
    DEVICE_PROBE_SECONDS,
    DEVICE_PROBE_MIN_RMS,
    PRE_VAD_HIGHPASS_HZ,
    RAW_QUEUE_MAXSIZE,
    HUD_UPDATE_INTERVAL_S,
    vad_tracker,
)
import voice_controller.actions as actions
from voice_controller.audio_utils import (
    DualChannelSpeechSelector,
    measure_channel_profile,
    rank_input_candidates,
    select_channel,
)
from voice_controller.engine_stt import get_stt_engine, is_plausible_speech
from voice_controller.engine_intent import get_intent_router
from voice_controller.overlay import HUDOverlay
from voice_controller.vad import get_silero_vad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VoicePipeline")


class VoiceControllerPipeline:
    """Manages audio capture with pre-roll buffer, Silero neural VAD, and asynchronous single-flight dispatch."""

    def __init__(self, hud: HUDOverlay):
        self.hud = hud
        self.audio_task_queue: queue.Queue = queue.Queue(maxsize=2)
        self.live_task_queue: queue.Queue = queue.Queue(maxsize=1)
        self.is_running = False
        self.shutdown_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.live_worker_thread: Optional[threading.Thread] = None
        self._last_live_transcribe_time: float = 0.0
        self._last_partial_text: str = ""

        # Audio stream settings & resampling
        self.native_samplerate = SAMPLE_RATE
        self.resample_up = 1
        self.resample_down = 1
        self.native_chunk_size = CHUNK_SIZE

        # Candidate audio input devices for fallback
        self.candidate_devices: List[dict] = []
        self._detect_input_device()

        # Neural VAD detector
        self.vad = get_silero_vad()

        # Pre-roll ring buffer (6 chunks = 300ms)
        self.preroll_buffer: Deque[np.ndarray] = collections.deque(maxlen=PREROLL_CHUNKS)

        # VAD state
        self.is_speaking = False
        self.speech_buffer: List[np.ndarray] = []
        self.silence_samples = 0
        self.speech_start_time = 0.0

        # Double-fire deduplication state
        self.last_transcription = ""
        self.last_transcription_time = 0.0

        # Rolling latency tracking (ms)
        self.latency_history: List[float] = []

        # Calibration state (in-stream)
        self._calibrating = False
        self._calib_samples: List[np.ndarray] = []

        # Engines
        self.stt_engine = None
        self.intent_router = None
        self.stream: Optional[sd.InputStream] = None

        # Raw audio ring queue between PortAudio callback and audio worker thread
        self.raw_audio_queue: queue.Queue = queue.Queue(maxsize=RAW_QUEUE_MAXSIZE)
        self.audio_worker_thread: Optional[threading.Thread] = None
        self._raw_queue_drops = 0
        self._last_hud_update = 0.0

        # Dual-channel speech selector & preprocessor (DC removal + 75Hz highpass + 16kHz resample)
        self.channel_selector = DualChannelSpeechSelector(
            native_samplerate=self.native_samplerate,
            target_samplerate=SAMPLE_RATE,
            highpass_hz=PRE_VAD_HIGHPASS_HZ,
        )

        # Audio Stream Health & Watchdog
        self._callback_count = 0
        self._last_callback_time = 0.0
        self._callback_errors = 0
        self._watchdog_thread: Optional[threading.Thread] = None
        self._last_restart_time = 0.0
        # Preferred/locked channel
        self._preferred_channel = 0

        # Live capture-level telemetry
        self._level_window: List[float] = []
        self._peak_window = 0.0
        self._max_rms_seen = 0.0
        self._last_level_log = time.perf_counter()

        # Digital-silence recovery state
        self._silent_since: Optional[float] = None
        self._silence_restarts = 0

        # Attach HUD to Silero VAD
        if hasattr(self.vad, "hud"):
            self.vad.hud = self.hud

    def _apply_device(self, cand: dict):
        """Configure sample rate, channels, and resampling parameters for a selected candidate device."""
        self.device_index = cand["index"]
        dev_info = cand["info"]
        self.device_name = dev_info.get("name", "default")
        self.input_channels = min(2, max(1, int(dev_info.get("max_input_channels", 1))))
        vad_tracker.set_device_name(self.device_name)

        native_sr = int(dev_info.get("default_samplerate", SAMPLE_RATE))
        self.native_samplerate = native_sr
        if native_sr != SAMPLE_RATE:
            import math
            g = math.gcd(SAMPLE_RATE, native_sr)
            self.resample_up = SAMPLE_RATE // g
            self.resample_down = native_sr // g
            self.native_chunk_size = int(round(0.05 * native_sr))
        else:
            self.resample_up = 1
            self.resample_down = 1
            self.native_chunk_size = CHUNK_SIZE

        # Re-initialize preprocessor for active sample rate
        self.channel_selector = DualChannelSpeechSelector(
            native_samplerate=self.native_samplerate,
            target_samplerate=SAMPLE_RATE,
            highpass_hz=PRE_VAD_HIGHPASS_HZ,
        )

    def _detect_input_device(self):
        """Identify the best working input device.

        Uses :func:`voice_controller.audio_utils.rank_input_candidates` to order real
        WASAPI/MME endpoints ahead of DirectSound pseudo-devices, honours an optional
        ``VOICE_CONTROL_DEVICE`` override, then probes each candidate with a short
        live capture. A candidate is only accepted when the stream opens *and* the
        captured audio is not digital silence (dead/hijacked endpoint), so the system
        can no longer boot into a "microphone hears nothing" state.
        """
        try:
            host_apis = sd.query_hostapis()
            devices = sd.query_devices()

            override = os.environ.get(DEVICE_OVERRIDE_ENV, "").strip() or None
            ranked = rank_input_candidates(
                host_apis,
                devices,
                api_preference=INPUT_API_PREFERENCE,
                override=override,
            )

            self.candidate_devices = []
            for cand in ranked:
                dev_info = devices[cand["index"]]
                probe_rms = self._probe_device(cand["index"], dev_info)
                if probe_rms is None:
                    logger.warning(
                        "Device rejected (could not open): '%s' (idx=%s, API: %s)",
                        cand["name"], cand["index"], cand["api"],
                    )
                    continue
                if probe_rms < DEVICE_PROBE_MIN_RMS:
                    logger.warning(
                        "Device rejected (digital silence, rms=%.2e): '%s' (idx=%s, API: %s)",
                        probe_rms, cand["name"], cand["index"], cand["api"],
                    )
                    continue
                cand["info"] = dev_info
                self.candidate_devices.append(cand)
                if len(self.candidate_devices) == 1:
                    cand["probe_rms"] = probe_rms

            if self.candidate_devices:
                chosen = self.candidate_devices[0]
                self._apply_device(chosen)
                logger.info(
                    "Audio input selected: '%s' (idx=%s, API: %s, Native SR: %dHz, ch=%d, chunk=%d, live rms=%.4f). Verified %d fallback(s).",
                    self.device_name,
                    self.device_index,
                    chosen["api"],
                    self.native_samplerate,
                    self.input_channels,
                    self.native_chunk_size,
                    chosen.get("probe_rms", 0.0),
                    len(self.candidate_devices) - 1,
                )
            else:
                dev_info = sd.query_devices(kind="input")
                self._apply_device({"index": None, "name": dev_info.get("name", "NO MICROPHONE"), "api": "Default", "info": dev_info})
                self.candidate_devices.append({"index": None, "name": self.device_name, "api": "Default", "info": dev_info})
                logger.critical("NO MICROPHONE: No candidate input device passed the liveness probe (all silent or inaccessible).")
                self.hud.update_state(status="IGNORED", feedback="NO MICROPHONE DETECTED")
        except Exception as err:
            logger.critical("Could not query input audio device (NO MICROPHONE): %s", err)
            self.hud.update_state(status="IGNORED", feedback="NO MICROPHONE DETECTED")
            self.device_index = None
            self.input_channels = CHANNELS
            self.native_samplerate = SAMPLE_RATE
            self.resample_up = 1
            self.resample_down = 1
            self.native_chunk_size = CHUNK_SIZE

    def _probe_device(self, device_index: int, dev_info: dict) -> Optional[float]:
        """Open a stream and capture ``DEVICE_PROBE_SECONDS`` to measure the real signal level.

        Returns the AC-coupled RMS of the strongest channel, or ``None`` when the
        endpoint cannot be opened at all.
        """
        native_sr = int(dev_info.get("default_samplerate", SAMPLE_RATE))
        channels = min(2, max(1, int(dev_info.get("max_input_channels", 1))))
        blocksize = int(round(0.05 * native_sr)) if native_sr != SAMPLE_RATE else CHUNK_SIZE
        collected: List[np.ndarray] = []

        def probe_callback(indata, frames, time_info, status):
            collected.append(indata.copy())

        try:
            stream = sd.InputStream(
                device=device_index,
                samplerate=native_sr,
                channels=channels,
                blocksize=blocksize,
                dtype="float32",
                callback=probe_callback,
            )
            stream.start()
            time.sleep(DEVICE_PROBE_SECONDS)
            stream.stop()
            stream.close()
        except Exception as probe_err:
            logger.debug("Device probe could not open idx=%s: %s", device_index, probe_err)
            return None

        if not collected:
            return 0.0

        audio = np.concatenate(collected)
        if audio.ndim > 1:
            # Score by the strongest channel so one dead capsule cannot mask a live mic
            per_channel = [vad_tracker.compute_rms(audio[:, c]) for c in range(audio.shape[1])]
            return float(max(per_channel)) if per_channel else 0.0
        return vad_tracker.compute_rms(audio)

    def _auto_sample_ambient_floor(self):
        """Quick 1.0s baseline sample of ambient room noise if device is uncalibrated or floor not found."""
        try:
            has_calibrated = False
            if CALIBRATION_FILE_PATH.exists():
                try:
                    import json
                    with open(CALIBRATION_FILE_PATH, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and self.device_name in data:
                        rec = data[self.device_name]
                        if isinstance(rec, dict) and rec.get("schema") == CALIB_SCHEMA:
                            has_calibrated = True
                        else:
                            logger.info("Calibration from old algorithm schema for '%s' — will resample.", self.device_name)
                except Exception:
                    pass

            if not has_calibrated:
                logger.info("Sampling 1.0s ambient room noise for '%s'...", self.device_name)

                def sample_cb(indata, frames, time_info, status):
                    # Calibrate on the same capsule the live pipeline latches onto, otherwise
                    # the noise floor is measured on a channel that may not carry voice.
                    if indata.ndim > 1 and indata.shape[1] > 1:
                        raw, _ = select_channel(
                            indata,
                            self._preferred_channel,
                            CHANNEL_HYSTERESIS,
                            samplerate=self.native_samplerate,
                        )
                    else:
                        raw = indata.flatten().copy()
                    raw = raw - float(np.mean(raw))
                    if self.native_samplerate != SAMPLE_RATE:
                        if resample_poly is None:
                            return
                        c16 = resample_poly(raw, self.resample_up, self.resample_down).astype(np.float32)
                        chunks.append(c16)
                    else:
                        chunks.append(raw.astype(np.float32))

                sample_error = None
                for attempt in (1, 2):
                    chunks = []
                    try:
                        with sd.InputStream(
                            device=self.device_index,
                            samplerate=self.native_samplerate,
                            channels=self.input_channels,
                            dtype="float32",
                            blocksize=self.native_chunk_size,
                            callback=sample_cb,
                        ):
                            time.sleep(1.0)
                        sample_error = None
                        break
                    except Exception as sample_err:
                        sample_error = sample_err
                        logger.warning(
                            "Ambient auto-sample attempt %d/2 failed on '%s': %s",
                            attempt,
                            self.device_name,
                            sample_err,
                        )
                        # The endpoint needs a moment to release the previous WASAPI session;
                        # the first open after the device probe often fails with -9999.
                        time.sleep(0.5)

                if sample_error is not None:
                    logger.warning(
                        "Could not open an extra capture stream for ambient calibration on '%s'. The main "
                        "stream still works; this only leaves the fallback energy-VAD threshold at its default. "
                        "Last error: %s",
                        self.device_name,
                        sample_error,
                    )

                if chunks:
                    rms_vals = [vad_tracker.compute_rms(c) for c in chunks if len(c) > 0]
                    if rms_vals:
                        ambient_p20 = float(np.percentile(rms_vals, 20))
                        if ambient_p20 > 0.003:
                            vad_tracker.save_calibrated_floor(ambient_p20, device_name=self.device_name)
                            logger.info("Ambient baseline noise floor auto-set to: %.4f for '%s'", vad_tracker.noise_floor, self.device_name)
            else:
                logger.info("Loaded existing calibrated noise floor: %.4f for '%s'", vad_tracker.noise_floor, self.device_name)
        except Exception as err:
            logger.warning("Ambient auto-sample skipped: %s", err)

    def initialize_engines(self):
        """Warm up Moonshine and Laya models, then auto-sample ambient noise."""
        self.hud.update_state(
            status="PROCESSING",
            is_loading=True,
            loading_message="Loading speech & intent models...",
        )
        logger.info("Initializing and pre-warming STT and Intent models...")
        self.stt_engine = get_stt_engine()
        self.intent_router = get_intent_router()

        # Ambient baseline check
        self._auto_sample_ambient_floor()

        # Windows capture-endpoint sanity check: a muted or near-zero-gain microphone
        # produces a perfectly healthy PortAudio stream that only ever captures silence.
        self._check_microphone_endpoint()

        self.hud.update_state(
            is_loading=False,
            status="LISTENING",
            audio_level=0.0,
        )
        onset_th, hangover_th = vad_tracker.get_thresholds()
        logger.info(
            "Pipeline ready for microphone input (Device: '%s', noise floor: %.4f, onset threshold: %.4f, hangover: %.4f).",
            self.device_name,
            vad_tracker.noise_floor,
            onset_th,
            hangover_th,
        )

    def _check_microphone_endpoint(self):
        """Log/raise Windows capture mute + gain state so silence is never misdiagnosed."""
        # Windows microphone privacy consent: when denied, shared-mode capture silently
        # returns digital silence while the stream still reports healthy.
        try:
            perms = actions.get_microphone_permissions()
        except Exception as err:
            perms = {"available": False, "error": str(err)}

        if perms.get("available"):
            blocked = [
                label
                for label, value in (
                    ("microphone access", perms.get("global_access")),
                    ("desktop apps", perms.get("desktop_apps")),
                    ("machine policy", perms.get("machine_policy")),
                    ("this app", perms.get("this_app")),
                )
                if str(value).lower() == "deny"
            ]
            if blocked:
                logger.critical(
                    "Windows microphone privacy is BLOCKING access (%s). Fix: Settings > Privacy & security > "
                    "Microphone > enable 'Microphone access' and 'Let desktop apps access your microphone'.",
                    ", ".join(blocked),
                )
                self.hud.update_state(status="IGNORED", feedback="Windows privacy blocks microphone access")
                return
            logger.info(
                "Windows microphone permissions OK (global=%s, desktop apps=%s, machine policy=%s, this app=%s).",
                perms.get("global_access"),
                perms.get("desktop_apps"),
                perms.get("machine_policy"),
                perms.get("this_app"),
            )

        try:
            mic_status = actions.get_microphone_status()
        except Exception as err:
            logger.debug("Microphone endpoint check failed: %s", err)
            return

        if not mic_status.get("available"):
            logger.info("Windows capture endpoint status unavailable: %s", mic_status.get("error"))
            return

        level_pct = float(mic_status.get("level_pct", 100.0))
        if mic_status.get("muted"):
            logger.critical("Windows microphone is MUTED — voice commands cannot be captured. Unmute it in Sound settings.")
            self.hud.update_state(status="IGNORED", feedback="Microphone is muted in Windows")
        elif level_pct < MIC_LEVEL_LOW_PCT:
            # Automatic or configured microphone level fix for reliable recognition
            target_env = os.environ.get(MIC_LEVEL_ENV, "").strip()
            wanted_pct = None
            if target_env:
                try:
                    wanted_pct = float(target_env)
                except ValueError:
                    logger.warning(
                        "Ignoring invalid %s='%s' (expected a percentage such as 100).", MIC_LEVEL_ENV, target_env
                    )
            elif AUTO_FIX_MIC_LEVEL:
                wanted_pct = MIC_LEVEL_TARGET_PCT

            if wanted_pct is not None:
                result = actions.set_microphone_level(wanted_pct)
                if result.success:
                    logger.info(
                        "Windows microphone input level automatically raised to %.0f%% (was %.0f%%) for reliable recognition.",
                        result.data.get("level_pct", wanted_pct),
                        result.data.get("previous_pct", level_pct),
                    )
                    self.hud.update_state(
                        status="LISTENING",
                        feedback=f"Mic level auto-raised to {result.data.get('level_pct', wanted_pct):.0f}%",
                    )
                    return
                logger.warning("Input-level fix request failed: %s", result.message)

            logger.warning(
                "Windows microphone input level is only %.0f%% — raise it to ~100%% in Sound settings for reliable "
                "recognition (or set %s=100 to let the app do it).",
                level_pct,
                MIC_LEVEL_ENV,
            )
            self.hud.update_state(status="LISTENING", feedback=f"Mic level {level_pct:.0f}% — raise it in Windows Sound")
        else:
            logger.info("Windows microphone endpoint OK (level %.0f%%, muted=%s).", level_pct, mic_status.get("muted"))

    def _flush_speech_buffer(self):
        """Package buffered speech audio, unlock channel, and dispatch to STT worker queue."""
        # Unlock channel so next utterance can independently re-evaluate channels
        self.channel_selector.unlock()

        # Drain any pending live transcription snapshots so final STT is not delayed
        while not self.live_task_queue.empty():
            try:
                self.live_task_queue.get_nowait()
            except queue.Empty:
                break
        self._last_partial_text = ""

        if not self.speech_buffer:
            self.is_speaking = False
            self.silence_samples = 0
            self.vad.reset()
            self.hud.update_state(live_transcription="", audio_level=0.0)
            return

        total_duration = time.perf_counter() - self.speech_start_time
        if total_duration >= MIN_SPEECH_DURATION:
            audio_full = np.concatenate(self.speech_buffer)

            # Single-flight: drop stale audio buffer if inference worker is busy
            if self.audio_task_queue.full():
                try:
                    _ = self.audio_task_queue.get_nowait()
                    logger.info("Dropped stale audio buffer to maintain real-time responsiveness.")
                except queue.Empty:
                    pass

            try:
                self.audio_task_queue.put_nowait(audio_full)
                logger.info("Captured utterance: %.2fs (%d samples). Dispatched to STT.", total_duration, len(audio_full))
            except queue.Full:
                pass

        self.speech_buffer = []
        self.is_speaking = False
        self.silence_samples = 0
        self.vad.reset()
        self.hud.update_state(audio_level=0.0)

    def _audio_callback(self, indata, frames, time_info, status):
        """PortAudio callback: copy-only into bounded raw queue.

        Zero Torch inference, zero resampling, zero heavy math, zero GUI calls.
        """
        if status:
            logger.debug("SoundDevice status warning: %s", status)

        if not self.is_running or self.shutdown_event.is_set():
            return

        # Track callback heartbeat for watchdog
        self._callback_count += 1
        self._last_callback_time = time.perf_counter()

        try:
            if self.raw_audio_queue.full():
                try:
                    _ = self.raw_audio_queue.get_nowait()
                    self._raw_queue_drops += 1
                except queue.Empty:
                    pass
            self.raw_audio_queue.put_nowait(indata.copy())
        except Exception as err:
            self._callback_errors += 1
            if self._callback_errors <= 3 or self._callback_errors % 100 == 0:
                logger.error(
                    "Audio callback enqueue error #%d: %s", self._callback_errors, err
                )

    def _audio_worker_loop(self):
        """Dedicated audio processing worker running outside PortAudio callback:
        - Dequeues raw float32 audio blocks from raw_audio_queue
        - DC removal + 75Hz high-pass filtering + 16kHz resampling via channel_selector
        - Dual-channel Silero neural VAD evaluation
        - Utterance-level channel locking
        - Feeds pre-roll and speech buffers
        - Coalesced visual meter updates to HUD
        """
        logger.info("Audio processing worker thread started.")
        while self.is_running and not self.shutdown_event.is_set():
            try:
                raw_indata = self.raw_audio_queue.get(timeout=0.15)
            except queue.Empty:
                continue

            try:
                self._handle_audio_block(raw_indata)
            except Exception as err:
                logger.error("Audio worker processing error: %s", err, exc_info=True)
            finally:
                self.raw_audio_queue.task_done()

    def _handle_audio_block(self, indata: np.ndarray):
        """Preprocess audio, run dual-channel Silero VAD, and manage utterance state machine."""
        ch0_16k, ch1_16k, active_idx = self.channel_selector.process_block(indata)

        # In-stream calibration recording
        if getattr(self, "_calibrating", False):
            self._calib_samples.append((ch0_16k, ch1_16k))

        # Evaluate dual-channel Silero neural VAD
        is_speech_chunk, best_ch, p0, p1 = self.vad.evaluate_dual(
            ch0_16k,
            ch1_16k,
            is_speaking=self.is_speaking,
            locked_channel=self.channel_selector.locked_channel,
            hud=None,
        )

        active_chunk = ch0_16k if best_ch == 0 or ch1_16k is None else ch1_16k
        rms = vad_tracker.compute_rms(active_chunk)

        # Track digital silence (rms==0) so the watchdog can heal a dead-but-"active" stream
        if rms < 1e-5:
            if self._silent_since is None:
                self._silent_since = time.perf_counter()
        else:
            self._silent_since = None

        now_time = time.perf_counter()

        onset_th, _ = vad_tracker.get_thresholds()
        dynamic_scale = max(0.015, (onset_th - vad_tracker.noise_floor) * 2.5)
        norm_level = min(1.0, max(0.0, (rms - vad_tracker.noise_floor) / dynamic_scale))

        # Only adapt noise floor during true ambient silence (quiet room baseline).
        # NEVER adapt when the user is speaking or when acoustic energy / neural prob indicates voice!
        if not self.is_speaking and not is_speech_chunk:
            if p0 < 0.05 and p1 < 0.05 and norm_level < 0.15:
                vad_tracker.adapt_floor(rms)

        # Energy-assisted onset bridge: if audio level is clearly spiking (green lines jumping)
        # and neural VAD indicates voice activity, trigger speech onset immediately!
        if not is_speech_chunk:
            if not self.is_speaking:
                # Onset: green lines jumping (norm_level >= 0.20) + neural voice hint
                if norm_level >= 0.20 and (p0 >= 0.05 or p1 >= 0.05):
                    is_speech_chunk = True
                    best_ch = 0 if p0 >= p1 else 1
                    active_chunk = ch0_16k if best_ch == 0 or ch1_16k is None else ch1_16k
            else:
                # Continuation / hangover while speaking: keep alive if energy or voice persists
                if norm_level >= 0.15 or p0 >= 0.05 or p1 >= 0.05:
                    is_speech_chunk = True

        # Coalesced visual pulse level updates to HUD (throttled)
        if (now_time - self._last_hud_update) >= HUD_UPDATE_INTERVAL_S:
            self._last_hud_update = now_time
            self.hud.update_state(audio_level=norm_level)

        # Capture-level telemetry: distinguishes dead mic from VAD disagreement
        self._level_window.append(rms)
        self._peak_window = max(self._peak_window, float(np.max(np.abs(active_chunk))) if len(active_chunk) else 0.0)
        self._max_rms_seen = max(self._max_rms_seen, rms)
        if (now_time - self._last_level_log) >= 2.0 and self._level_window:
            window = np.asarray(self._level_window, dtype=np.float32)
            onset_th, _ = vad_tracker.get_thresholds()
            logger.info(
                "Capture level (%.1fs): rms p50=%.4f p95=%.4f max=%.4f peak=%.3f | floor=%.4f onset=%.4f | ch=%d (p0=%.2f, p1=%.2f, drops=%d)",
                now_time - self._last_level_log,
                float(np.percentile(window, 50)),
                float(np.percentile(window, 95)),
                float(np.max(window)),
                self._peak_window,
                vad_tracker.noise_floor,
                onset_th,
                best_ch,
                p0,
                p1,
                self._raw_queue_drops,
            )
            self._level_window = []
            self._peak_window = 0.0
            self._last_level_log = now_time

        # Continually store pre-roll chunks
        self.preroll_buffer.append(active_chunk)

        if is_speech_chunk:
            if not self.is_speaking:
                self.is_speaking = True
                # Lock channel for the entire utterance!
                self.channel_selector.lock_channel(best_ch)
                self.speech_start_time = time.perf_counter()
                self._last_live_transcribe_time = self.speech_start_time
                self._last_partial_text = ""
                # Prepend the pre-roll buffer so initial consonants/syllables are never clipped
                self.speech_buffer = list(self.preroll_buffer)
                self.silence_samples = 0
                logger.info(
                    "Speech onset detected on CH%d (prob=%.2f, RMS=%.4f, floor=%.4f). Locked channel.",
                    best_ch,
                    p0 if best_ch == 0 else p1,
                    rms,
                    vad_tracker.noise_floor,
                )
                self.hud.update_state(
                    status="HEARING",
                    live_transcription="...",
                )

            self.speech_buffer.append(active_chunk)
            self.silence_samples = 0

            # Continuous live transcription: feed live audio snapshot to background partial transcriber
            if (
                self.stt_engine is not None
                and len(self.speech_buffer) >= 6
                and (now_time - self._last_live_transcribe_time) >= LIVE_TRANSCRIBE_INTERVAL_S
            ):
                self._last_live_transcribe_time = now_time
                try:
                    snapshot = np.concatenate(self.speech_buffer)
                    if self.live_task_queue.full():
                        try:
                            self.live_task_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.live_task_queue.put_nowait(snapshot)
                except Exception as queue_err:
                    logger.debug("Live task queue put failed: %s", queue_err)

            # Safety watchdog: prevent infinite runaway buffer in continuous noise
            if (time.perf_counter() - self.speech_start_time) >= MAX_SPEECH_DURATION_S:
                logger.warning("Utterance hit max duration (%.1fs). Forcing buffer slice.", MAX_SPEECH_DURATION_S)
                self._flush_speech_buffer()

        elif self.is_speaking:
            # Silence detected after speech was initiated
            self.speech_buffer.append(active_chunk)
            self.silence_samples += len(active_chunk)

            silence_secs = self.silence_samples / float(SAMPLE_RATE)
            if silence_secs >= SILENCE_DURATION:
                # End of utterance reached cleanly
                self._flush_speech_buffer()

    def _calibrate_mic_procedure(self):
        """Record 2 seconds of ambient audio in-stream to calibrate baseline noise floor for current device."""
        logger.info("Starting 2-second ambient microphone noise floor calibration for '%s'...", self.device_name)
        self.hud.update_state(
            status="PROCESSING",
            feedback=f"Calibrating mic ambient floor ({self.device_name[:18]})...",
        )
        try:
            self._calib_samples = []
            self._calibrating = True
            time.sleep(2.0)
            self._calibrating = False

            if self._calib_samples:
                ch0_parts = [s[0] for s in self._calib_samples if s[0] is not None and len(s[0]) > 0]
                if ch0_parts:
                    ch0_all = np.concatenate(ch0_parts)
                    p0 = measure_channel_profile(ch0_all, SAMPLE_RATE)
                    channel_profiles = {"0": p0}
                    preferred = 0

                    if self._calib_samples[0][1] is not None:
                        ch1_parts = [s[1] for s in self._calib_samples if s[1] is not None and len(s[1]) > 0]
                        if ch1_parts:
                            ch1_all = np.concatenate(ch1_parts)
                            p1 = measure_channel_profile(ch1_all, SAMPLE_RATE)
                            channel_profiles["1"] = p1
                            # Prefer cleaner channel
                            if p0["low_band_rms"] < p1["low_band_rms"] * 0.8:
                                preferred = 0
                            elif p1["low_band_rms"] < p0["low_band_rms"] * 0.8:
                                preferred = 1

                    measured_floor = p0["speech_band_rms"] if preferred == 0 else channel_profiles.get("1", p0)["speech_band_rms"]
                    measured_floor = max(0.003, float(measured_floor))
                    vad_tracker.save_calibrated_floor(
                        measured_floor,
                        device_name=self.device_name,
                        channel_profiles=channel_profiles,
                        preferred_channel=preferred,
                    )
                    onset_th, _ = vad_tracker.get_thresholds()
                    logger.info(
                        "Microphone '%s' calibrated floor: %.4f (preferred ch%d, onset threshold: %.4f)",
                        self.device_name,
                        measured_floor,
                        preferred,
                        onset_th,
                    )
                    self.hud.update_state(
                        status="EXECUTED",
                        feedback=f"Calibrated noise floor: {measured_floor:.4f} (ch{preferred})",
                    )
                    return
            self.hud.update_state(status="IGNORED", feedback="Calibration failed: no audio captured")
        except Exception as err:
            self._calibrating = False
            logger.error("Calibration failed: %s", err)
            self.hud.update_state(status="IGNORED", feedback=f"Calibration error: {err}")
            self.hud.update_state(status="IGNORED", feedback="Calibration failed: no audio captured")
        except Exception as err:
            self._calibrating = False
            logger.error("Calibration failed: %s", err)
            self.hud.update_state(status="IGNORED", feedback=f"Calibration error: {err}")

    def _live_worker_loop(self):
        """Dedicated background thread continuously generating live partial transcriptions."""
        logger.info("Live transcription worker thread started.")
        while self.is_running and not self.shutdown_event.is_set():
            try:
                audio_np = self.live_task_queue.get(timeout=0.15)
            except queue.Empty:
                continue

            try:
                if self.stt_engine is not None and len(audio_np) > 0 and self.is_speaking:
                    partial = self.stt_engine.transcribe_partial(audio_np)
                    if partial and partial != self._last_partial_text:
                        self._last_partial_text = partial
                        self.hud.update_state(
                            status="HEARING",
                            live_transcription=partial,
                        )
            except Exception as err:
                logger.debug("Live transcription partial error: %s", err)
            finally:
                self.live_task_queue.task_done()

    def _worker_loop(self):
        """Background inference worker with latency percentiles and deduplication."""
        logger.info("Inference worker thread started.")
        while self.is_running and not self.shutdown_event.is_set():
            try:
                audio_np = self.audio_task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._process_utterance(audio_np)
            except Exception as err:
                logger.error("Error processing utterance: %s", err, exc_info=True)
                self.hud.update_state(
                    status="IGNORED",
                    feedback=f"Error: {err}",
                )
                time.sleep(1.2)
                self.hud.update_state(status="LISTENING")
            finally:
                self.audio_task_queue.task_done()

    def _save_rejected_wav(self, audio_16k: np.ndarray, reason: str = ""):
        """Save rejected audio buffer to scratch/failed_utt_*.wav for forensic listening."""
        try:
            import wave
            from pathlib import Path
            scratch_dir = Path("scratch")
            scratch_dir.mkdir(parents=True, exist_ok=True)
            fname = scratch_dir / f"failed_utt_{time.strftime('%H%M%S')}.wav"
            scaled = np.clip(audio_16k * 32767.0, -32768, 32767).astype(np.int16)
            with wave.open(str(fname), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(scaled.tobytes())
            logger.info("Saved rejected utterance audio to: %s (reason: %s)", fname, reason)
        except Exception as err:
            logger.warning("Could not dump rejected utterance WAV: %s", err)

    def _process_utterance(self, audio_np: np.ndarray):
        """Execute complete pipeline pass from audio to action."""
        t_start = time.perf_counter()
        self.hud.update_state(status="PROCESSING")

        # 1. STT Transcription (passing calibrated/adapted noise floor)
        stt_result, stt_latency_ms = self.stt_engine.transcribe_with_latency(
            audio_np, noise_floor=vad_tracker.noise_floor
        )

        last_rms = float(getattr(self.stt_engine, "last_input_rms", 0.0))
        last_gain = float(getattr(self.stt_engine, "last_gain", 1.0))

        if not stt_result.safe_for_execution or not stt_result.trusted:
            logger.warning(
                "[STTEngine] Untrusted/unsafe transcript rejected: text='%s' speech_detected=%s quality=%.2f safe=%s no_speech_prob=%.4f avg_logprob=%.4f compression_ratio=%.4f input_rms=%.4f gain=%.1fx",
                stt_result.text,
                stt_result.speech_detected,
                stt_result.quality_score,
                stt_result.safe_for_execution,
                stt_result.no_speech_prob,
                stt_result.avg_logprob,
                stt_result.compression_ratio,
                last_rms,
                last_gain,
            )
            self._save_rejected_wav(
                audio_np,
                reason=f"untrusted_speech_{stt_result.speech_detected}_quality_{stt_result.quality_score:.2f}_rms_{last_rms:.4f}",
            )
            if not stt_result.speech_detected or (last_rms and last_rms < 0.01 and last_gain >= 2.0):
                feedback = "Too quiet — speak closer to the mic"
            elif stt_result.text and not stt_result.safe_for_execution:
                feedback = f"Low audio quality ({stt_result.quality_score:.1f}) — speak clearer"
            else:
                feedback = "Didn't catch that"

            self.hud.update_state(
                status="IGNORED",
                transcription=stt_result.text or "",
                feedback=feedback,
                live_transcription="",
            )
            time.sleep(1.2)
            self.hud.update_state(status="LISTENING", live_transcription="")
            return

        text = stt_result.text
        now = time.perf_counter()
        # 2. Deduplication check (prevent double-firing on reverberation/echo, 0.7s window)
        if text == self.last_transcription and (now - self.last_transcription_time) < DEDUP_INTERVAL_S:
            logger.info("Suppressed duplicate utterance within %.1fs: '%s'", DEDUP_INTERVAL_S, text)
            self.hud.update_state(status="LISTENING")
            return

        self.last_transcription = text
        self.last_transcription_time = now

        logger.info(
            "Transcribed in %.1fms: '%s' (no_speech_prob=%.4f, avg_logprob=%.4f, compression_ratio=%.4f)",
            stt_latency_ms,
            text,
            stt_result.no_speech_prob,
            stt_result.avg_logprob,
            stt_result.compression_ratio,
        )

        # 3. Intent Routing (Two-stage: Fast-path -> Laya)
        t_intent_start = time.perf_counter()
        pred = self.intent_router.predict(text)
        intent_latency_ms = (time.perf_counter() - t_intent_start) * 1000.0

        action = pred.get("action", "unrecognized")
        conf = pred.get("confidence", 0.0)
        source = pred.get("source", "laya")
        slot_value = pred.get("slot_value")
        target_app = pred.get("target_app")
        is_ambiguous = pred.get("is_ambiguous", False)

        logger.info(
            "Intent in %.1fms (%s): action='%s', conf=%.2f, slot=%s, ambiguous=%s",
            intent_latency_ms,
            source,
            action,
            conf,
            slot_value,
            is_ambiguous,
        )

        # Special Action: Calibrate mic
        if action == "calibrate_mic":
            self._calibrate_mic_procedure()
            time.sleep(1.0)
            self.hud.update_state(status="LISTENING")
            return

        # 4. Action Execution Decision Gate
        if conf >= CONFIDENCE_THRESHOLD and action != "unrecognized":
            t_action_start = time.perf_counter()
            result = actions.execute(action, slot_value=slot_value, target_app=target_app)
            action_latency_ms = (time.perf_counter() - t_action_start) * 1000.0

            total_latency_ms = (time.perf_counter() - t_start) * 1000.0
            self.latency_history.append(total_latency_ms)
            if len(self.latency_history) > 100:
                self.latency_history.pop(0)

            # Rolling P50 & P95 metrics
            p50 = float(np.percentile(self.latency_history, 50))
            p95 = float(np.percentile(self.latency_history, 95))
            logger.info("Total pipeline latency: %.1fms (P50=%.1fms, P95=%.1fms) -> '%s'", total_latency_ms, p50, p95, result.message)

            self.hud.update_state(
                status="EXECUTED",
                transcription=text,
                feedback=result.message,
                live_transcription="",
            )
        elif is_ambiguous:
            logger.info("Command marked ambiguous: '%s'", text)
            self.hud.update_state(
                status="IGNORED",
                transcription=text,
                feedback="Command ambiguous — please repeat",
                live_transcription="",
            )
        else:
            logger.info("Action skipped (unrecognized or low confidence).")
            self.hud.update_state(
                status="IGNORED",
                transcription=text,
                feedback="Command not recognized",
                live_transcription="",
            )

        # Return to listening state after brief display
        time.sleep(1.5)
        self.hud.update_state(status="LISTENING", live_transcription="")

    def _start_audio_stream(self):
        """Open sounddevice input stream at native hardware sample rate with multi-device fallback."""
        attempts = []
        if self.candidate_devices:
            current = [c for c in self.candidate_devices if c["index"] == self.device_index]
            others = [c for c in self.candidate_devices if c["index"] != self.device_index]
            attempts = current + others
        else:
            attempts = [{"index": self.device_index, "name": self.device_name, "api": "Selected", "info": {"default_samplerate": self.native_samplerate, "max_input_channels": self.input_channels}}]

        last_err = None
        for cand in attempts:
            try:
                self._apply_device(cand)
                logger.info(
                    "Opening microphone stream (Device: %s [idx=%s, API: %s], Native SR: %dHz, channels: %d, blocksize=%d)...",
                    self.device_name,
                    self.device_index,
                    cand.get("api", "Unknown"),
                    self.native_samplerate,
                    self.input_channels,
                    self.native_chunk_size,
                )
                self.stream = sd.InputStream(
                    device=self.device_index,
                    samplerate=self.native_samplerate,
                    channels=self.input_channels,
                    dtype="float32",
                    blocksize=self.native_chunk_size,
                    callback=self._audio_callback,
                )
                self.stream.start()
                logger.info("Microphone stream successfully active on '%s'.", self.device_name)
                return
            except Exception as err:
                last_err = err
                logger.warning("Failed to start audio stream on '%s' (idx=%s): %s", cand.get("name"), cand.get("index"), err)
                if self.stream:
                    try:
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None

        # Ultimate fallback: system default device at 16000Hz mono
        try:
            logger.warning("Attempting ultimate fallback to system default audio input at 16kHz...")
            self.device_index = None
            self.device_name = "System Default Fallback"
            self.native_samplerate = SAMPLE_RATE
            self.input_channels = 1
            self.resample_up = 1
            self.resample_down = 1
            self.native_chunk_size = CHUNK_SIZE
            self.stream = sd.InputStream(
                device=None,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SIZE,
                callback=self._audio_callback,
            )
            self.stream.start()
            logger.info("System default audio stream active at 16kHz.")
            return
        except Exception as final_err:
            logger.critical("All audio input fallbacks failed: %s (original error: %s)", final_err, last_err)
            raise final_err

    def _watchdog_worker(self):
        """Monitor PortAudio stream callback health and self-heal stalled streams."""
        has_alerted = False
        while self.is_running and not self.shutdown_event.is_set():
            time.sleep(0.5)
            if self.stream is not None and getattr(self.stream, "active", False):
                if self._callback_count > 0:
                    gap = time.perf_counter() - self._last_callback_time
                    if gap > 2.0 and not has_alerted:
                        logger.critical("Microphone stream stalled! No audio callback received for %.2fs.", gap)
                        self.hud.update_state(status="IGNORED", feedback="Microphone stream stalled — restarting")
                        has_alerted = True
                        self._restart_audio_stream()
                    elif gap <= 1.0 and has_alerted:
                        logger.info("Microphone stream resumed callbacks.")
                        has_alerted = False

                # A Windows capture stream can stay "active" while delivering pure digital
                # silence (endpoint reconfiguration, privacy transition, driver hiccup).
                # Restarting the stream heals it instead of leaving a silently deaf app.
                silent_for = 0.0 if self._silent_since is None else (time.perf_counter() - self._silent_since)
                if silent_for > 3.0 and self._max_rms_seen > DEVICE_PROBE_MIN_RMS and self._silence_restarts < 5:
                    self._silence_restarts += 1
                    logger.warning(
                        "Microphone delivered digital silence for %.1fs — restarting the stream (%d/5).",
                        silent_for,
                        self._silence_restarts,
                    )
                    self.hud.update_state(status="IGNORED", feedback="Microphone silent — restarting")
                    self._restart_audio_stream()
                    self._silent_since = None
            elif self.stream is not None and not getattr(self.stream, "active", False) and not self.shutdown_event.is_set():
                logger.warning("Microphone stream is not active; attempting restart.")
                self.hud.update_state(status="IGNORED", feedback="Microphone disconnected — reconnecting")
                self._restart_audio_stream()

    def _restart_audio_stream(self):
        """Tear down and re-open the capture stream, rate-limited to once every 5s.

        Without this, a driver hiccup (USB mic replug, Realtek endpoint reset, another
        app grabbing the mic) leaves the pipeline running but permanently deaf.
        """
        now = time.perf_counter()
        if (now - self._last_restart_time) < 5.0:
            return
        self._last_restart_time = now

        try:
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            self._start_audio_stream()
            self._callback_count = 0
            self._last_callback_time = time.perf_counter()
            logger.info("Microphone stream restart complete (device '%s').", self.device_name)
            self.hud.update_state(status="LISTENING", feedback="Microphone reconnected")
        except Exception as err:
            logger.error("Microphone stream restart failed: %s", err, exc_info=True)

    def _verify_live_capture(self, window_s: float = 2.0):
        """Prove the freshly opened stream is delivering non-silent audio.

        A Windows shared-mode capture stream can open successfully, report ``active``,
        and still deliver pure digital silence (privacy consent, hardware mute, driver
        state, another app holding the endpoint). Sampling the live callback level at
        boot converts that silent failure into one explicit, actionable log line.
        """
        start_time = time.perf_counter()
        time.sleep(window_s)
        if not self.is_running or self.shutdown_event.is_set():
            return

        callbacks = self._callback_count
        max_rms = self._max_rms_seen
        if callbacks == 0:
            logger.critical(
                "No audio callbacks arrived within %.1fs of opening '%s' — the stream is not running.",
                window_s,
                self.device_name,
            )
            self.hud.update_state(status="IGNORED", feedback="No audio from microphone — check Windows Sound")
            return
        if max_rms < DEVICE_PROBE_MIN_RMS:
            perms = actions.get_microphone_permissions()
            logger.critical(
                "Microphone '%s' delivered DIGITAL SILENCE (max rms=%.2e over %d blocks in %.1fs). "
                "Windows mic consent: global=%s, desktop apps=%s, this app=%s. "
                "Check: Settings > Privacy & security > Microphone (allow desktop apps), "
                "Sound > Input volume/mute, and that no other app holds the device in exclusive mode.",
                self.device_name,
                max_rms,
                callbacks,
                window_s,
                perms.get("global_access"),
                perms.get("desktop_apps"),
                perms.get("this_app"),
            )
            self.hud.update_state(status="IGNORED", feedback="Microphone silent — raise Windows input level")
            return
        logger.info(
            "Live capture verified: %d callbacks, max rms=%.4f over the first %.1fs on '%s' (ch=%d).",
            callbacks,
            max_rms,
            window_s,
            self.device_name,
            self._preferred_channel,
        )

    def start(self):
        """Start audio worker, inference workers, stream watchdog, and input stream."""
        self.is_running = True
        self.shutdown_event.clear()

        # Launch dedicated audio processing worker (VAD, high-pass filtering, channel selection)
        self.audio_worker_thread = threading.Thread(target=self._audio_worker_loop, daemon=True, name="AudioWorkerThread")
        self.audio_worker_thread.start()

        # Launch final inference worker thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="WorkerThread")
        self.worker_thread.start()

        # Launch live transcription worker thread
        self.live_worker_thread = threading.Thread(target=self._live_worker_loop, daemon=True, name="LiveTranscribeThread")
        self.live_worker_thread.start()

        # Launch callback watchdog
        self._watchdog_thread = threading.Thread(target=self._watchdog_worker, daemon=True, name="AudioWatchdog")
        self._watchdog_thread.start()

        # Start sounddevice input stream
        self._start_audio_stream()
        self._verify_live_capture()

    def stop(self):
        """Gracefully stop audio stream and release resources."""
        logger.info("Stopping VoiceControllerPipeline...")
        self.is_running = False
        self.shutdown_event.set()

        # Drain raw audio queue
        while not self.raw_audio_queue.empty():
            try:
                self.raw_audio_queue.get_nowait()
            except queue.Empty:
                break

        # Drain live task queue
        while not self.live_task_queue.empty():
            try:
                self.live_task_queue.get_nowait()
            except queue.Empty:
                break

        # Drain audio task queue
        while not self.audio_task_queue.empty():
            try:
                self.audio_task_queue.get_nowait()
            except queue.Empty:
                break

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    logger.info("Starting Local Voice Control System...")

    hud = HUDOverlay()
    pipeline = VoiceControllerPipeline(hud)

    hud.on_close_callback = pipeline.stop

    def startup_worker():
        try:
            pipeline.initialize_engines()
            pipeline.start()
        except Exception as err:
            logger.critical("Fatal initialization failure: %s", err, exc_info=True)
            hud.update_state(status="IGNORED", feedback=f"Init error: {err}")

    init_thread = threading.Thread(target=startup_worker, daemon=True, name="InitThread")
    init_thread.start()

    try:
        hud.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        pipeline.stop()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
