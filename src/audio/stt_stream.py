"""
Real-time streaming speech-to-text (STT) worker for EchoFlux.
Captures 16kHz PCM audio via sounddevice into a thread-safe sliding ring buffer.
Uses faster-whisper with rolling context window to emit partial hypotheses every 80-120ms.
"""

import sys
import time
import queue
import logging
import threading
from typing import Callable, Optional
import numpy as np

logger = logging.getLogger("EchoFlux.STT")

# Suppress noisy internal VAD logs from faster_whisper
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

try:
    import sounddevice as sd
except Exception as e:
    logger.warning("sounddevice not available: %s", e)
    sd = None

try:
    from faster_whisper import WhisperModel
except Exception as e:
    logger.warning("faster-whisper not available: %s", e)
    WhisperModel = None


class StreamingSTT:
    """
    Continuous audio capture and low-latency streaming transcriber.
    Emits partial transcripts and real-time audio RMS levels for the HUD visualizer.
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "auto",
        compute_type: str = "default",
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
        buffer_duration_sec: float = 3.0,
        on_partial: Optional[Callable[[str, float], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * (chunk_duration_ms / 1000.0))
        self.buffer_size = int(sample_rate * buffer_duration_sec)
        self.on_partial = on_partial
        self.on_final = on_final

        self.audio_queue = queue.Queue()
        self.ring_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        self.is_running = False
        self._capture_thread = None
        self._transcribe_thread = None
        self.stream = None

        # Device selection: if auto, use cuda if available
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device

        if compute_type == "default":
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        self.model_size = model_size
        self.model: Optional[WhisperModel] = None
        self._last_rms = 0.0
        self.ambient_rms = 0.035  # Adaptive baseline noise floor

    def load_model(self):
        """Load faster-whisper model."""
        if WhisperModel is None:
            logger.error("faster-whisper is not installed. STT disabled.")
            return

        logger.info(
            "Loading faster-whisper model '%s' on %s (%s)...",
            self.model_size, self.device, self.compute_type
        )
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Speech model loaded successfully.")
        except Exception as e:
            logger.warning("Failed to load on %s (%s): %s; falling back to CPU int8", self.device, self.compute_type, e)
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice streaming input callback with stereo-to-mono downmixing."""
        if status:
            logger.debug("Sounddevice input status: %s", status)
        if not self.is_running:
            return

        # Handle multi-channel hardware (e.g. Realtek dual microphone arrays)
        if indata.ndim > 1 and indata.shape[1] > 1:
            ch0 = indata[:, 0]
            ch1 = indata[:, 1]
            rms0 = float(np.sqrt(np.mean(ch0 ** 2)))
            rms1 = float(np.sqrt(np.mean(ch1 ** 2)))
            # If one channel is significantly stronger, pick that channel to avoid phase cancellation
            if rms1 > rms0 * 1.3:
                audio_chunk = ch1.copy()
            elif rms0 > rms1 * 1.3:
                audio_chunk = ch0.copy()
            else:
                audio_chunk = (ch0 + ch1) * 0.5
        elif indata.ndim > 1:
            audio_chunk = indata[:, 0].copy()
        else:
            audio_chunk = indata.copy()

        # Remove DC offset to center waveform around 0
        audio_chunk = audio_chunk - float(np.mean(audio_chunk))
        self.audio_queue.put(audio_chunk)

    def _capture_worker(self):
        """Worker thread feeding ring buffer from microphone queue."""
        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Calculate RMS energy for visualizer
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            # Smoothly adapt ambient noise floor when audio is at rest
            if rms < self.ambient_rms * 1.6 or self.ambient_rms == 0:
                self.ambient_rms = 0.95 * self.ambient_rms + 0.05 * rms

            # Active speech energy above noise floor for normalized visual HUD level (0.0 to 1.0)
            active_energy = max(0.0, rms - self.ambient_rms * 0.8)
            self._last_rms = min(1.0, active_energy * 6.0)

            with self.buffer_lock:
                # Roll ring buffer and append new chunk
                chunk_len = len(chunk)
                if chunk_len >= self.buffer_size:
                    self.ring_buffer = chunk[-self.buffer_size:]
                else:
                    self.ring_buffer = np.roll(self.ring_buffer, -chunk_len)
                    self.ring_buffer[-chunk_len:] = chunk

    def _transcribe_worker(self):
        """Worker thread running inference on rolling ring buffer every 80-120ms."""
        last_text = ""
        silence_count = 0

        while self.is_running:
            start_t = time.perf_counter()

            if self.model is None:
                time.sleep(0.1)
                continue

            with self.buffer_lock:
                audio_data = self.ring_buffer.copy()

            # Dynamic VAD / silence gate: check energy of the most recent 0.4s
            recent_chunk = audio_data[-int(self.sample_rate * 0.4):]
            recent_rms = float(np.sqrt(np.mean(recent_chunk ** 2)))

            speech_threshold = max(0.030, self.ambient_rms * 1.25)

            if recent_rms < speech_threshold:
                silence_count += 1
                if silence_count > 6 and last_text:
                    if self.on_final:
                        self.on_final(last_text)
                    last_text = ""
                    if self.on_partial:
                        self.on_partial("", self._last_rms)
                else:
                    # Keep notifying UI of silence / idle rms
                    if self.on_partial:
                        self.on_partial(last_text, self._last_rms)
                time.sleep(0.08)
                continue

            silence_count = 0

            try:
                # Transcribe audio buffer (vad_filter=False because audio energy gate already triggered)
                segments, info = self.model.transcribe(
                    audio_data,
                    beam_size=1,
                    language="en",
                    condition_on_previous_text=False,
                    vad_filter=False,
                )
                text_parts = [segment.text.strip() for segment in segments]
                current_text = " ".join(text_parts).strip()

                if current_text:
                    last_text = current_text
                    if self.on_partial:
                        self.on_partial(current_text, self._last_rms)
                elif last_text:
                    if self.on_partial:
                        self.on_partial(last_text, self._last_rms)

            except Exception as e:
                logger.error("STT transcription error: %s", e)

            # Throttle cycle to maintain ~90-120ms iteration
            elapsed = time.perf_counter() - start_t
            sleep_time = max(0.02, 0.09 - elapsed)
            time.sleep(sleep_time)

    def start(self):
        """Start audio recording and transcription threads."""
        if self.is_running:
            return

        self.is_running = True
        self.load_model()

        if sd is not None:
            try:
                # Detect maximum supported input channels for the default mic
                dev_info = sd.query_devices(kind='input')
                channels_to_open = min(2, max(1, int(dev_info.get("max_input_channels", 1))))

                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=channels_to_open,
                    dtype="float32",
                    blocksize=self.chunk_size,
                    callback=self._audio_callback,
                )
                self.stream.start()
                logger.info(
                    "Microphone audio stream started (%s, %d channels @ %d Hz).",
                    dev_info.get('name', 'Default'), channels_to_open, self.sample_rate
                )
            except Exception as e:
                logger.warning("Could not open microphone stream: %s. Using simulated feed.", e)
                self.stream = None

        self._capture_thread = threading.Thread(target=self._capture_worker, daemon=True, name="STTCapture")
        self._transcribe_thread = threading.Thread(target=self._transcribe_worker, daemon=True, name="STTTranscribe")

        self._capture_thread.start()
        self._transcribe_thread.start()

    def feed_simulated_audio(self, pcm_chunk: np.ndarray):
        """Feed simulated audio chunks (e.g., for automated testing or pipeline tests)."""
        self.audio_queue.put(pcm_chunk)

    def stop(self):
        """Stop streaming and release audio resources."""
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        logger.info("STT streaming engine stopped.")
