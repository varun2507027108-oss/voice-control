# Voice Control System: Architecture, Tech Stack & Implementation Guide

A fully local, low-latency, privacy-first voice control pipeline for Windows with a transparent, macOS-styled desktop Notepad overlay.

---

## 1. Project Overview & Design Philosophy

The system provides hands-free Windows desktop control through real-time speech recognition, sub-42ms intent categorization, native operating system automation, and a clean macOS-themed Notepad desktop interface.

### Key Highlights
- **100% Offline & Local**: No audio data or transcripts ever leave your local computer.
- **Hardware-Accelerated ASR**: Runs `faster-whisper` (`small.en`) on CUDA with `float16` precision using your NVIDIA GPU (with graceful CPU fallback).
- **Speech-Aware Dual-Channel Architecture**: Independent per-channel preprocessing and dual-instance Silero neural VAD. Resolves hardware asymmetries where one microphone capsule carries severe low-frequency rumble (e.g. 21–29 Hz Realtek laptop array rumble) while the other carries clean acoustic voice.
- **Utterance-Level Channel Locking**: Once speech onset is confirmed, the system locks onto the speech-bearing capsule for the entirety of the spoken utterance, preventing channel flapping mid-phrase.
- **Clean Real-Time Audio Separation**: PortAudio callback performs *only* sample copy into a bounded queue. Heavy filtering, resampling, VAD inference, and GUI updates are decoupled onto dedicated worker threads to guarantee zero PortAudio underruns.
- **Safe STT Priority Concurrency**: Dedicated single STT worker with non-blocking priority locks. Live partial preview yields immediately when a final utterance arrives.
- **Safe App Allowlist**: Strict executable whitelist (`SAFE_APP_WHITELIST`) prevents arbitrary shell execution from transcribed voice input.
- **Fast Intent Routing**: Categorizes complex natural language voice commands in **~40ms** using `laya.Router`.
- **Zero-Flicker macOS Desktop Notepad**: A borderless, draggable, always-on-top window styled after the native macOS Notes application, featuring traffic light window controls (🔴 🟡 🟢), clean typography, and non-technical status feedback.

---

## 2. Tech Stack & Dependencies

| Layer | Technology | Version | Role / Purpose |
|---|---|---|---|
| **Runtime** | Python | `>=3.11, <3.14` | Primary execution runtime (Python 3.11 / 3.13 supported) |
| **Compute / Acceleration** | PyTorch (`torch`) | `2.6.0+cu124` | CUDA-accelerated tensor operations & neural inference |
| **Hardware Device** | NVIDIA GeForce RTX 3050 Laptop GPU | 6 GB VRAM | GPU target for Whisper FP16 inference |
| **Speech-to-Text (STT)** | `faster-whisper` (CTranslate2) | `1.2.x` | High-throughput GPU/CPU ASR runtime for `small.en` |
| **Model ID** | `small.en` (Whisper) | Pretrained | English ASR model, FP16 on CUDA / INT8 on CPU |
| **Neural VAD** | `silero-vad` (ONNX) | `6.x` | Offline dual-session neural speech boundary detection (~1ms / 32ms frame) |
| **Intent Categorization** | `laya` | `0.1.x` | Ultra-fast local semantic router with sub-42ms latency |
| **Audio Capture** | `sounddevice` | `0.5.x` | PortAudio wrapper capturing native WASAPI audio streams |
| **Signal Processing** | `scipy.signal`, `numpy` | Latest | 75 Hz high-pass rumble filter, polyphase resampler, SNR profiling |
| **Windows Audio** | `pycaw` | `20251023` | Core Audio Windows API wrapper for master endpoint volume control |
| **Display Control** | `screen_brightness_control`| Latest | Multi-monitor display brightness adjustment via WMI/VCP |
| **Input & Media** | `pyautogui` | Latest | Virtual keystroke injection for media controls (`playpause`, `nexttrack`, etc.) |
| **OS Security & Execution** | `ctypes`, `subprocess` | Standard Lib | Calling `user32.LockWorkStation` and allowlisted process execution |
| **GUI & Overlay** | `tkinter` | Standard Lib | Native OS windowing with transparency (`-alpha`) and topmost ordering |

---

## 3. Project Directory Structure

```
c:\Users\varun\voice control\
├── voice_controller/                 # Core Python package
│   ├── __init__.py                   # Package initialization and exports
│   ├── config.py                     # Central configuration (audio, VAD thresholds, styling)
│   ├── audio_utils.py                # Preprocessor, 75Hz highpass filter, channel selector, profiles
│   ├── vad.py                        # Dual-instance Silero VAD (isolated ONNX sessions per channel)
│   ├── engine_stt.py                 # faster-whisper ASR engine (priority lock, structured SttResult)
│   ├── engine_intent.py              # Laya semantic router with custom guardrails
│   ├── actions.py                    # Windows automation dispatchers & strict application allowlist
│   ├── overlay.py                    # Transparent macOS-style Notepad HUD overlay
│   └── main.py                       # Pipeline orchestrator & worker threads
├── tests/
│   ├── fixtures/
│   │   └── open_chrome_16k.wav       # Real speech test fixture (mono 16kHz float32)
│   ├── test_pipeline.py              # Subsystem unit & integration tests (25 test cases)
│   ├── test_audio_pipeline.py        # End-to-end audio pipeline, rumble rejection & security tests
│   └── test_heldout.py               # Held-out semantic generalization / fail-closed suite
├── scratch/
│   ├── live_meter.py                 # Real-time per-channel diagnostic & hardware vs software verdict
│   ├── channel_probe.py              # Dual-channel spectral/energy inspection (no np.mean downmix)
│   └── level_meter.py                # Real-time per-channel CLI VU meter with 75 Hz filtering
├── requirements.txt                  # Pinned dependency requirements
├── pyproject.toml                    # Standard Python project metadata & pytest configuration
└── PROJECT_GUIDE.md                  # Complete technical architecture and implementation guide
```

---

## 4. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph AudioCapture["1. Real-Time Audio Capture"]
        Mic["Microphone Array (Realtek 2-ch)"] -->|Native WASAPI (e.g. 48kHz Stereo)| SD["sounddevice.InputStream"]
        SD -->|Callback Copy Only| RawQueue["Bounded Audio Queue (maxsize=25)"]
    end

    subgraph AudioWorker["2. Audio Worker & Dual-Channel VAD"]
        RawQueue --> PreProc0["CH0: DC removal + HP 75Hz + Resample 16kHz"]
        RawQueue --> PreProc1["CH1: DC removal + HP 75Hz + Resample 16kHz"]
        PreProc0 --> Silero0["Silero VAD Session 0"]
        PreProc1 --> Silero1["Silero VAD Session 1 (Isolated State)"]
        Silero0 --> Decision{"Speech Confirmed?"}
        Silero1 --> Decision
        Decision -->|Lock Channel for Utterance| SpeechBuf["Utterance Speech Buffer"]
        Decision -->|Trailing Silence >= 0.8s| Flush["Flush Utterance & Unlock Channel"]
    end

    subgraph STTLayer["3. Speech Recognition (ASR Worker)"]
        Flush --> STTQueue["STT Priority Queue"]
        STTQueue --> STTLock{"Inference Lock (Final > Live Partial)"}
        STTLock --> Whisper["faster-whisper (small.en, CUDA FP16)"]
        Whisper --> SttRes["Structured SttResult (quality, safe_for_execution)"]
    end

    subgraph IntentLayer["4. Semantic Routing & Execution Safety"]
        SttRes -->|safe_for_execution == True| Laya["laya.Router(preload=True)"]
        Laya --> Guardrails["Context Guardrails (Display vs Audio)"]
        Guardrails --> AllowlistCheck{"Allowlist & Security Valid?"}
        AllowlistCheck -->|Allowed App / Action| Dispatch["actions.execute(action)"]
        AllowlistCheck -->|Blocked / Unrecognized| RejectionNotice["Display Safe Feedback"]
    end

    subgraph UILayer["5. macOS-Themed Desktop Notepad HUD"]
        Dispatch -->|Success Notice| NoteLog["Notepad Note Entry"]
        RejectionNotice --> NoteLog
        SttRes -->|Live Partial / Final Text| NoteLog
    end
```

---

## 5. Subsystem Deep Dives

### 5.1. Audio Stream & Real-Time Audio Callback
- **Copy-Only Callback**: The PortAudio callback in `voice_controller/main.py` performs *only* `raw_audio_queue.put_nowait(indata.copy())`. It does not resample, does not run neural inference, does not score channels, and does not perform GUI calls. If the queue fills up during system lag, the oldest block is dropped with telemetry logged.
- **Device Selection**: `rank_input_candidates()` orders **WASAPI → MME → DirectSound** endpoints, excludes WDM-KS, demotes PortAudio pseudo-devices (`Primary Sound Capture Driver`), and respects `VOICE_CONTROL_DEVICE=<index|name>`.
- **Verified Stream Guarantee**: Opens exactly one production stream and verifies real callbacks and audio health. If no valid endpoint works, the system displays an explicit **NO MICROPHONE** state rather than silently running with a dead fallback.

### 5.2. Two-Channel Speech Detection & Rumble Rejection
- **Rumble Elimination**: On the target Realtek hardware, channel 1 carries intense 21–29 Hz mechanical/fan rumble (+24 dB energy) while channel 0 is acoustic voice. The `ChannelPreprocessor` applies a 75 Hz high-pass Butterworth filter and removes DC before VAD, completely eliminating low-frequency domination.
- **Speech-First Selection**: Speech detection runs *before* channel selection. Dual-instance Silero VAD evaluates both channels independently. The capsule containing confirmed speech is chosen.
- **Utterance-Level Locking**: Once speech onset is detected, `DualChannelSpeechSelector` locks that channel for the entire utterance. Channel flapping is impossible.

### 5.3. Speech-to-Text Engine (`engine_stt.py`)
- **Model**: `faster-whisper` `small.en` loaded on `cuda:0` with `float16` precision (CPU INT8 fallback).
- **Concurrency & Priority**: A non-blocking `_inference_lock` coordinates final utterance transcription and live partial drafts. When final speech arrives, live partial jobs immediately yield and drop, preventing model corruption or contention.
- **Structured Evaluation**: Returns `SttResult` with `text`, `quality_score`, `speech_detected`, and `safe_for_execution`. If speech is too quiet or noisy, the user receives clear diagnostic feedback ("Too quiet", "Low audio quality") instead of a generic "Didn't catch that".
- **Zero Cache Churn**: Removed routine `torch.cuda.empty_cache()` calls from inference hot paths, keeping GPU memory resident and avoiding driver allocation pauses.

### 5.4. Application Execution & Allowlist Security (`actions.py`)
- **Safe Application Allowlist**: Replaced arbitrary executable launching with `SAFE_APP_WHITELIST`. Only authorized applications (e.g. `chrome`, `edge`, `terminal`, `powershell`, `vscode`, `notepad`, `calculator`, `explorer`, `spotify`, `settings`) can be launched. Arbitrary recognized phrases cannot execute arbitrary commands.
- **Windows Automation**: Master volume via `pycaw`, brightness via `screen_brightness_control`, media controls via `pyautogui`, and workstation locking via `ctypes.windll.user32.LockWorkStation`.
- **Mic Gain Safety**: `AUTO_FIX_MIC_LEVEL` defaults to `False`. The system inspects and diagnoses Windows mic volume without modifying user sound settings without explicit opt-in (`VOICE_CONTROL_AUTO_FIX_MIC=1`).

### 5.5. macOS-Themed Notepad Overlay (`overlay.py`)
- **Design & Layout**:
  - Borderless window (`overrideredirect(True)`) with subtle window alpha (`-alpha 0.88`).
  - Dark mode color palette (`#1E1E22` body, `#2A2A2E` header bar).
  - Authentic macOS traffic light window controls (🔴 Minimize, 🟡 Hide, 🟢 Reset).
  - Clean macOS sans-serif font stack (`Segoe UI`, `SF Pro`, `Helvetica Neue`).
- **Throttled Updates**: VU meter updates are coalesced and throttled (`HUD_UPDATE_INTERVAL_S = 0.05`), ensuring the Tkinter event loop remains responsive.

---

## 6. Supported Voice Commands Reference Table

| Intent | Sample Natural Utterances | Executed Action |
|---|---|---|
| `open_browser` | *"Open Google Chrome"*, *"Launch web browser"* | Spawns `chrome.exe` (fallback: `msedge.exe`) |
| `open_terminal` | *"Launch terminal"*, *"Open command prompt"* | Spawns `wt.exe` (fallback: `powershell.exe`) |
| `open_editor` | *"Open VS Code editor"*, *"Launch my code editor"* | Spawns `code.cmd` (fallback: `notepad.exe`) |
| `volume_up` | *"Turn up the volume"*, *"Make it louder"* | Increases Windows audio by +10% |
| `volume_down` | *"Turn down the volume"*, *"Quieter please"* | Decreases Windows audio by -10% |
| `volume_mute` | *"Mute audio"*, *"Silence the sound"* | Toggles Windows master mute |
| `brightness_up` | *"Turn up brightness"*, *"Brighten the screen"* | Increases monitor brightness by +10% |
| `brightness_down` | *"Dim the screen"*, *"Make the display darker"* | Decreases monitor brightness by -10% |
| `media_play_pause` | *"Pause music"*, *"Resume playback"* | Toggles playback via `playpause` key |
| `media_next` | *"Skip track"*, *"Next song"* | Sends `nexttrack` key |
| `media_prev` | *"Previous song"*, *"Rewind track"* | Sends `prevtrack` key |
| `lock_workstation` | *"Lock my computer"*, *"Lock workstation"* | Invokes `LockWorkStation()` |
| `unrecognized` | Anything unrelated to desktop controls | Gracefully logged as unrecognized |

---

## 7. Execution & Diagnostics

### Running the System
```powershell
python -m voice_controller.main
```

### Running the Complete Test Suite
```powershell
python -m pytest tests/
```
Runs both unit/pipeline tests and end-to-end audio pipeline tests (real WAV fixture execution, dual-channel rumble rejection, utterance locking, phase-inversion protection, bounded queue overflow, STT concurrency priority, and application allowlist security).

### Diagnostic Suite (`scratch/`)
1. **`scratch/live_meter.py`**:
   Comprehensive 3s ambient quiet baseline followed by 10s speech capture. Analyzes both channels independently with isolated Silero VAD state resets, reports SNR and spectral balance, and provides an authoritative hardware vs. software verdict.
   ```powershell
   python scratch/live_meter.py
   ```
2. **`scratch/channel_probe.py`**:
   Captures multi-channel audio and compares raw RMS, 75 Hz filtered RMS, speech-band RMS (100 Hz–4 kHz), and low-band rumble (<100 Hz) without any destructive downmixing.
   ```powershell
   python scratch/channel_probe.py
   ```
3. **`scratch/level_meter.py`**:
   Real-time terminal VU meter displaying live per-channel speech-band RMS and peak levels with active 75 Hz high-pass conditioning.
   ```powershell
   python scratch/level_meter.py
   ```
