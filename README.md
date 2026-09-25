# EchoFlux ⚡ — Real-Time Mid-Sentence Desktop Voice Assistant

EchoFlux is an ultra-low-latency desktop voice assistant powered by **Laya AI**'s non-autoregressive decision engine. It runs continuously in the background, listening to microphone input, streaming live text-to-speech captions to a sleek, transparent, always-on-top PyQt6 HUD overlay, and executing system commands **mid-sentence** without waiting for the speaker to finish.

---

## 🌟 Key Architecture & Highlights

- **Mid-Sentence Execution Pipeline**: Emits partial ASR hypotheses every 80-120ms with a sliding unconsumed text consumer. Commands trigger the moment you utter them (e.g., *"I was thinking of studying so **open VS Code** because..."* immediately opens VS Code mid-sentence and debounces duplicates).
- **Laya AI Non-Autoregressive Decision Engine**: Uses lightweight, sub-15ms non-autoregressive typed primitives:
  - `intent`: `choice` (`["volume_up", "volume_down", "volume_mute", "volume_set", "open_app", "open_browser_tab", "close_tab", "next_tab", "prev_tab", "media_play_pause", "media_next", "take_screenshot", "lock_workstation", "noop"]`)
  - `is_actionable`: `noul` returning $P(\text{true})$ that the phrase contains an imperative command ($\ge 0.82$).
  - `urgency`: `score` ($0$: passive speech, $1$: direct command, $2$: critical/stop).
  - Regex slot-fillers for entity extraction (app names, URLs, percentages, steps).
- **Glassmorphism HUD Overlay (PyQt6)**:
  - Frameless, translucent, click-through capable, always on top.
  - Animated 5-bar audio waveform visualizer pill driven by microphone RMS energy.
  - Live rolling speech transcript with active token highlighting.
  - Transient accented action chips (e.g., `⚡ Opened Spotify`, `🔊 Volume set to 50%`).
  - Auto-fadeout after 3 seconds of silence (fades to 20% opacity; springs back to 95% when speech resumes).
- **Cross-Platform OS Automation**:
  - Audio: Hardware volume control via `pycaw` (Windows) / `osascript` (macOS) / `pactl` (Linux).
  - Apps: Dynamic application launcher and aliases (VS Code, Spotify, Discord, Terminal, Slack, Chrome, etc.).
  - Browser: New tab launcher and web search via default browser.
  - Shortcuts: Close tab, tab cycling, media play/pause/skip, screenshots, workstation locking.

---

## 📁 Project Structure

```
├── requirements.txt            # Pinned dependencies (PyQt6, faster-whisper, pycaw, pyautogui, etc.)
├── conftest.py                 # Pytest configuration
├── main.py                     # Entry point orchestrating threads, signals, and shutdown
├── src/
│   ├── engine/
│   │   └── laya_router.py      # Laya AI ONNX inference & non-autoregressive decision client
│   ├── audio/
│   │   └── stt_stream.py       # 16kHz audio capture & rolling faster-whisper streaming worker
│   ├── core/
│   │   └── stream_dispatcher.py# Mid-sentence sliding-window consumer & token deduplication
│   ├── actions/
│   │   └── system_actions.py   # Cross-platform OS automation (volume, apps, browser, shortcuts)
│   └── ui/
│       └── overlay.py          # PyQt6 glassmorphic floating HUD overlay with live visualizer
└── tests/
    ├── test_laya_router.py     # Intent classification & slot extraction test suite
    ├── test_stream_dispatcher.py # Mid-sentence sliding window and debounce verification
    ├── test_system_actions.py  # System action execution & formatting tests
    └── test_ui_overlay.py      # PyQt6 HUD overlay & bridge tests
```

---

## 🚀 Installation & Setup

### 1. Requirements
- Python 3.11+ (Python 3.13 supported)
- Microphone input device

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Download Laya AI ONNX Model
By default, EchoFlux includes a fast embedded non-autoregressive decision engine that works out of the box with zero downloads. To use custom ONNX weights:
```bash
# Place your Laya ONNX checkpoint in models/
mkdir models
# e.g., models/laya.onnx
```
Specify the model path when starting:
```bash
python main.py --laya-model models/laya.onnx
```

---

## 🎙️ Usage

### Run with Live Microphone Input
```bash
python main.py
```
Options:
- `--model-size`: `tiny.en` (default), `base.en`, or `small.en`
- `--device`: `auto` (default, uses CUDA if available), `cpu`, or `cuda`
- `--laya-model`: Path to custom Laya ONNX model

### Run Interactive Demonstration Mode
Simulates continuous speech with mid-sentence commands to preview the HUD and execution without speaking:
```bash
python main.py --demo
```

### Run Test Suite
```bash
pytest
```

---

## 🗣️ Supported Voice Commands

| Category | Example Phrases | Mid-Sentence Behavior |
| :--- | :--- | :--- |
| **Volume Control** | *"Turn it up"* / *"Volume up"* / *"Louder"* | Raises volume by 10% immediately |
| | *"Lower the volume"* / *"Turn it down"* | Lowers volume by 10% immediately |
| | *"Set volume to 50 percent"* / *"Volume 70"* | Sets absolute volume level |
| | *"Mute audio"* / *"Unmute"* | Toggles system audio mute |
| **App Launching** | *"Open VS Code"* / *"Launch Spotify"* / *"Start Discord"* | Spawns application |
| **Browser & Tabs** | *"Open github.com in a new tab"* | Launches URL in default browser |
| | *"Search for weather in Tokyo"* | Performs Google search |
| | *"Close tab"* | Sends Ctrl+W / Cmd+W |
| | *"Next tab"* / *"Previous tab"* | Cycles through open tabs |
| **Media Shortcuts** | *"Pause music"* / *"Play music"* | Toggles media playback |
| | *"Skip song"* / *"Next track"* | Skips to next track |
| **Utilities** | *"Take a screenshot"* | Saves screenshot to Pictures/Screenshots |
| | *"Lock workstation"* / *"Lock screen"* | Locks workstation (Urgency Score 2) |
