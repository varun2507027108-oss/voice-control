import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import wave
from math import gcd
import numpy as np

PS = (
    "Add-Type -AssemblyName System.Speech;"
    "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
    "$s.SetOutputToWaveFile('tts_cmd.wav');$s.Speak('open chrome');$s.Dispose()"
)
print("Synthesizing 'open chrome' via Windows TTS...")
subprocess.run(["powershell", "-Command", PS], check=True)

with wave.open("tts_cmd.wav", "rb") as w:
    sr = w.getframerate()
    nchannels = w.getnchannels()
    sampwidth = w.getsampwidth()
    frames = w.readframes(w.getnframes())
    print(f"Synthesized WAV: SR={sr}, channels={nchannels}, sampwidth={sampwidth}, total_frames={len(frames)//(sampwidth*nchannels)}")
    x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if nchannels > 1:
        x = x[::nchannels]

if sr != 16000:
    from scipy.signal import resample_poly
    g = gcd(16000, sr)
    x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    print(f"Resampled to 16000Hz: {len(x)} samples ({len(x)/16000:.2f}s)")

# Save resampled WAV for inspection
with wave.open("tts_cmd_16k.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())

from voice_controller.engine_stt import get_stt_engine
out = get_stt_engine().transcribe(x)
print("STT raw output:", repr(out))

from voice_controller.engine_intent import get_intent_router
pred = get_intent_router().predict(out)
print("Intent prediction:", pred)
