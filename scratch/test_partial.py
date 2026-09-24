import wave
import numpy as np
from voice_controller.engine_stt import get_stt_engine

engine = get_stt_engine()
with wave.open("tests/fixtures/open_chrome_16k.wav", "rb") as w:
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

print(f"Total audio duration: {len(a)/16000:.2f}s")
for end_s in [0.6, 1.0, 1.4, len(a)/16000]:
    chunk = a[:int(end_s * 16000)]
    res, lat = engine.transcribe_with_latency(chunk)
    print(f"End {end_s:.2f}s -> text: '{res.text}' (trusted={res.trusted}, lat={lat:.1f}ms)")
