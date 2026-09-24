import numpy as np
import sounddevice as sd

# Prefer WASAPI input device to match production pipeline
wasapi_idx = next(
    (i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h.get("name", "").upper()),
    None,
)

selected_dev_idx = None
if wasapi_idx is not None:
    default_dev = sd.query_hostapis()[wasapi_idx].get("default_input_device", -1)
    if default_dev >= 0 and sd.query_devices(default_dev).get("max_input_channels", 0) > 0:
        selected_dev_idx = default_dev
    else:
        for d_idx, d in enumerate(sd.query_devices()):
            if d.get("hostapi") == wasapi_idx and d.get("max_input_channels", 0) > 0:
                selected_dev_idx = d_idx
                break

if selected_dev_idx is not None:
    dev = sd.query_devices(selected_dev_idx)
    device_arg = selected_dev_idx
else:
    dev = sd.query_devices(kind="input")
    device_arg = None

SR = int(dev.get("default_samplerate", 48000))
CH = min(2, max(1, int(dev.get("max_input_channels", 1))))

print(f"Using device [{device_arg}]: {dev.get('name')} @ {SR}Hz (channels={CH})")

def grab(sec):
    buf = []
    with sd.InputStream(
        device=device_arg,
        samplerate=SR,
        channels=CH,
        dtype="float32",
        blocksize=int(SR * 0.05),
        callback=lambda i, f, t, s: buf.append(np.mean(i, axis=1).astype(np.float32) if i.ndim > 1 and i.shape[1] > 1 else i[:, 0].copy()),
    ):
        sd.sleep(int(sec * 1000))
    return np.concatenate(buf)

def ac_rms(x):
    return float(np.sqrt(np.mean((x - x.mean()) ** 2)))

if __name__ == "__main__":
    import wave
    input("BE QUIET (ambient noise calibration), press Enter...")
    amb = grab(4)
    floor = ac_rms(amb)
    print(f"Ambient floor: {floor:.4f}")
    
    with wave.open("scratch/level_meter_amb.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((np.clip(amb, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

    input('Say "open chrome" 3 times normally, then press Enter...')
    sp = grab(6)
    with wave.open("scratch/level_meter_speech.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((np.clip(sp, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

    w = int(SR * 0.1)
    p90 = float(np.percentile([ac_rms(sp[i : i + w]) for i in range(0, len(sp) - w, w // 2)], 90))
    ratio = p90 / max(floor, 1e-6)
    print(f"\n--- RESULTS ---")
    print(f"floor={floor:.4f}  speech_p90={p90:.4f}  ratio={ratio:.1f}")
    if ratio < 3.0:
        print("Verdict: Ratio < 3. Audio SNR is insufficient for reliable recognition.")
    elif ratio >= 5.0:
        print("Verdict: Ratio >= 5. Audio SNR is healthy.")
    else:
        print("Verdict: Ratio between 3 and 5. Marginally acceptable.")
