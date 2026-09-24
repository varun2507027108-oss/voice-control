"""Per-channel audio probe for default/WASAPI input devices.
Records 4 seconds and measures RMS and peak per channel for stereo,
and tests mono capture.
"""
import sys
import time
import wave
import numpy as np
import sounddevice as sd

def compute_rms(arr: np.ndarray) -> float:
    arr = arr.astype(np.float32)
    arr = arr - float(np.mean(arr))
    return float(np.sqrt(np.mean(np.square(arr))))

def probe_channels():
    print("=" * 60)
    print("Channel Probe: Investigating Stereo vs Mono Capture")
    print("=" * 60)

    # 1. Query input devices
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()
    
    print("\nAvailable Input Devices:")
    for idx, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            api_name = hostapis[d["hostapi"]]["name"]
            print(f"  [{idx}] {d['name']} ({api_name}) - in_ch={d['max_input_channels']}, default_sr={d['default_samplerate']}")

    # Pick candidate device
    default_dev_idx = None
    try:
        default_dev = sd.query_devices(kind="input")
        default_dev_idx = default_dev.get("index", None)
        print(f"\nSystem default input: [{default_dev_idx}] {default_dev.get('name')}")
    except Exception as e:
        print(f"Could not query default device: {e}")

    # Let's test with the primary input device
    test_device = None
    # Look for WASAPI Realtek or default
    for idx, d in enumerate(devs):
        api_name = hostapis[d["hostapi"]]["name"]
        if d.get("max_input_channels", 0) > 0 and "WASAPI" in api_name and "WDM" not in api_name and "KS" not in api_name:
            test_device = idx
            break

    if test_device is None:
        test_device = default_dev_idx

    print(f"\nProbing Device: {test_device} ({devs[test_device]['name'] if test_device is not None else 'Default'})")
    sr = int(devs[test_device]["default_samplerate"]) if test_device is not None else 48000
    
    # 2. Record 4 seconds in STEREO (channels=2)
    print("\n[TEST 1] Recording 4s in STEREO (channels=2)... Please make some noise or speak!")
    try:
        stereo_rec = sd.rec(int(4 * sr), samplerate=sr, channels=2, dtype="float32", device=test_device)
        sd.wait()
        
        ch0 = stereo_rec[:, 0]
        ch1 = stereo_rec[:, 1]
        
        rms_ch0 = compute_rms(ch0)
        rms_ch1 = compute_rms(ch1)
        peak_ch0 = float(np.max(np.abs(ch0)))
        peak_ch1 = float(np.max(np.abs(ch1)))
        
        print(f"  Stereo Results (SR={sr}):")
        print(f"    Channel 0: RMS = {rms_ch0:.6f}, Peak = {peak_ch0:.6f}")
        print(f"    Channel 1: RMS = {rms_ch1:.6f}, Peak = {peak_ch1:.6f}")
        
        # Save stereo wav
        with wave.open("scratch/probe_stereo.wav", "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            int_data = (np.clip(stereo_rec, -1.0, 1.0) * 32767).astype(np.int16)
            wf.writeframes(int_data.tobytes())
        print("  Saved 'scratch/probe_stereo.wav'")
        
        if rms_ch0 < 0.001 and rms_ch1 > 0.01:
            print("  --> WARNING: Channel 0 is DEAD/SILENT while Channel 1 is active!")
        elif rms_ch1 < 0.001 and rms_ch0 > 0.01:
            print("  --> INFO: Channel 1 is silent while Channel 0 is active.")
        else:
            print(f"  --> Ratio ch1/ch0 = {rms_ch1 / (rms_ch0 + 1e-9):.2f}")
    except Exception as err:
        print(f"  Stereo recording failed: {err}")

    # 3. Record 2 seconds in MONO (channels=1)
    print("\n[TEST 2] Recording 2s in MONO (channels=1)...")
    try:
        mono_rec = sd.rec(int(2 * sr), samplerate=sr, channels=1, dtype="float32", device=test_device)
        sd.wait()
        mono_ch = mono_rec[:, 0]
        rms_mono = compute_rms(mono_ch)
        peak_mono = float(np.max(np.abs(mono_ch)))
        print(f"  Mono Results (SR={sr}):")
        print(f"    Channel (Mono): RMS = {rms_mono:.6f}, Peak = {peak_mono:.6f}")
        
        with wave.open("scratch/probe_mono.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            int_data = (np.clip(mono_rec, -1.0, 1.0) * 32767).astype(np.int16)
            wf.writeframes(int_data.tobytes())
        print("  Saved 'scratch/probe_mono.wav'")
    except Exception as err:
        print(f"  Mono recording failed: {err}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    probe_channels()
