"""Per-channel audio probe for default/WASAPI input devices.

Analyzes channels independently WITHOUT np.mean() downmix.
Measures:
- Raw RMS
- Filtered RMS (75 Hz high-pass)
- Speech-band RMS (100-4000 Hz)
- Low-frequency rumble RMS (<100 Hz)
- Silero neural speech probability per channel
"""

import argparse
import sys
import time
import numpy as np
import sounddevice as sd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_controller.audio_utils import filter_highpass, measure_channel_profile, rank_input_candidates
from voice_controller.config import INPUT_API_PREFERENCE, PRE_VAD_HIGHPASS_HZ, SAMPLE_RATE
from voice_controller.actions import get_microphone_permissions, get_microphone_status


def compute_rms(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.0
    arr = arr.astype(np.float32)
    arr = arr - float(np.mean(arr))
    return float(np.sqrt(np.mean(np.square(arr))))


def silero_max_prob(audio_16k: np.ndarray) -> float:
    """Return max Silero speech probability with clean state."""
    if len(audio_16k) < 512:
        return 0.0
    try:
        from voice_controller.vad import get_silero_vad
        vad = get_silero_vad()
        vad.reset()
        max_p = 0.0
        for start in range(0, len(audio_16k) - 512 + 1, 512):
            chunk = audio_16k[start : start + 512]
            p = vad.get_speech_probability(chunk)
            if p > max_p:
                max_p = p
        vad.reset()
        return max_p
    except Exception as err:
        print(f"  [WARN] Silero scoring failed: {err}")
        return float("nan")


def probe_channels(device_arg=None, record_secs=4.0):
    print("=" * 72)
    print("INDEPENDENT CHANNEL PROBE: Realtek / Multi-Channel Diagnostic")
    print("=" * 72)

    # 1. Endpoint & permissions
    mic = get_microphone_status()
    perms = get_microphone_permissions()
    print("\n[WINDOWS CAPTURE STATUS]")
    print(f"  Muted: {mic.get('muted')}")
    print(f"  Input level: {mic.get('level_pct')}%")
    print(f"  Device: {mic.get('device_id') or mic.get('error')}")
    print(f"  Privacy: global={perms.get('global_access')} desktop={perms.get('desktop_apps')}")

    # 2. Query input devices
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()
    ranked = rank_input_candidates(hostapis, devs, api_preference=INPUT_API_PREFERENCE, override=device_arg)
    
    if not ranked:
        print("\n[ERROR] No audio input device found.")
        return

    chosen = ranked[0]
    dev_info = devs[chosen["index"]]
    sr = int(dev_info.get("default_samplerate", 48000))
    channels = min(2, max(1, int(dev_info.get("max_input_channels", 1))))

    print(f"\n[TARGET DEVICE] [{chosen['index']}] {chosen['name']} ({chosen['api']})")
    print(f"  Native SR: {sr} Hz, Max Channels: {channels}")

    print(f"\n[RECORDING] Capturing {record_secs:.1f}s audio... Please speak normally (e.g. 'open chrome').")
    try:
        rec = sd.rec(int(record_secs * sr), samplerate=sr, channels=channels, dtype="float32", device=chosen["index"])
        sd.wait()
    except Exception as err:
        print(f"\n[FATAL] Recording failed: {err}")
        return

    print("\n" + "-" * 72)
    print("PER-CHANNEL ACOUSTIC & SPECTRAL ANALYSIS")
    print("-" * 72)

    from scipy.signal import resample_poly
    import math

    g = math.gcd(SAMPLE_RATE, sr)
    up = SAMPLE_RATE // g
    down = sr // g

    profiles = {}
    probs = {}

    for c in range(channels):
        raw_ch = rec[:, c] if rec.ndim > 1 else rec.flatten()
        raw_rms = compute_rms(raw_ch)
        peak = float(np.max(np.abs(raw_ch))) if len(raw_ch) else 0.0

        # High-pass filtered at 75 Hz
        filt_ch = filter_highpass(raw_ch, sr, cutoff_hz=PRE_VAD_HIGHPASS_HZ)
        filt_rms = compute_rms(filt_ch)

        # 16kHz resampled for spectral & Silero
        ch_16k = resample_poly(filt_ch, up, down).astype(np.float32) if sr != SAMPLE_RATE else filt_ch
        
        # Profile measurement (low band vs speech band)
        prof = measure_channel_profile(ch_16k, SAMPLE_RATE)
        profiles[c] = prof

        # Silero VAD probability
        prob = silero_max_prob(ch_16k)
        probs[c] = prob

        print(f"\n  --- CHANNEL {c} ---")
        print(f"    Raw RMS            : {raw_rms:.6f} (Peak: {peak:.4f})")
        print(f"    Filtered RMS (75Hz): {filt_rms:.6f} ({'suppressed -' + f'{(1 - filt_rms/(raw_rms+1e-9))*100:.1f}%' if filt_rms < raw_rms * 0.9 else 'clean'})")
        print(f"    Low-band (<100Hz)  : {prof['low_band_rms']:.6f}")
        print(f"    Speech-band        : {prof['speech_band_rms']:.6f}")
        print(f"    Speech/Noise SNR   : {prof['snr_db']:.1f} dB")
        print(f"    Silero Max Prob    : {prob:.4f}")

    print("\n" + "=" * 72)
    print("DIAGNOSTIC VERDICT")
    print("=" * 72)

    max_raw = max(compute_rms(rec[:, c] if rec.ndim > 1 else rec) for c in range(channels))
    max_prob = max(probs.values()) if probs else 0.0
    best_speech_ch = max(probs, key=probs.get) if probs else 0

    if max_raw < 0.001:
        print("  [VERDICT] DIGITAL SILENCE / NO SIGNAL")
        print("  Hardware or driver delivered silence. Check Windows microphone mute and permissions.")
    elif max_prob >= 0.50:
        print(f"  [VERDICT] SPEECH DETECTED on CHANNEL {best_speech_ch} (prob = {max_prob:.2f})")
        if channels > 1:
            other_ch = 1 - best_speech_ch
            if profiles[other_ch]["low_band_rms"] > profiles[best_speech_ch]["low_band_rms"] * 2.0:
                print(f"  Notice: Channel {other_ch} has heavy low-frequency rumble ({profiles[other_ch]['low_band_rms']:.4f}).")
                print(f"  The dual-channel speech-first selector successfully locks onto clean Channel {best_speech_ch}.")
    elif max_prob < 0.20:
        if channels > 1 and profiles[1]["low_band_rms"] > profiles[0]["low_band_rms"] * 2.0:
            print("  [VERDICT] NOISE / RUMBLE DOMINATED")
            print(f"  Channel 1 has prominent low-frequency rumble ({profiles[1]['low_band_rms']:.4f} vs CH0 {profiles[0]['low_band_rms']:.4f}).")
            print("  No speech was recognized by Silero in this recording. Speak louder/closer.")
        else:
            print(f"  [VERDICT] SIGNAL PRESENT BUT NO SPEECH DETECTED (max prob = {max_prob:.2f})")
            print("  Audio is reaching Python but Silero classified it as ambient/noise.")
    else:
        print(f"  [VERDICT] AMBIGUOUS SPEECH PROBABILITY (max prob = {max_prob:.2f})")
        print("  Marginal speech detected. Increase microphone input volume or speak closer.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independent channel diagnostic probe")
    parser.add_argument("--device", default=None, help="Device index or name override")
    parser.add_argument("--seconds", type=float, default=4.0, help="Recording duration in seconds")
    args = parser.parse_args()
    probe_channels(device_arg=args.device, record_secs=args.seconds)
