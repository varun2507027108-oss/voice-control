"""Per-channel level meter for evaluating microphone SNR and signal strength.

Never uses np.mean(axis=1) downmix.
Evaluates CH0 and CH1 independently with 75 Hz high-pass filtering.
"""

import sys
from pathlib import Path
import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_controller.audio_utils import filter_highpass
from voice_controller.config import PRE_VAD_HIGHPASS_HZ, SAMPLE_RATE

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


def grab_stereo(sec: float) -> np.ndarray:
    buf = []

    def cb(indata, frames, time_info, status):
        buf.append(indata.copy())

    with sd.InputStream(
        device=device_arg,
        samplerate=SR,
        channels=CH,
        dtype="float32",
        blocksize=int(SR * 0.05),
        callback=cb,
    ):
        sd.sleep(int(sec * 1000))
    return np.concatenate(buf, axis=0) if buf else np.zeros((0, CH), dtype=np.float32)


def ac_rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean((x - float(np.mean(x))) ** 2)))


if __name__ == "__main__":
    import wave

    print("\n" + "=" * 60)
    print("STEP 1: Measuring ambient noise baseline (STAY QUIET for 3s)...")
    print("=" * 60)
    amb = grab_stereo(3.0)

    for c in range(CH):
        ch_raw = amb[:, c] if amb.ndim > 1 else amb
        ch_filt = filter_highpass(ch_raw, SR, PRE_VAD_HIGHPASS_HZ)
        print(f"  Channel {c} ambient: raw={ac_rms(ch_raw):.5f} | 75Hz high-pass={ac_rms(ch_filt):.5f}")

    print("\n" + "=" * 60)
    print("STEP 2: Measuring speech signal (SPEAK NORMALLY for 5s, e.g. 'open chrome')...")
    print("=" * 60)
    sp = grab_stereo(5.0)

    print("\n--- RESULTS PER CHANNEL ---")
    w = int(SR * 0.1)

    for c in range(CH):
        amb_raw = amb[:, c] if amb.ndim > 1 else amb
        amb_filt = filter_highpass(amb_raw, SR, PRE_VAD_HIGHPASS_HZ)
        amb_floor = max(1e-6, ac_rms(amb_filt))

        sp_raw = sp[:, c] if sp.ndim > 1 else sp
        sp_filt = filter_highpass(sp_raw, SR, PRE_VAD_HIGHPASS_HZ)

        windows = [ac_rms(sp_filt[i : i + w]) for i in range(0, len(sp_filt) - w, w // 2)]
        p90 = float(np.percentile(windows, 90)) if windows else 0.0
        ratio = p90 / amb_floor

        print(f"\n[Channel {c}]")
        print(f"  Ambient Floor (HP): {amb_floor:.5f}")
        print(f"  Speech P90 (HP)   : {p90:.5f}")
        print(f"  Clean SNR Ratio   : {ratio:.1f}x")

        if ratio < 3.0:
            print("  Verdict: Insufficient speech SNR on this channel.")
        elif ratio >= 5.0:
            print("  Verdict: HEALTHY speech SNR on this channel.")
        else:
            print("  Verdict: Marginally acceptable speech SNR.")
    print("=" * 60 + "\n")
