"""Live microphone meter: proves whether *your voice* reaches Python, and on which capsule.

Analyzes channels independently without cross-channel contamination.
Resets Silero streaming state between channel tests.
Measures raw RMS, 75Hz filtered RMS, low-band rumble, speech-band energy, and neural VAD probability.
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_controller.actions import get_microphone_permissions, get_microphone_status
from voice_controller.audio_utils import filter_highpass, measure_channel_profile, rank_input_candidates
from voice_controller.config import INPUT_API_PREFERENCE, PRE_VAD_HIGHPASS_HZ, SAMPLE_RATE


def pick_device(device_arg: Optional[str]) -> Dict:
    """Resolve the capture device with the same ranking rules the pipeline uses."""
    host_apis = sd.query_hostapis()
    devices = sd.query_devices()
    ranked = rank_input_candidates(
        host_apis,
        devices,
        api_preference=INPUT_API_PREFERENCE,
        override=device_arg,
    )
    if not ranked:
        raise SystemExit("No audio input device found at all (check Windows Sound settings).")

    print("\n[DEVICES] Input candidates in pipeline order:")
    for pos, cand in enumerate(ranked[:6]):
        info = devices[cand["index"]]
        ch = int(info.get("max_input_channels", 0))
        sr = int(info.get("default_samplerate", 0))
        marker = "  <-- used" if pos == 0 else ""
        print(f"  [{cand['index']}] {cand['name']} ({cand['api']}, {ch}ch, {sr}Hz){marker}")

    chosen = ranked[0]
    print(f"\n[SELECTED] [{chosen['index']}] {chosen['name']} ({chosen['api']})")
    return chosen


def resample_to_16k(audio: np.ndarray, native_sr: int) -> np.ndarray:
    """Resample mono float32 audio to 16 kHz for Silero VAD."""
    if native_sr == SAMPLE_RATE or len(audio) == 0:
        return np.asarray(audio, dtype=np.float32)
    from scipy.signal import resample_poly

    g = math.gcd(SAMPLE_RATE, native_sr)
    return resample_poly(audio, SAMPLE_RATE // g, native_sr // g).astype(np.float32)


def silero_max_prob(audio_16k: np.ndarray) -> float:
    """Return highest Silero speech probability with state reset before and after evaluation."""
    if len(audio_16k) < 512:
        return 0.0
    try:
        from voice_controller.vad import get_silero_vad

        vad = get_silero_vad()
        vad.reset()  # Reset state so previous channel doesn't contaminate
        best = 0.0
        for start in range(0, len(audio_16k) - 512 + 1, 512):
            frame = audio_16k[start : start + 512]
            p = vad.get_speech_probability(frame)
            if p > best:
                best = p
        vad.reset()
        return best
    except Exception as err:
        print(f"[WARN] Silero scoring unavailable: {err}")
        return float("nan")


def rms(audio: np.ndarray) -> float:
    """AC-coupled RMS of a mono signal."""
    if len(audio) == 0:
        return 0.0
    x = np.asarray(audio, dtype=np.float32)
    x = x - float(np.mean(x))
    return float(np.sqrt(np.mean(np.square(x))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Live per-channel microphone meter")
    parser.add_argument("--device", default=None, help="Device index or name substring (overrides ranking)")
    parser.add_argument("--quiet-seconds", type=float, default=3.0, help="Silent baseline duration")
    parser.add_argument("--speak-seconds", type=float, default=10.0, help="Speech test duration")
    args = parser.parse_args()

    print("=" * 78)
    print("VOICE CONTROL - LIVE PER-CHANNEL MICROPHONE METER")
    print("=" * 78)

    mic = get_microphone_status()
    perms = get_microphone_permissions()
    print("\n[WINDOWS] Capture endpoint:")
    print(f"  muted      : {mic.get('muted')}")
    print(f"  input level: {mic.get('level_pct')}%")
    print(f"  endpoint   : {mic.get('device_id') or mic.get('error')}")
    print("[WINDOWS] Microphone consent (privacy):")
    print(
        f"  global={perms.get('global_access')} desktop_apps={perms.get('desktop_apps')} "
        f"machine_policy={perms.get('machine_policy')} this_app={perms.get('this_app')}"
    )

    chosen = pick_device(args.device)
    dev_info = sd.query_devices(chosen["index"])
    native_sr = int(dev_info.get("default_samplerate", SAMPLE_RATE))
    channels = min(2, max(1, int(dev_info.get("max_input_channels", 1))))
    blocksize = int(round(0.05 * native_sr))

    phase = {"name": "quiet"}
    quiet_frames: List[np.ndarray] = []
    speak_frames: List[np.ndarray] = []

    def callback(indata, frames, time_info, status):
        if status:
            print(f"  [stream status] {status}")
        (quiet_frames if phase["name"] == "quiet" else speak_frames).append(indata.copy())

    print(f"\n[OPEN] {chosen['name']} @ {native_sr}Hz, {channels}ch, blocksize={blocksize}")

    def show(seconds_left: float, frames: List[np.ndarray]) -> None:
        """Print one live meter line: per-channel raw & filtered levels."""
        if not frames:
            print(f"  ... no audio callbacks yet ({seconds_left:4.1f}s left)")
            return
        block = frames[-1]
        levels = []
        for c in range(channels):
            ch_data = block[:, c] if block.ndim > 1 else block
            raw_val = rms(ch_data)
            filt_val = rms(filter_highpass(ch_data, native_sr, PRE_VAD_HIGHPASS_HZ))
            bar = "#" * int(min(1.0, raw_val / 0.05) * 8)
            levels.append(f"ch{c}: raw={raw_val:.4f} hp={filt_val:.4f} {bar}")
        print(f"  [{seconds_left:4.1f}s] {' | '.join(levels)}")

    try:
        with sd.InputStream(
            device=chosen["index"],
            samplerate=native_sr,
            channels=channels,
            dtype="float32",
            blocksize=blocksize,
            callback=callback,
        ):
            print("\n" + "-" * 78)
            print(f"PHASE 1 - STAY QUIET for {args.quiet_seconds:.0f}s (measuring room noise)...")
            print("-" * 78)
            deadline = time.time() + args.quiet_seconds
            while time.time() < deadline:
                show(deadline - time.time(), quiet_frames)
                time.sleep(0.5)

            phase["name"] = "speak"
            print("\n" + "-" * 78)
            print(f"PHASE 2 - SPEAK NOW, normally, toward the laptop ({args.speak_seconds:.0f}s).")
            print('          Try: "volume up", "open chrome", "what time is it"')
            print("-" * 78)
            deadline = time.time() + args.speak_seconds
            while time.time() < deadline:
                show(deadline - time.time(), speak_frames)
                time.sleep(0.5)
    except Exception as err:
        print(f"\n[FATAL] Could not open the capture stream: {err}")
        raise SystemExit(1)

    analyse(quiet_frames, speak_frames, channels, native_sr, chosen, mic)


def analyse(
    quiet_frames: List[np.ndarray],
    speak_frames: List[np.ndarray],
    channels: int,
    native_sr: int,
    chosen: Dict,
    mic: Dict,
) -> None:
    """Compare quiet and speech phases per channel independently and print actionable verdict."""
    print("\n" + "=" * 78)
    print("RESULTS & SPECTRAL BREAKDOWN")
    print("=" * 78)

    def stack(frames: List[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros((0, channels), dtype=np.float32)
        return np.concatenate(frames, axis=0)

    quiet = stack(quiet_frames)
    speak = stack(speak_frames)
    print(
        f"Captured {len(quiet)} quiet samples and {len(speak)} speech samples "
        f"({len(speak) / float(native_sr):.1f}s)"
    )

    results = []
    profiles = {}
    for c in range(channels):
        q_raw = quiet[:, c] if quiet.size else np.zeros(0, dtype=np.float32)
        s_raw = speak[:, c] if speak.size else np.zeros(0, dtype=np.float32)

        q_rms = rms(q_raw)
        s_rms = rms(s_raw)

        # 75Hz filtered signals
        q_filt = filter_highpass(q_raw, native_sr, PRE_VAD_HIGHPASS_HZ) if len(q_raw) else q_raw
        s_filt = filter_highpass(s_raw, native_sr, PRE_VAD_HIGHPASS_HZ) if len(s_raw) else s_raw

        q_filt_rms = rms(q_filt)
        s_filt_rms = rms(s_filt)

        # 16kHz resampled for Silero and spectral analysis
        s_16k = resample_to_16k(s_filt, native_sr)
        prof = measure_channel_profile(s_16k, SAMPLE_RATE)
        profiles[c] = prof

        # Silero VAD (with clean reset)
        prob = silero_max_prob(s_16k)
        results.append((c, q_rms, s_rms, q_filt_rms, s_filt_rms, prob, prof))

        ratio = (s_rms / q_rms) if q_rms > 1e-6 else float("inf")
        print(f"\n  [Channel {c}]")
        print(f"    Raw RMS      : quiet={q_rms:.4f}  speech={s_rms:.4f}  (gain ratio x{ratio:.2f})")
        print(f"    Filtered RMS : quiet={q_filt_rms:.4f}  speech={s_filt_rms:.4f}")
        print(f"    Low-band RMS : {prof['low_band_rms']:.4f} (<100Hz rumble)")
        print(f"    Speech-band  : {prof['speech_band_rms']:.4f} (100-4000Hz)")
        print(f"    Band SNR     : {prof['snr_db']:.1f} dB")
        print(f"    Silero Max   : {prob:.4f}")

    valid = [(r[0], r[5]) for r in results if not np.isnan(r[5])]
    best = max(valid, key=lambda item: item[1]) if valid else (0, 0.0)
    speech_rms_max = max(r[2] for r in results) if results else 0.0

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    if not speak_frames or len(speak) == 0:
        print("  NO CALLBACKS: The capture stream never invoked audio callbacks. Endpoint is dead or locked.")
    elif speech_rms_max < 0.001:
        print("  DIGITAL SILENCE: Endpoint delivered 0.0 signal while speaking. Check Windows Sound settings & mute.")
    elif best[1] >= 0.50:
        print(f"  VOICE FOUND on Channel {best[0]} (Silero max prob {best[1]:.2f}).")
        if channels > 1:
            other = 1 - best[0]
            if profiles[other]["low_band_rms"] > profiles[best[0]]["low_band_rms"] * 1.5:
                print(
                    f"  Note: Channel {other} carries heavy low-frequency rumble ({profiles[other]['low_band_rms']:.4f})."
                )
                print(
                    f"  The new dual-channel speech-first selector successfully locks onto clean Channel {best[0]}!"
                )
    elif best[1] < 0.20:
        if channels > 1 and profiles[1]["low_band_rms"] > profiles[0]["low_band_rms"] * 2.0:
            print(
                f"  NOISE ONLY (RUMBLE DOMINATED): Channel 1 has strong ~25Hz acoustic rumble ({profiles[1]['low_band_rms']:.4f})."
            )
            print("  Silero detected no voice on either channel. Raise Windows input level or speak louder.")
        else:
            print(f"  SIGNAL PRESENT BUT NO SPEECH DETECTED: RMS reached {speech_rms_max:.4f}, but Silero max prob was only {best[1]:.2f}.")
            print("  Input may be too soft or muffled. Raise microphone input level to ~100% in Windows Sound.")
    else:
        print(f"  AMBIGUOUS: Moderate speech probability ({best[1]:.2f}). Re-run test speaking with standard cadence.")

    level_pct = mic.get("level_pct")
    if level_pct is not None and float(level_pct) < 50:
        print(f"  [ACTION] Windows input level is only {level_pct}% - raise it to 100% in Settings > Sound > Input.")
    if mic.get("muted"):
        print("  [ACTION] Windows reports capture endpoint is MUTED - unmute it.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
