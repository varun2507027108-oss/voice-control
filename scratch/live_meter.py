"""Live microphone meter: proves whether *your voice* reaches Python, and on which capsule.

Answers the only questions that matter when the app "cannot hear you":

    1. Which device did the pipeline pick, and does Windows allow it?
    2. Does the endpoint deliver *any* signal at all?
    3. Which microphone-array channel carries the voice, and can Silero VAD hear it?

Usage:
    python scratch/live_meter.py                        # 3s quiet baseline + 10s speech
    python scratch/live_meter.py --speak-seconds 15
    python scratch/live_meter.py --device 12            # pin a device index
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_controller.actions import get_microphone_permissions, get_microphone_status
from voice_controller.audio_utils import channel_scores, rank_input_candidates
from voice_controller.config import INPUT_API_PREFERENCE, SAMPLE_RATE


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
    """Resample mono float32 audio to 16 kHz for the Silero VAD (no-op when already 16 kHz)."""
    if native_sr == SAMPLE_RATE or len(audio) == 0:
        return np.asarray(audio, dtype=np.float32)
    from scipy.signal import resample_poly

    g = math.gcd(SAMPLE_RATE, native_sr)
    return resample_poly(audio, SAMPLE_RATE // g, native_sr // g).astype(np.float32)


def silero_max_prob(audio_16k: np.ndarray) -> float:
    """Return the highest Silero speech probability across 32 ms frames (NaN when unavailable)."""
    if len(audio_16k) < 512:
        return 0.0
    try:
        import torch
        from voice_controller.vad import get_silero_vad

        model = get_silero_vad().model
        if model is None:
            return float("nan")
        best = 0.0
        for start in range(0, len(audio_16k) - 512 + 1, 512):
            frame = audio_16k[start:start + 512].astype(np.float32)
            best = max(best, float(model(torch.from_numpy(frame), 16000)[0, 0]))
        return best
    except Exception as err:  # pragma: no cover - model availability varies
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
    print("VOICE CONTROL - LIVE MICROPHONE METER")
    print("=" * 78)

    mic = get_microphone_status()
    perms = get_microphone_permissions()
    print("\n[WINDOWS] Capture endpoint:")
    print(f"  muted      : {mic.get('muted')}")
    print(f"  input level: {mic.get('level_pct')}%   (raise to ~100% for best recognition)")
    print(f"  endpoint   : {mic.get('device_id') or mic.get('error')}")
    print("[WINDOWS] Microphone consent (privacy):")
    print(f"  global={perms.get('global_access')} desktop_apps={perms.get('desktop_apps')} "
          f"machine_policy={perms.get('machine_policy')} this_app={perms.get('this_app')}")

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
        """Print one live meter line: per-channel broadband RMS and speech-band score."""
        if not frames:
            print(f"  ... no audio callbacks yet ({seconds_left:4.1f}s left)")
            return
        block = frames[-1]
        level = [rms(block[:, c]) if block.ndim > 1 else rms(block) for c in range(channels)]
        band = channel_scores(block, native_sr if block.ndim > 1 else None)
        bars = " | ".join(
            f"ch{c}: {level[c]:.4f} " + "#" * int(min(1.0, level[c] / 0.05) * 12)
            for c in range(channels)
        )
        band_txt = " ".join(f"ch{i}={float(v):.4f}" for i, v in enumerate(band)) if band.size else "n/a"
        print(f"  [{seconds_left:4.1f}s] {bars}   speech-band: {band_txt}")

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


def analyse(quiet_frames: List[np.ndarray], speak_frames: List[np.ndarray], channels: int,
            native_sr: int, chosen: Dict, mic: Dict) -> None:
    """Compare the quiet and speech phases per channel and print an actionable verdict."""
    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)

    def stack(frames: List[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros((0, channels), dtype=np.float32)
        return np.concatenate(frames, axis=0)

    quiet = stack(quiet_frames)
    speak = stack(speak_frames)
    print(f"Captured {len(quiet)} quiet samples and {len(speak)} speech samples "
          f"({len(speak) / float(native_sr):.1f}s)")

    results = []
    for c in range(channels):
        quiet_rms = rms(quiet[:, c]) if quiet.size else 0.0
        speech_rms = rms(speak[:, c]) if speak.size else 0.0
        source = speak[:, c] if speak.size else np.zeros(1, dtype=np.float32)
        prob = silero_max_prob(resample_to_16k(source, native_sr))
        results.append((c, quiet_rms, speech_rms, prob))
        ratio = (speech_rms / quiet_rms) if quiet_rms > 1e-6 else float("inf")
        print(f"  ch{c}: quiet rms={quiet_rms:.4f}  speech rms={speech_rms:.4f}  "
              f"speech/quiet x{ratio:.2f}  silero max={prob:.3f}")

    valid = [(c, p) for c, _q, _s, p in results if not np.isnan(p)]
    best = max(valid, key=lambda item: item[1]) if valid else (0, float("nan"))
    speech_rms_max = max(r[2] for r in results) if results else 0.0
    if speak.size and channels:
        loud_band = int(max(range(channels), key=lambda c: float(np.mean(np.abs(np.diff(speak[:, c]))))))
    else:
        loud_band = 0

    verdicts: List[str] = []
    if speak.size == 0 or speech_rms_max < 0.002:
        verdicts.append(
            "NO SIGNAL: the endpoint delivered near-silence while you spoke. That is a Windows/driver "
            "problem, not a code problem - check input level/mute (Settings > Sound > Input), the "
            "Realtek/array 'Microphone Effects' or audio enhancements, and that no other app holds the mic."
        )
    elif best[1] >= 0.5:
        verdicts.append(
            f"VOICE FOUND on channel {best[0]} (Silero max prob {best[1]:.2f}); loudest speech-band channel "
            f"is {loud_band}. If the pipeline latches a different channel, the capsule choice is the bug."
        )
    elif best[1] < 0.2:
        verdicts.append(
            f"ENERGY BUT NO SPEECH: audio arrives (rms up to {speech_rms_max:.4f}) yet Silero scores it as "
            f"non-speech ({best[1]:.2f}). Typical of input that is far too quiet or distorted - raise the "
            "Windows microphone level to ~100% and speak closer to the array."
        )
    else:
        verdicts.append("INCONCLUSIVE: re-run and speak clearly for the whole speech phase.")

    level_pct = mic.get("level_pct")
    if level_pct is not None and float(level_pct) < 50:
        verdicts.append(f"Windows input level is only {level_pct}% - raise it (Settings > Sound > Input).")
    if mic.get("muted"):
        verdicts.append("Windows reports the capture endpoint as MUTED - unmute it.")

    print("\nVERDICT:")
    for line in verdicts:
        print(f"  - {line}")

    print("\nMachine-readable summary:")
    print("  " + json.dumps({
        "device": chosen["name"],
        "api": chosen["api"],
        "native_sr": native_sr,
        "channels": channels,
        "per_channel": [
            {"channel": c, "quiet_rms": round(q, 5), "speech_rms": round(s, 5),
             "silero_max": None if np.isnan(p) else round(p, 3)}
            for c, q, s, p in results
        ],
    }))


if __name__ == "__main__":
    main()

