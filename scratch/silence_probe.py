"""Isolate *why* the Realtek capture endpoint sometimes delivers digital silence.

Windows lets a capture stream open successfully and then hand out pure zeros. This
probe runs four controlled scenarios and prints per-quarter-second RMS so the exact
trigger becomes visible:

    A  single capture stream, quiet room          (control)
    B  second capture stream opened right after A (stream churn / double open)
    C  capture + simultaneous playback stream     (full-duplex / AEC path)
    D  capture with playback starting mid-stream  (driver reconfiguration)

Usage:
    python scratch/silence_probe.py --device 12
    python scratch/silence_probe.py --device 12 --tone-seconds 1.0
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def window_rms(block: np.ndarray, sr: int, channels: int) -> List[float]:
    """Per-quarter-second AC-coupled RMS of every channel of a captured block."""
    out: List[float] = []
    step = max(1, int(0.25 * sr))
    for start in range(0, len(block), step):
        piece = block[start:start + step]
        if len(piece) < step // 2:
            break
        for c in range(channels):
            x = piece[:, c] - float(np.mean(piece[:, c]))
            out.append(float(np.sqrt(np.mean(np.square(x)))))
    return out


def describe(tag: str, block: np.ndarray, sr: int, channels: int) -> None:
    """Print a compact per-channel RMS trace so silent stretches are obvious."""
    values = np.asarray(window_rms(block, sr, channels), dtype=np.float32)
    if values.size == 0:
        print(f"  {tag}: no audio captured")
        return
    grid = values.reshape(-1, channels)
    for idx, row in enumerate(grid):
        levels = " ".join(f"ch{c}={v:.4f}" for c, v in enumerate(row))
        flag = "   <-- SILENCE" if float(np.max(row)) < 1e-4 else ""
        print(f"  {tag} t={idx * 0.25:.2f}s {levels}{flag}")
    print(f"  {tag}: overall max rms={float(np.max(values)):.5f}")


def capture(seconds: float, device: int, sr: int, channels: int) -> Tuple[np.ndarray, List[str]]:
    """Record ``seconds`` of audio with an InputStream and return (audio, status_messages)."""
    frames: List[np.ndarray] = []
    statuses: List[str] = []

    def cb(indata, frame_count, time_info, status):
        if status:
            statuses.append(str(status))
        frames.append(indata.copy())

    with sd.InputStream(device=device, samplerate=sr, channels=channels, dtype="float32",
                        blocksize=int(0.05 * sr), callback=cb):
        time.sleep(seconds)
    if not frames:
        return np.zeros((0, channels), dtype=np.float32), statuses
    return np.concatenate(frames, axis=0), statuses



def main() -> None:
    parser = argparse.ArgumentParser(description="Probe capture silence triggers")
    parser.add_argument("--device", type=int, default=None, help="Capture device index (default: system default)")
    parser.add_argument("--seconds", type=float, default=1.5, help="Duration of each capture scenario")
    parser.add_argument("--tone-seconds", type=float, default=0.6, help="Playback tone duration for C/D")
    args = parser.parse_args()

    dev_info = sd.query_devices(args.device, kind="input")
    sr = int(dev_info.get("default_samplerate", 48000))
    channels = min(2, max(1, int(dev_info.get("max_input_channels", 1))))
    print(f"Capture device: [{args.device}] {dev_info['name']} @ {sr}Hz, {channels}ch")

    out_info = sd.query_devices(kind="output")
    out_sr = int(out_info.get("default_samplerate", 48000))
    print(f"Playback device: {out_info['name']} @ {out_sr}Hz\n")

    tone = (0.06 * np.sin(2 * np.pi * 1000 * np.arange(int(args.tone_seconds * out_sr)) / out_sr)).astype(np.float32)

    print("A) single capture stream (control):")
    audio, statuses = capture(args.seconds, args.device, sr, channels)
    describe("A", audio, sr, channels)
    for status in statuses[:3]:
        print(f"  [status] {status}")

    print("\nB) second capture stream opened immediately after A (stream churn):")
    audio, statuses = capture(args.seconds, args.device, sr, channels)
    describe("B", audio, sr, channels)
    for status in statuses[:3]:
        print(f"  [status] {status}")

    print("\nC) capture + simultaneous playback stream (full-duplex):")
    frames: List[np.ndarray] = []

    def cb_c(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    try:
        with sd.InputStream(device=args.device, samplerate=sr, channels=channels, dtype="float32",
                            blocksize=int(0.05 * sr), callback=cb_c):
            sd.play(tone, samplerate=out_sr, blocking=True)
            time.sleep(max(0.2, args.seconds - args.tone_seconds))
        audio = np.concatenate(frames, axis=0) if frames else np.zeros((0, channels), dtype=np.float32)
    except Exception as err:
        print(f"  full-duplex failed: {err}")
        audio = np.zeros((0, channels), dtype=np.float32)
    describe("C", audio, sr, channels)

    print("\nD) capture alone, then playback starts mid-stream:")
    captures: List[np.ndarray] = []

    def cb_d(indata, frame_count, time_info, status):
        captures.append(indata.copy())

    try:
        with sd.InputStream(device=args.device, samplerate=sr, channels=channels, dtype="float32",
                            blocksize=int(0.05 * sr), callback=cb_d):
            time.sleep(0.7)
            sd.play(tone, samplerate=out_sr, blocking=True)
            time.sleep(0.7)
        audio = np.concatenate(captures, axis=0) if captures else np.zeros((0, channels), dtype=np.float32)
    except Exception as err:
        print(f"  mid-stream playback failed: {err}")
        audio = np.zeros((0, channels), dtype=np.float32)
    describe("D", audio, sr, channels)

    print("\nInterpretation:")
    print("  - A silent: the endpoint itself is dead/muted (Windows settings, privacy, driver).")
    print("  - A ok, B silent: reopening the stream breaks the endpoint (boot stream-churn bug).")
    print("  - A/B ok, C silent: full-duplex/AEC silences capture (other audio apps are playing).")
    print("  - D goes silent exactly when playback starts: the driver reconfigures mid-stream.")


if __name__ == "__main__":
    main()

