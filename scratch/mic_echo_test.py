"""Decisive acoustic loopback test: play a tone out the speakers, measure whether the
capture endpoint actually hears it.

If the tone shows up in the capture, the microphone path (driver -> PortAudio -> numpy) is
alive and any remaining failure is gain / capsule / speech level. If it does not, the
endpoint delivers no acoustic signal at all, whatever Windows claims about volume/mute.

Usage:
    python scratch/mic_echo_test.py                 # auto-rank the input device
    python scratch/mic_echo_test.py --in 12         # force an input device index
    python scratch/mic_echo_test.py --freq 440 --amp 0.25 --secs 2.0
"""

import argparse
import math
import sys
import time

import numpy as np
import sounddevice as sd


def band_magnitude(x: np.ndarray, sr: int, freq: float, halfwidth: float = 25.0) -> float:
    """Peak magnitude in a narrow band around `freq` (Hann-windowed FFT)."""
    if x.size < 256:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) / x.size
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    mask = (freqs >= freq - halfwidth) & (freqs <= freq + halfwidth)
    return float(np.max(spec[mask])) if mask.any() else 0.0


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def pick_input() -> int:
    devices = sd.query_devices()
    best, best_score = None, -1.0
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) < 1:
            continue
        score = 1.0 if "realtek" in dev.get("name", "").lower() else 0.0
        if score > best_score:
            best, best_score = idx, score
    if best is None:
        print("ERROR: no input device found.")
        sys.exit(2)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_idx", type=int, default=None)
    ap.add_argument("--out", dest="out_idx", type=int, default=None)
    ap.add_argument("--freq", type=float, default=440.0)
    ap.add_argument("--amp", type=float, default=0.25)
    ap.add_argument("--secs", type=float, default=2.0, help="tone duration")
    ap.add_argument("--pre", type=float, default=1.0, help="baseline capture before the tone")
    ap.add_argument("--post", type=float, default=1.0, help="capture after the tone")
    args = ap.parse_args()
    run(args)


def run(args):
    in_idx = pick_input() if args.in_idx is None else args.in_idx
    in_info = sd.query_devices(in_idx)
    out_idx = sd.default.device[1] if args.out_idx is None else args.out_idx
    try:
        out_info = sd.query_devices(out_idx)
    except Exception:
        out_info = {"name": "(none)", "default_samplerate": 48000}

    in_sr = int(in_info.get("default_samplerate", 48000))
    out_sr = int(out_info.get("default_samplerate", 48000))
    channels = min(2, max(1, int(in_info.get("max_input_channels", 1))))
    blocksize = int(round(0.05 * in_sr))

    print(f"Input : [{in_idx}] {in_info['name']} (API={sd.query_hostapis(in_info['hostapi'])['name']}, "
          f"SR={in_sr}, ch={channels})")
    print(f"Output: [{out_idx}] {out_info['name']} (SR={out_sr})")
    print(f"Tone  : {args.freq:.0f} Hz amp={args.amp} {args.secs:.1f}s | capture "
          f"{args.pre + args.secs + args.post:.1f}s\n")

    captured = []

    def cap_cb(indata, frames, time_info, status):
        if status:
            print(f"  [stream status] {status}")
        captured.append(np.array(indata, dtype=np.float32, copy=True))

    t = np.arange(int(out_sr * args.secs), dtype=np.float32) / float(out_sr)
    tone = (args.amp * np.sin(2.0 * math.pi * args.freq * t)).astype(np.float32)
    fade = min(2000, max(1, tone.size // 8))
    tone[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    tone[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    try:
        in_stream = sd.InputStream(
            device=in_idx, samplerate=in_sr, channels=channels,
            dtype="float32", blocksize=blocksize, callback=cap_cb,
        )
    except Exception as err:
        print(f"ERROR: could not open input stream on [{in_idx}]: {err}")
        sys.exit(3)

    in_stream.start()
    time.sleep(args.pre)

    playable = True
    try:
        sd.play(tone, out_sr, device=out_idx, blocking=False)
    except Exception as err:
        playable = False
        print(f"WARNING: could not play the tone ({err}); measuring room noise only.\n")

    time.sleep(args.secs + args.post)
    in_stream.stop()
    in_stream.close()

    if not captured:
        print("RESULT: ZERO CALLBACKS. The capture stream never delivered audio.")
        sys.exit(4)

    audio = np.concatenate(captured, axis=0)
    n_pre = int(args.pre * in_sr)
    n_tone = int((args.pre + args.secs) * in_sr)
    print(f"Captured {audio.shape[0]} frames ({audio.shape[0] / in_sr:.1f}s) in "
          f"{len(captured)} callbacks ({channels} channel(s)).\n")

    best_ratio = 0.0
    for ch in range(channels):
        raw = audio[:, ch]
        pre = raw[:n_pre] - float(np.mean(raw[:n_pre]))
        dur = raw[n_pre:n_tone] - float(np.mean(raw[n_pre:n_tone]))
        mag_pre = band_magnitude(pre, in_sr, args.freq)
        mag_dur = band_magnitude(dur, in_sr, args.freq)
        ratio = (mag_dur / mag_pre) if mag_pre > 1e-9 else float("inf")
        db = 20.0 * math.log10(mag_dur / mag_pre) if mag_pre > 1e-9 and mag_dur > 0 else float("nan")
        if np.isfinite(ratio):
            best_ratio = max(best_ratio, ratio)
        print(f"ch{ch}: pre rms={rms(pre):.5f} tone rms={rms(dur):.5f} | "
              f"band mag pre={mag_pre:.2e} tone={mag_dur:.2e} | ratio={ratio:.2f}x ({db:+.1f} dB)")

    print()
    if not playable:
        print("RESULT: INCONCLUSIVE (no tone was played). Check the output device above.")
    elif best_ratio >= 4.0:
        print("RESULT: MIC HEARS THE ROOM -> capture path alive; remaining issues are "
              "gain/capsule/speech level, not a dead endpoint.")
    elif best_ratio >= 1.8:
        print("RESULT: WEAK ACOUSTIC COUPLING -> tone barely present. The speakers may simply "
              "be quiet, or the capsule is very insensitive.")
    else:
        print("RESULT: MIC DOES NOT HEAR THE ROOM -> the endpoint delivers no acoustic signal. "
              "Try another input index / host API, raise Windows input volume, disable "
              "'Audio Enhancements', or check for a hardware mute switch.")


if __name__ == "__main__":
    main()
