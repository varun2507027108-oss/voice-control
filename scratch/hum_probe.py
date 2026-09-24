"""Classify a capture endpoint: floating (unplugged) jack vs a live, connected microphone.

A floating/unterminated input shows a large DC offset and mains hum (50/100 Hz in India,
60/120 Hz in the US) instead of voice-band energy. A live mic shows a small DC offset and
its energy spread across the 300-3400 Hz speech band.

Usage:
    python scratch/hum_probe.py --in 12 --secs 3
    python scratch/hum_probe.py --in 13 --secs 3     # WDM-KS realtek mic input
"""

import argparse
import math
import sys
import time

import numpy as np
import sounddevice as sd


def analyse(col, sr):
    x = col.astype(np.float64)
    dc = float(np.mean(x))
    x = x - dc
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win)) / max(1.0, x.size)
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)

    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(np.sqrt(np.sum(np.square(spec[m])))) if m.any() else 0.0

    hum = band(45.0, 65.0) + band(95.0, 125.0)
    speech = band(300.0, 3400.0)
    high = band(3400.0, min(8000.0, sr / 2.0))
    low = band(20.0, 100.0)

    # Dominant discrete tones (>10 dB above the local median) in the voice band
    use = (freqs >= 20.0) & (freqs <= 4000.0)
    s = spec[use]
    f = freqs[use]
    med = float(np.median(s)) if s.size else 0.0
    peaks = []
    if s.size > 8 and med > 0:
        for i in range(1, s.size - 1):
            if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] > med * 3.0:
                peaks.append((float(20.0 * math.log10(s[i] / med)), float(f[i]), float(s[i])))
        peaks.sort(reverse=True)

    return {
        "dc": dc, "rms": rms, "peak": peak,
        "hum": hum, "speech": speech, "low": low, "high": high,
        "hum_db": 20.0 * math.log10(hum / speech) if hum > 0 and speech > 0 else float("-inf"),
        "peaks": peaks[:6],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="idx", type=int, required=True)
    ap.add_argument("--secs", type=float, default=3.0)
    args = ap.parse_args()

    info = sd.query_devices(args.idx)
    api = sd.query_hostapis(info["hostapi"])["name"]
    sr = int(info.get("default_samplerate", 48000))
    ch = min(2, max(1, int(info.get("max_input_channels", 1))))
    blocks = []

    def cb(indata, frames, t, status):
        blocks.append(np.array(indata, dtype=np.float32, copy=True))

    print(f"Probing [{args.idx}] {info['name']} | API={api} | SR={sr} | ch={ch} | {args.secs:.1f}s\n")
    st = sd.InputStream(device=args.idx, samplerate=sr, channels=ch, dtype="float32",
                        blocksize=int(round(0.05 * sr)), callback=cb)
    st.start()
    time.sleep(args.secs)
    st.stop()
    st.close()

    if not blocks:
        print("ZERO CALLBACKS.")
        return 1

    audio = np.concatenate(blocks, axis=0)
    print(f"Captured {audio.shape[0]} frames.\n")
    verdicts = []
    for c in range(audio.shape[1]):
        a = analyse(audio[:, c], sr)
        print(f"ch{c}: dc={a['dc']:+.5f} rms={a['rms']:.5f} peak={a['peak']:.4f}")
        print(f"     energy low(<100Hz)={a['low']:.3e} hum(50+100Hz)={a['hum']:.3e} "
              f"speech(300-3400)={a['speech']:.3e} high={a['high']:.3e}")
        print(f"     hum-vs-speech={a['hum_db']:+.1f} dB")
        if a["peaks"]:
            print("     dominant tones: " + ", ".join(f"{freq:.0f}Hz({db:+.1f}dB)" for db, freq, _ in a["peaks"]))
        flat = abs(a["dc"]) > 0.005 and a["hum_db"] > -3.0
        verdicts.append("floating/unplugged" if flat else ("live-ish" if a["speech"] > 0 else "silent"))
        print(f"     -> looks like: {verdicts[-1]}\n")

    print("Interpretation: 'floating/unplugged' means nothing is electrically connected to that "
          "input (or the jack/cable is broken). 'live-ish' means the endpoint carries real "
          "acoustic energy in the voice band.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
