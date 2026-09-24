"""Inventory every Windows capture endpoint and probe each one for real signal.

Reads: host-API table, Windows default input/output, then a short live probe per
WASAPI/MME input device reporting per-channel rms / DC / speech-band energy.

A device with a flat hiss floor and a large DC offset is almost always a jack with
nothing plugged into it ("floating input") — exactly what a deaf-looking pipeline sees
when the app captures the wrong endpoint.

Usage:
    python scratch/list_inputs.py
    python scratch/list_inputs.py --secs 0.5 --hosts WASAPI
"""

import argparse
import sys
import time

import numpy as np
import sounddevice as sd

PSEUDO = ("primary sound capture", "sound mapper", "microsoft sound mapper")


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def speech_band(x, sr):
    """Energy in 300-3400 Hz relative to total (DC-removed)."""
    if x.size < 256:
        return 0.0
    x = x - float(np.mean(x))
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    band = spec[(freqs >= 300) & (freqs <= 3400)]
    total = spec[(freqs >= 20)]
    if band.size == 0 or total.sum() <= 1e-12:
        return 0.0
    return float(band.sum() / total.sum())


def probe(index, info, secs):
    sr = int(info.get("default_samplerate", 48000))
    ch = min(2, max(1, int(info.get("max_input_channels", 1))))
    blocksize = int(round(0.05 * sr))
    blocks = []
    status_msgs = []

    def cb(indata, frames, t, status):
        if status:
            status_msgs.append(str(status))
        blocks.append(np.array(indata, dtype=np.float32, copy=True))

    try:
        st = sd.InputStream(device=index, samplerate=sr, channels=ch,
                            dtype="float32", blocksize=blocksize, callback=cb)
        st.start()
        time.sleep(secs)
        st.stop()
        st.close()
    except Exception as err:
        return {"error": str(err), "callbacks": 0}

    if not blocks:
        return {"error": "no callbacks", "callbacks": 0}

    audio = np.concatenate(blocks, axis=0)
    per = []
    for c in range(audio.shape[1]):
        col = audio[:, c]
        per.append({
            "rms": rms(col - float(np.mean(col))),
            "peak": float(np.max(np.abs(col))) if col.size else 0.0,
            "dc": float(np.mean(col)),
            "band": speech_band(col, sr),
        })
    return {"error": None, "callbacks": len(blocks), "secs": audio.shape[0] / sr,
            "channels": per, "status": status_msgs[:2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=0.5, help="probe seconds per endpoint")
    ap.add_argument("--hosts", default="WASAPI,MME", help="comma-separated host-API substrings")
    ap.add_argument("--max", dest="max_probes", type=int, default=20)
    args = ap.parse_args()
    wanted = [h.strip().upper() for h in args.hosts.split(",") if h.strip()]

    host_apis = sd.query_hostapis()
    devices = sd.query_devices()
    print("Host APIs:")
    for i, h in enumerate(host_apis):
        print(f"  [{i}] {h['name']:<22} devices={len(h['devices']):<3} "
              f"default_in={h.get('default_input_device', -1)} default_out={h.get('default_output_device', -1)}")
    print(f"\nWindows default input={sd.default.device[0]}  output={sd.default.device[1]}\n")

    print("All devices:")
    for i, d in enumerate(devices):
        if int(d.get("max_input_channels", 0)) <= 0 and int(d.get("max_output_channels", 0)) <= 0:
            continue
        api = host_apis[d["hostapi"]]["name"]
        print(f"  [{i:>2}] in={int(d.get('max_input_channels', 0))} out={int(d.get('max_output_channels', 0))} "
              f"{d.get('default_samplerate', 0):>6.0f}Hz  {api:<20} {d['name']}")

    print(f"\nProbing capture endpoints ({args.hosts}) for {args.secs:.1f}s each:\n")
    print(f"  {'idx':>4} {'api':<10} {'name':<40} {'ch':>2} {'rms':>9} {'peak':>8} {'dc':>9} {'band%':>6}")
    probed = 0
    for i, d in enumerate(devices):
        api = host_apis[d["hostapi"]]["name"]
        if not any(w in api.upper() for w in wanted):
            continue
        if int(d.get("max_input_channels", 0)) <= 0:
            continue
        if probed >= args.max_probes:
            break
        probed += 1
        res = probe(i, d, args.secs)
        if res.get("error"):
            print(f"  {i:>4} {api[:10]:<10} {d['name'][:40]:<40} -- ERROR: {res['error'][:60]}")
            continue
        for c, m in enumerate(res["channels"]):
            print(f"  {i:>4} {api[:10]:<10} {d['name'][:40]:<40} {c:>2} {m['rms']:>9.5f} "
                  f"{m['peak']:>8.4f} {m['dc']:>+9.5f} {m['band'] * 100:>5.1f}%")
    print("\nNotes: a flat low rms with a large +/-dc offset is a floating (unplugged) input.")
    print("       Use the idx shown here with:  $env:VOICE_CONTROL_DEVICE='<idx or name>'")


if __name__ == "__main__":
    sys.exit(main())
