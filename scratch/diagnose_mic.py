"""Microphone health diagnostic for the Voice Control pipeline.

Captures a short sample from every candidate input device and reports
AC-coupled RMS/peak, per-channel RMS, inter-channel correlation
(phase-inverted mic arrays cancel when averaged) and dead-endpoint detection.

Usage:
    python scratch/diagnose_mic.py                 # 1.5s per device
    python scratch/diagnose_mic.py --speak         # prompt to speak during capture
    python scratch/diagnose_mic.py --seconds 3
    python scratch/diagnose_mic.py --device 12
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API_PREFERENCES = ["WASAPI", "DIRECTSOUND", "MME"]


def ac_rms(x: np.ndarray) -> float:
    """AC-coupled (demeaned) RMS of a 1D float32 array."""
    if x is None or len(x) == 0:
        return 0.0
    x = x.astype(np.float32)
    x = x - float(np.mean(x))
    return float(np.sqrt(np.mean(np.square(x))))


def candidate_devices():
    """Mirror VoiceControllerPipeline._detect_input_device ordering (WASAPI first)."""
    host_apis = sd.query_hostapis()
    devices = sd.query_devices()
    ranked = []
    for pref in API_PREFERENCES:
        for idx, h in enumerate(host_apis):
            up = h.get("name", "").upper()
            if "WDM" in up or "KS" in up:
                continue
            if pref in up and idx not in ranked:
                ranked.append(idx)
                break

    found, seen = [], set()
    for api_idx in ranked:
        h = host_apis[api_idx]
        def_dev = h.get("default_input_device", -1)
        order = ([def_dev] if 0 <= def_dev < len(devices) else []) + list(range(len(devices)))
        for d_idx in order:
            if d_idx in seen or not (0 <= d_idx < len(devices)):
                continue
            d = devices[d_idx]
            if d.get("hostapi") != api_idx or d.get("max_input_channels", 0) <= 0:
                continue
            found.append({"index": d_idx, "name": d.get("name", ""), "api": h.get("name", "")})
            seen.add(d_idx)
    return found


def capture(device_idx: int, seconds: float) -> np.ndarray:
    """Blocking capture of ``seconds`` of float32 audio from a device (all channels)."""
    info = sd.query_devices(device_idx)
    sr = int(info.get("default_samplerate", 48000))
    ch = min(2, max(1, int(info.get("max_input_channels", 1))))
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=ch, dtype="float32", device=device_idx)
    sd.wait()
    return np.asarray(rec, dtype=np.float32)


def analyze(idx: int, name: str, api: str, seconds: float) -> dict:
    """Capture and analyze one device, returning a structured health report."""
    try:
        rec = capture(idx, seconds)
    except Exception as err:
        return {"index": idx, "name": name, "api": api, "error": str(err)}
    if rec.ndim == 1:
        rec = rec.reshape(-1, 1)

    ch_rms = [ac_rms(rec[:, c]) for c in range(rec.shape[1])]
    ch_peak = [float(np.max(np.abs(rec[:, c]))) if len(rec) else 0.0 for c in range(rec.shape[1])]
    downmix = np.mean(rec, axis=1) if rec.shape[1] > 1 else rec[:, 0]
    corr = None
    if rec.shape[1] > 1 and np.std(rec[:, 0]) > 1e-9 and np.std(rec[:, 1]) > 1e-9:
        corr = float(np.corrcoef(rec[:, 0], rec[:, 1])[0, 1])

    rep = {
        "index": idx, "name": name, "api": api,
        "ch_rms": ch_rms, "ch_peak": ch_peak,
        "mix_rms": ac_rms(downmix),
        "mix_peak": float(np.max(np.abs(downmix))) if len(downmix) else 0.0,
        "corr": corr,
        "digital_silence": float(np.max(np.abs(rec))) < 1e-9 if len(rec) else True,
    }
    if corr is not None and corr < -0.5 and max(ch_rms) > 1e-4 and rep["mix_rms"] < 0.4 * max(ch_rms):
        rep["phase_cancel"] = True
    return rep


def verdict(rep: dict) -> str:
    """Human-readable health verdict for one device report."""
    if rep.get("error"):
        return f"UNOPENABLE: {rep['error']}"
    if rep.get("digital_silence"):
        return "DEAD: pure digital silence (0.0 peak)"
    if rep.get("phase_cancel"):
        return "CANCELING: channels phase-inverted (mean() downmix destroys voice)"
    if rep["mix_rms"] < 0.0005:
        return "VERY QUIET: almost no signal (check Windows mic level/mute)"
    return "OK"


def main():
    parser = argparse.ArgumentParser(description="Microphone health diagnostic")
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    print("=" * 72)
    print("MICROPHONE HEALTH DIAGNOSTIC")
    print("=" * 72)

    if args.device is not None:
        info = sd.query_devices(args.device)
        api = sd.query_hostapis()[info["hostapi"]]["name"]
        cands = [{"index": args.device, "name": info["name"], "api": api}]
    else:
        cands = candidate_devices()

    if not cands:
        print("No input devices discovered!")
        return 1

    reports = []
    for cand in cands:
        label = f"[{cand['index']}] {cand['name']} ({cand['api']})"
        if args.speak:
            input(f"\n>>> SPEAK NOW, then press Enter to capture {args.seconds}s from {label}...")
        else:
            print(f"\nCapturing {args.seconds}s from {label} ...")
        rep = analyze(cand["index"], cand["name"], cand["api"], args.seconds)
        reports.append(rep)
        if rep.get("error"):
            print(f"    {verdict(rep)}")
            continue
        print(f"    mix RMS={rep['mix_rms']:.5f}  peak={rep['mix_peak']:.5f}")
        print(f"    per-channel RMS={['%.5f' % r for r in rep['ch_rms']]}")
        if rep.get("corr") is not None:
            print(f"    channel correlation={rep['corr']:+.3f}")
        print(f"    VERDICT: {verdict(rep)}")

    print("\n" + "=" * 72 + "\nSUMMARY\n" + "=" * 72)
    healthy = [r for r in reports if "error" not in r and not r.get("digital_silence") and r["mix_rms"] >= 0.0005]
    for rep in reports:
        print(f"  [{rep['index']:>3}] {rep['name'][:42]:<42} {rep['api'][:16]:<16} {verdict(rep)}")

    if healthy:
        best = max(healthy, key=lambda r: r["mix_rms"])
        print(f"\nRecommended device: [{best['index']}] {best['name']} ({best['api']}) — mix RMS {best['mix_rms']:.5f}")
        if any(r.get("phase_cancel") for r in reports):
            print("NOTE: Phase-canceling array detected — production downmix must not use mean().")
    else:
        print("\nNo healthy device found! Check Windows Sound settings -> Input and mic privacy permissions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
