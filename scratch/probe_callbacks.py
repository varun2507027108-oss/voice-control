import sounddevice as sd

devs = sd.query_devices()
hostapis = sd.query_hostapis()

print("Default devices:", sd.default.device)

for idx, d in enumerate(devs):
    if d.get("max_input_channels", 0) > 0:
        api = hostapis[d["hostapi"]]["name"]
        sr = int(d["default_samplerate"])
        ch = min(2, d["max_input_channels"])
        bs = int(sr * 0.05)
        try:
            s = sd.InputStream(device=idx, samplerate=sr, channels=ch, blocksize=bs, callback=lambda a,b,c,d: None)
            s.start()
            s.stop()
            s.close()
            res = "OK"
        except Exception as e:
            res = f"FAILED: {e}"
        print(f"[{idx}] {d['name']} ({api}) -> {res}")
