#!/usr/bin/env python3
"""
Guided dataset collection. Days 4-6.

Prompts for the metadata, records a clip, writes a WAV plus a JSON sidecar into
the layout that leaklens.train expects:

    data/leak/leak_d1.2_p7_r2.0_a0_001.wav
    data/leak/leak_d1.2_p7_r2.0_a0_001.json     {"flow_lpm": 14.2, "distance_m": 2.0, ...}

    python tools/collect_dataset.py --device 2 --out data

SUGGESTED COLLECTION MATRIX  (about 90 minutes of work, and it is the single
highest-value thing you can do while the board is in transit)

  leak        5 orifice sizes x 3 pressures x 3 distances x 2 angles = 90 clips
  tool        blowgun, impact wrench, air ratchet, at 2 distances    = 12 clips
  bearing     a deliberately damaged bearing on a motor, 2 distances = 8 clips
  background  the empty room, compressor running but no leak, fans   = 20 clips

Ten seconds per clip. Get GROUND TRUTH flow for the leak clips using the tank
blowdown method in leaklens.costing.blowdown_flow_lpm - no flow meter needed.

Vary distance and angle deliberately. A model trained at one distance learns
loudness, not texture, and falls apart on demo day.
"""

import argparse
import json
import os
import sys
import time
import wave

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


def record(device, sr, seconds):
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1,
                   device=device, dtype="float32")
    for remaining in range(int(seconds), 0, -1):
        print(f"\r  recording... {remaining:2d}s ", end="", flush=True)
        time.sleep(1.0)
    sd.wait()
    print("\r  done.           ")
    return audio[:, 0]


def save(out_dir, cls, name, x, sr, meta):
    d = os.path.join(out_dir, cls)
    os.makedirs(d, exist_ok=True)
    wav_path = os.path.join(d, name + ".wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    return wav_path


def ask(prompt, default=None, cast=str):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        if not raw:
            continue
        try:
            return cast(raw)
        except ValueError:
            print("  ...could not parse that, try again")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    if sd is None:
        sys.exit("sounddevice not installed:  pip install sounddevice")

    print(__doc__)
    print(sd.query_devices())
    print()

    counter = {}
    while True:
        print("-" * 60)
        cls = ask("class (leak/tool/bearing/background, or 'q' to quit)", "leak")
        if cls.lower() in ("q", "quit", "exit"):
            break

        meta = dict(sample_rate=args.sr, seconds=args.seconds,
                    recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        tag = cls

        if cls == "leak":
            dia = ask("  orifice diameter mm", "1.0", float)
            press = ask("  gauge pressure bar", "7.0", float)
            flow = ask("  measured flow L/min (blank if unknown)", "-1", float)
            meta.update(orifice_mm=dia, pressure_bar=press)
            if flow > 0:
                meta["flow_lpm"] = flow
            tag = f"leak_d{dia}_p{press}"
        else:
            src = ask("  source description", cls)
            meta["source"] = src
            tag = f"{cls}_{src.replace(' ', '-')}"

        dist = ask("  distance m", "2.0", float)
        angle = ask("  off-axis angle deg", "0", float)
        meta.update(distance_m=dist, angle_deg=angle)
        tag += f"_r{dist}_a{angle}"

        counter[tag] = counter.get(tag, 0) + 1
        name = f"{tag}_{counter[tag]:03d}"

        input("  press ENTER when the rig is ready...")
        x = record(args.device, args.sr, args.seconds)

        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)))
        print(f"  rms {rms:.4f}  peak {peak:.3f}")
        if peak > 0.98:
            print("  !! CLIPPING - reduce preamp gain and redo this clip")
            if ask("  keep anyway? (y/n)", "n").lower() != "y":
                continue
        if rms < 1e-4:
            print("  !! almost silent - check the front end is powered")
            if ask("  keep anyway? (y/n)", "n").lower() != "y":
                continue

        meta.update(rms=rms, peak=peak)
        path = save(args.out, cls, name, x, args.sr, meta)
        print(f"  saved {path}")

    print("\ncollection finished. Now run:")
    print("    python -m leaklens.train --data data --out models --report")


if __name__ == "__main__":
    main()
