#!/usr/bin/env python3
"""
DAY 1-2 GATE TOOL. Run this before you write anything else.

Plug the heterodyned output of your breadboard front end into a USB sound card,
run this, and point the mic at a leak. If the band-power bar does not jump when
the leak is on and drop when it is off, STOP - your analog chain is broken and no
amount of software will save it. Usual culprit is op-amp bandwidth.

    python tools/capture.py --list
    python tools/capture.py --device 2
    python tools/capture.py --device 2 --record leak_test.wav --seconds 10

What you should see with a working front end:
  - a clear peak around 2 kHz (= 40 kHz signal minus 38 kHz local oscillator)
  - band power rising 15-30 dB when a leak is in the beam
  - the peak moving if you retune the LO

Controls: [space] mark A/B reference   [r] record   [q] quit
"""

import argparse
import queue
import sys
import time
import wave

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

import matplotlib
import matplotlib.pyplot as plt

BAND = (1200.0, 3200.0)      # expected difference-frequency band, Hz


def list_devices():
    if sd is None:
        print("sounddevice not installed:  pip install sounddevice")
        return
    print(sd.query_devices())


def band_power(spec, freqs, lo, hi):
    m = (freqs >= lo) & (freqs <= hi)
    return float(spec[m].mean()) if m.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--record", default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--band", nargs=2, type=float, default=BAND)
    args = ap.parse_args()

    if args.list:
        list_devices()
        return
    if sd is None:
        sys.exit("sounddevice not installed:  pip install sounddevice")

    q = queue.Queue()
    recorded = []
    state = dict(recording=args.record is not None, ref_db=None, t0=time.time())

    def cb(indata, frames, tinfo, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata[:, 0].copy())

    freqs = np.fft.rfftfreq(args.block, 1.0 / args.sr)
    win = np.hanning(args.block)

    fig, (ax_spec, ax_bar) = plt.subplots(
        2, 1, figsize=(9, 6), gridspec_kw=dict(height_ratios=[3, 1]))
    line, = ax_spec.semilogy(freqs, np.ones_like(freqs) * 1e-6)
    ax_spec.axvspan(args.band[0], args.band[1], color="tab:orange", alpha=0.15)
    ax_spec.set_xlim(0, min(8000, args.sr / 2))
    ax_spec.set_ylim(1e-7, 1e0)
    ax_spec.set_xlabel("Hz (heterodyned)")
    ax_spec.set_ylabel("magnitude")
    ax_spec.set_title("LeakLens front-end monitor  —  space: set reference   q: quit")
    ax_spec.grid(alpha=0.3)

    bar = ax_bar.barh([0], [0], color="tab:red")
    ax_bar.set_xlim(-60, 20)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("band power, dB (relative to reference)")
    ax_bar.grid(alpha=0.3)
    txt = ax_bar.text(0.02, 0.5, "", transform=ax_bar.transAxes, va="center",
                      fontsize=11, fontfamily="monospace")

    def on_key(ev):
        if ev.key == " ":
            state["ref_db"] = state.get("last_db")
            print(f"reference set at {state['ref_db']:.1f} dB")
        elif ev.key == "r":
            state["recording"] = not state["recording"]
            print("recording" if state["recording"] else "stopped")
        elif ev.key == "q":
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", on_key)

    with sd.InputStream(device=args.device, channels=1, samplerate=args.sr,
                        blocksize=args.block, callback=cb):
        print(f"listening on device {args.device} at {args.sr} Hz")
        while plt.fignum_exists(fig.number):
            try:
                block = q.get(timeout=0.5)
            except queue.Empty:
                continue

            if state["recording"]:
                recorded.append(block)
                if args.record and \
                   sum(b.size for b in recorded) >= args.seconds * args.sr:
                    state["recording"] = False
                    data = np.concatenate(recorded)[:int(args.seconds * args.sr)]
                    with wave.open(args.record, "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(args.sr)
                        w.writeframes((np.clip(data, -1, 1) * 32767)
                                      .astype(np.int16).tobytes())
                    print(f"wrote {args.record}")

            spec = np.abs(np.fft.rfft(block * win)) / args.block
            line.set_ydata(np.maximum(spec, 1e-9))

            bp = band_power(spec, freqs, *args.band)
            db = 20 * np.log10(bp + 1e-12)
            state["last_db"] = db
            rel = db - state["ref_db"] if state["ref_db"] is not None else db + 60

            bar[0].set_width(rel)
            bar[0].set_color("tab:green" if rel > 6 else "tab:red")
            peak_hz = freqs[np.argmax(spec)]
            txt.set_text(f"{rel:+6.1f} dB   peak {peak_hz:6.0f} Hz"
                         f"{'   <-- SOURCE DETECTED' if rel > 6 else ''}")

            plt.pause(0.001)

    print("done")


if __name__ == "__main__":
    main()
