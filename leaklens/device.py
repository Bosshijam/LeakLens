"""
THE INTERFACE CONTRACT.

This is the single most important file in the project for your schedule. Freeze
this API today. Build the whole Linux-side application against MockDevice while
the board is in transit, then swap in SerialDevice (or UnoQBridgeDevice) on
bring-up day. If the interface holds, integration is hours instead of days.

Three implementations:

  MockDevice   - no hardware at all. Synthesises a leak at a chosen angle so the
                 raster scanner, heatmap, classifier and UI can all be developed
                 and demoed before anything is delivered.

  SerialDevice - talks a plain ASCII line protocol over a serial port. Works with
                 a spare Arduino Nano / ESP32 TODAY, and works with the UNO Q's
                 MCU over USB CDC later. This is your day-1-to-day-23 workhorse.

  UnoQBridgeDevice - native Arduino_RouterBridge RPC. Thin subclass; fill in once
                 you have verified the exact App Lab API. Deliberately isolated so
                 that uncertainty touches ~20 lines and nothing else.

Wire protocol (ASCII, newline terminated, host always initiates):

    P                 -> "P leaklens <version>"      ping
    A <az> <el>       -> "OK"                        move head, blocks until settled
    E <dwell_ms>      -> "E <rms>"                   sample envelope, return RMS
    W <ms>            -> "W <sr> <n>" + n int16 CSV  capture waveform
    D                 -> "D <metres>"                distance (-1 if no sensor)
    L <0|1>           -> "OK"                        laser on/off
"""

from __future__ import annotations

import time
import math
from abc import ABC, abstractmethod

import numpy as np

PROTOCOL_VERSION = 1


class LeakLensDevice(ABC):
    """Everything the Linux side is allowed to ask of the hardware."""

    # --- geometry limits, override per build -------------------------------
    AZ_MIN, AZ_MAX = -40.0, 40.0
    EL_MIN, EL_MAX = -30.0, 30.0

    @abstractmethod
    def ping(self) -> str: ...

    @abstractmethod
    def set_angle(self, az_deg: float, el_deg: float) -> None:
        """Move the head and BLOCK until it has mechanically settled."""

    @abstractmethod
    def read_envelope(self, dwell_ms: int = 40) -> float:
        """RMS level of the heterodyned band over the dwell window."""

    @abstractmethod
    def read_waveform(self, ms: int = 500) -> tuple[np.ndarray, float]:
        """Return (samples, sample_rate) for classification."""

    def read_distance(self) -> float:
        """Metres to target, or -1.0 if no ranging sensor is fitted."""
        return -1.0

    def set_laser(self, on: bool) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- shared helper -----------------------------------------------------
    def clamp(self, az, el):
        return (min(max(az, self.AZ_MIN), self.AZ_MAX),
                min(max(el, self.EL_MIN), self.EL_MAX))


# ==========================================================================
class MockDevice(LeakLensDevice):
    """
    Simulates a dish-on-pan-tilt looking at one or more leaks.

    Beam model: Gaussian with the given full-width-half-maximum, which is what a
    parabolic dish actually approximates well near boresight. Response falls off
    with 1/r^2 plus atmospheric absorption at ~1.2 dB/m for 40 kHz.

    Pass real recordings in `clips` to make read_waveform return genuine audio,
    so the classifier path is exercised with real data before hardware exists.
    """

    def __init__(self, leaks=((12.0, -6.0, 1.0),), beam_fwhm_deg=2.5,
                 noise=0.02, distance_m=3.0, clips=None, seed=0,
                 settle_s=0.0):
        # leaks: iterable of (az_deg, el_deg, strength)
        self.leaks = list(leaks)
        self.sigma = beam_fwhm_deg / 2.3548
        self.noise = noise
        self.distance_m = distance_m
        self.clips = clips or {}
        self.rng = np.random.default_rng(seed)
        self.settle_s = settle_s
        self._az = 0.0
        self._el = 0.0
        self._laser = False
        self.moves = 0

    def ping(self):
        return f"P leaklens-mock {PROTOCOL_VERSION}"

    def set_angle(self, az_deg, el_deg):
        self._az, self._el = self.clamp(az_deg, el_deg)
        self.moves += 1
        if self.settle_s:
            time.sleep(self.settle_s)

    def _beam_response(self):
        total = 0.0
        for (laz, lel, strength) in self.leaks:
            d2 = (self._az - laz) ** 2 + (self._el - lel) ** 2
            total += strength * math.exp(-d2 / (2 * self.sigma ** 2))
        # range attenuation: spreading + absorption
        atten = (1.0 / max(self.distance_m, 0.3) ** 2) * \
                10 ** (-1.2 * self.distance_m / 20.0)
        return total * atten * 10.0

    def read_envelope(self, dwell_ms=40):
        sig = self._beam_response()
        # noise shrinks with longer dwell, as real averaging does
        n = self.noise / math.sqrt(max(dwell_ms, 1) / 40.0)
        return float(max(0.0, sig + self.rng.normal(0.0, n)))

    def read_waveform(self, ms=500):
        sr = 16000.0
        n = int(sr * ms / 1000.0)
        amp = self._beam_response()
        if self.clips:
            key = "leak" if amp > 3 * self.noise else "background"
            if key in self.clips:
                clip, csr = self.clips[key]
                if clip.size >= n:
                    start = self.rng.integers(0, clip.size - n + 1)
                    return clip[start:start + n].astype(float), csr
        # synthetic: narrowband hiss around 2 kHz + broadband noise floor
        t = np.arange(n) / sr
        carrier = np.sin(2 * np.pi * 2000 * t)
        hiss = self.rng.normal(0, 1, n)
        b = np.exp(-((np.fft.rfftfreq(n, 1 / sr) - 2000) ** 2) / (2 * 300 ** 2))
        hiss = np.fft.irfft(np.fft.rfft(hiss) * b, n=n)
        sig = amp * (0.3 * carrier + hiss / (np.std(hiss) + 1e-12))
        return sig + self.rng.normal(0, self.noise, n), sr

    def read_distance(self):
        return self.distance_m

    def set_laser(self, on):
        self._laser = bool(on)


# ==========================================================================
class SerialDevice(LeakLensDevice):
    """
    ASCII line protocol over pyserial. Works with the Arduino sketch in
    firmware/leaklens_mcu/ on a Nano, an ESP32, or the UNO Q's STM32.
    """

    def __init__(self, port, baud=115200, timeout=3.0, settle_ms=120):
        import serial  # pyserial; imported lazily so MockDevice needs no deps
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.settle_ms = settle_ms
        time.sleep(2.0)          # classic Arduino auto-reset wait
        self.ser.reset_input_buffer()

    def _cmd(self, line, expect_prefix=None):
        self.ser.write((line + "\n").encode())
        self.ser.flush()
        reply = self.ser.readline().decode(errors="replace").strip()
        if not reply:
            raise TimeoutError(f"no reply to {line!r}")
        if expect_prefix and not reply.startswith(expect_prefix):
            raise IOError(f"bad reply to {line!r}: {reply!r}")
        return reply

    def ping(self):
        return self._cmd("P", "P")

    def set_angle(self, az_deg, el_deg):
        az, el = self.clamp(az_deg, el_deg)
        self._cmd(f"A {az:.2f} {el:.2f}", "OK")

    def read_envelope(self, dwell_ms=40):
        r = self._cmd(f"E {int(dwell_ms)}", "E")
        return float(r.split()[1])

    def read_waveform(self, ms=500):
        self.ser.write(f"W {int(ms)}\n".encode())
        self.ser.flush()
        hdr = self.ser.readline().decode().strip()
        if not hdr.startswith("W"):
            raise IOError(f"bad waveform header: {hdr!r}")
        _, sr, n = hdr.split()
        sr, n = float(sr), int(n)
        vals = []
        while len(vals) < n:
            chunk = self.ser.readline().decode().strip()
            if not chunk:
                raise TimeoutError("waveform truncated")
            vals.extend(int(v) for v in chunk.split(",") if v)
        return np.asarray(vals[:n], dtype=float), sr

    def read_distance(self):
        try:
            return float(self._cmd("D", "D").split()[1])
        except Exception:
            return -1.0

    def set_laser(self, on):
        self._cmd(f"L {1 if on else 0}", "OK")

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


# ==========================================================================
class UnoQBridgeDevice(LeakLensDevice):
    """
    Native UNO Q Bridge (MessagePack RPC over the internal serial link).

    NOT VERIFIED - fill this in on bring-up day against the Arduino App Lab
    examples. The exact call signature is the one thing in this codebase that
    genuinely requires hardware in hand. Everything above it is already tested.

    Deliberately kept to a handful of lines so that the unknown is contained.
    """

    def __init__(self, bridge=None):
        if bridge is None:
            from arduino.app_bricks.bridge import Bridge   # verify import path
            bridge = Bridge()
        self.b = bridge

    def ping(self):
        return self.b.call("ping")

    def set_angle(self, az_deg, el_deg):
        az, el = self.clamp(az_deg, el_deg)
        self.b.call("set_angle", az, el)

    def read_envelope(self, dwell_ms=40):
        return float(self.b.call("read_envelope", int(dwell_ms)))

    def read_waveform(self, ms=500):
        res = self.b.call("read_waveform", int(ms))
        return np.asarray(res["samples"], dtype=float), float(res["sr"])

    def read_distance(self):
        return float(self.b.call("read_distance"))

    def set_laser(self, on):
        self.b.call("set_laser", bool(on))


def open_device(spec: str, **kw) -> LeakLensDevice:
    """
    spec = "mock" | "serial:/dev/ttyUSB0" | "bridge"
    Keeps main scripts free of if/else ladders.
    """
    if spec == "mock":
        return MockDevice(**kw)
    if spec.startswith("serial:"):
        return SerialDevice(spec.split(":", 1)[1], **kw)
    if spec == "bridge":
        return UnoQBridgeDevice(**kw)
    raise ValueError(f"unknown device spec: {spec}")
