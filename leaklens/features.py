"""
Feature extraction for LeakLens.

The signal we work on is ALREADY heterodyned: the 40 kHz ultrasonic band has been
mixed down to roughly 1-4 kHz by the analog front end, so an ordinary sound card
or a slow ADC can capture it.

Two families of features, and the second one matters more than you'd expect:

  SPECTRAL  - log-mel spectrogram statistics. With a narrowband 40 kHz transducer
              there is limited spectral texture available, so this alone is weak.

  MODULATION - the spectrum OF THE ENVELOPE. This is what actually separates the
              classes for us. A leak is a steady turbulent hiss (flat, low
              modulation energy). An impact wrench is violently amplitude-modulated
              at a few Hz. A worn bearing modulates periodically at shaft rate.
              Say this in your report - it is the technically interesting bit and
              it is a direct consequence of the narrowband transducer choice.

No librosa dependency: everything is numpy + scipy so it installs cleanly on the
UNO Q's Debian side.
"""

import numpy as np
from scipy import signal as sps

# --------------------------------------------------------------------------
# mel filterbank
# --------------------------------------------------------------------------

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(n_filters, n_fft, sr, fmin=100.0, fmax=None):
    """Triangular mel filters -> (n_filters, n_fft//2 + 1)."""
    if fmax is None:
        fmax = sr / 2.0
    edges_mel = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2)
    edges_hz = mel_to_hz(edges_mel)
    bins = np.floor((n_fft + 1) * edges_hz / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if mid == lo:
            mid = lo + 1
        if hi == mid:
            hi = mid + 1
        if hi > n_fft // 2:
            hi = n_fft // 2
        if mid >= hi or lo >= mid:
            continue
        fb[i, lo:mid] = np.linspace(0.0, 1.0, mid - lo, endpoint=False)
        fb[i, mid:hi] = np.linspace(1.0, 0.0, hi - mid, endpoint=False)
    return fb


# --------------------------------------------------------------------------
# core transforms
# --------------------------------------------------------------------------

def log_mel_spectrogram(x, sr, n_fft=512, hop=128, n_mels=40,
                        fmin=100.0, fmax=None):
    """Return (n_mels, n_frames) log-mel spectrogram."""
    x = np.asarray(x, dtype=float)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    f, t, Z = sps.stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                       window="hann", padded=False, boundary=None)
    power = np.abs(Z) ** 2
    fb = mel_filterbank(n_mels, n_fft, sr, fmin, fmax)
    mel = fb @ power
    return np.log(mel + 1e-12)


def envelope(x, sr, env_rate=200.0):
    """
    Amplitude envelope, resampled to env_rate Hz.
    Uses the analytic signal magnitude, then decimates.
    """
    x = np.asarray(x, dtype=float)
    env = np.abs(sps.hilbert(x))
    # anti-alias then decimate to env_rate
    factor = max(1, int(round(sr / env_rate)))
    b, a = sps.butter(4, 0.8 / factor)
    env = sps.filtfilt(b, a, env)
    return env[::factor], sr / factor


def modulation_spectrum(env, env_sr, n_bands=16, fmin=0.5, fmax=64.0):
    """
    Log-spaced band energies of the envelope's own spectrum.
    Captures 'how does the loudness wobble over time'.
    """
    env = np.asarray(env, dtype=float)
    env = env - env.mean()
    if env.size < 8:
        return np.zeros(n_bands)
    win = np.hanning(env.size)
    spec = np.abs(np.fft.rfft(env * win)) ** 2
    freqs = np.fft.rfftfreq(env.size, 1.0 / env_sr)

    fmax = min(fmax, env_sr / 2.0 - 1e-6)
    if fmax <= fmin:
        return np.zeros(n_bands)
    edges = np.geomspace(fmin, fmax, n_bands + 1)
    out = np.zeros(n_bands)
    for i in range(n_bands):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out[i] = spec[m].mean() if m.any() else 0.0
    return np.log(out + 1e-12)


# --------------------------------------------------------------------------
# scalar descriptors
# --------------------------------------------------------------------------

def scalar_features(x, sr):
    x = np.asarray(x, dtype=float)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    peak = np.max(np.abs(x)) + 1e-12
    crest = peak / rms

    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) + 1e-12
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    centroid = float((freqs * spec).sum() / spec.sum())
    spread = float(np.sqrt(((freqs - centroid) ** 2 * spec).sum() / spec.sum()))
    # spectral flatness: geometric / arithmetic mean. ~1 = noise-like (a leak),
    # low = tonal (a resonance or a motor whine)
    flatness = float(np.exp(np.log(spec).mean()) / spec.mean())

    zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(int)))))

    return np.array([
        np.log(rms), np.log(crest), centroid, spread, flatness, zcr
    ], dtype=float)


# --------------------------------------------------------------------------
# the one function the rest of the codebase calls
# --------------------------------------------------------------------------

FEATURE_VERSION = 1


def extract(x, sr, n_mels=40, n_mod=16):
    """
    Signal -> 1-D feature vector.

    Layout:  [ mel_mean (n_mels) | mel_std (n_mels) | modulation (n_mod) | scalars (6) ]
    Default total = 40 + 40 + 16 + 6 = 102 features.

    Keep this function identical between training and inference. If you change it,
    bump FEATURE_VERSION and retrain - a silent mismatch here produces a model that
    "works" but predicts nonsense, which is a miserable bug to find at 2 a.m.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("empty signal")
    # normalise amplitude so the classifier keys on TEXTURE not loudness.
    # loudness is handled separately by the leak-rate regressor, which needs it.
    xn = x / (np.sqrt(np.mean(x ** 2)) + 1e-12)

    mel = log_mel_spectrogram(xn, sr, n_mels=n_mels)
    env, env_sr = envelope(xn, sr)
    mod = modulation_spectrum(env, env_sr, n_bands=n_mod)

    return np.concatenate([
        mel.mean(axis=1),
        mel.std(axis=1),
        mod,
        scalar_features(x, sr),      # note: raw x, keeps absolute level info
    ])


def feature_names(n_mels=40, n_mod=16):
    names = [f"mel_mean_{i}" for i in range(n_mels)]
    names += [f"mel_std_{i}" for i in range(n_mels)]
    names += [f"mod_{i}" for i in range(n_mod)]
    names += ["log_rms", "log_crest", "centroid", "spread", "flatness", "zcr"]
    return names
