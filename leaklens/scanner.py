"""
Raster scan controller and heatmap assembly.

This is the heart of the Linux-side application. It knows nothing about whether
it is talking to real hardware or MockDevice - that is the whole point.

Scan strategy is two-pass, which matters for your 30-second frame budget:

  COARSE  - wide grid, short dwell. Finds regions of interest fast.
  FINE    - dense grid over only the hot regions, long dwell for better SNR.

A single-pass 30x20 grid at 50 ms dwell is 30 s. Two-pass typically resolves the
same leak to better precision in about half that, because most of the field of
view is empty and does not deserve equal attention. Mention this in the report;
it is a real engineering decision, not an implementation detail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScanResult:
    grid: np.ndarray               # (n_el, n_az) envelope readings
    az_axis: np.ndarray            # degrees
    el_axis: np.ndarray            # degrees
    peak_az: float
    peak_el: float
    peak_value: float
    snr: float                     # peak over background median, in dB
    distance_m: float
    duration_s: float
    waveform: np.ndarray | None = None
    waveform_sr: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def has_source(self) -> bool:
        """Is there anything here at all, before we ask what it is."""
        return self.snr > 6.0


def _grid_scan(dev, az_axis, el_axis, dwell_ms, progress=None):
    grid = np.zeros((len(el_axis), len(az_axis)))
    total = grid.size
    done = 0
    for i, el in enumerate(el_axis):
        # serpentine: reverse alternate rows so the head never flies back across
        # the whole field of view. Saves real seconds on a slow servo.
        cols = range(len(az_axis)) if i % 2 == 0 else range(len(az_axis) - 1, -1, -1)
        for j in cols:
            dev.set_angle(az_axis[j], el)
            grid[i, j] = dev.read_envelope(dwell_ms)
            done += 1
            if progress and done % 10 == 0:
                progress(done / total)
    return grid


def _peak(grid, az_axis, el_axis):
    idx = np.unravel_index(np.argmax(grid), grid.shape)
    i, j = idx
    # parabolic sub-pixel interpolation - gets you well below one grid step,
    # which is how you claim precision finer than your raster spacing
    def interp(v_lo, v_mid, v_hi):
        denom = (v_lo - 2 * v_mid + v_hi)
        if abs(denom) < 1e-12:
            return 0.0
        return 0.5 * (v_lo - v_hi) / denom

    daz = 0.0
    if 0 < j < grid.shape[1] - 1:
        daz = interp(grid[i, j - 1], grid[i, j], grid[i, j + 1])
    dele = 0.0
    if 0 < i < grid.shape[0] - 1:
        dele = interp(grid[i - 1, j], grid[i, j], grid[i + 1, j])

    step_az = az_axis[1] - az_axis[0] if len(az_axis) > 1 else 0.0
    step_el = el_axis[1] - el_axis[0] if len(el_axis) > 1 else 0.0
    return (float(az_axis[j] + daz * step_az),
            float(el_axis[i] + dele * step_el),
            float(grid[i, j]))


def scan(dev,
         az_range=(-30.0, 30.0), el_range=(-20.0, 20.0),
         coarse_step=5.0, fine_step=1.0,
         coarse_dwell=25, fine_dwell=80,
         fine_halfwidth=6.0,
         capture_waveform=True,
         progress=None) -> ScanResult:
    """
    Two-pass scan. Returns a ScanResult ready for classification and display.
    """
    t0 = time.time()

    az_axis = np.arange(az_range[0], az_range[1] + 1e-9, coarse_step)
    el_axis = np.arange(el_range[0], el_range[1] + 1e-9, coarse_step)
    coarse = _grid_scan(dev, az_axis, el_axis, coarse_dwell, progress)

    caz, cel, _ = _peak(coarse, az_axis, el_axis)

    # fine pass around the coarse peak
    faz_axis = np.arange(max(az_range[0], caz - fine_halfwidth),
                         min(az_range[1], caz + fine_halfwidth) + 1e-9, fine_step)
    fel_axis = np.arange(max(el_range[0], cel - fine_halfwidth),
                         min(el_range[1], cel + fine_halfwidth) + 1e-9, fine_step)
    fine = _grid_scan(dev, faz_axis, fel_axis, fine_dwell, progress)

    paz, pel, pval = _peak(fine, faz_axis, fel_axis)

    # SNR against the coarse-map background, excluding the hot region
    mask = np.ones_like(coarse, dtype=bool)
    for i, el in enumerate(el_axis):
        for j, az in enumerate(az_axis):
            if abs(az - paz) < fine_halfwidth and abs(el - pel) < fine_halfwidth:
                mask[i, j] = False
    bg = np.median(coarse[mask]) if mask.any() else np.median(coarse)
    snr = 20.0 * np.log10((pval + 1e-12) / (bg + 1e-12))

    # park on the peak, grab audio for classification
    dev.set_angle(paz, pel)
    wf, sr = (None, None)
    if capture_waveform:
        wf, sr = dev.read_waveform(600)

    return ScanResult(
        grid=fine, az_axis=faz_axis, el_axis=fel_axis,
        peak_az=paz, peak_el=pel, peak_value=pval,
        snr=float(snr), distance_m=dev.read_distance(),
        duration_s=time.time() - t0,
        waveform=wf, waveform_sr=sr,
        meta=dict(coarse_grid=coarse, coarse_az=az_axis, coarse_el=el_axis,
                  background=float(bg)),
    )


def heatmap_rgba(grid, cmap="inferno", floor_pct=60.0, gamma=0.7):
    """
    Grid -> RGBA array suitable for overlaying on a camera frame.

    floor_pct discards the lower part of the dynamic range so the background
    stays transparent and only real energy shows. Without this the whole image
    glows faintly and looks like a bug.
    """
    import matplotlib.cm as cm

    g = np.asarray(grid, dtype=float)
    lo = np.percentile(g, floor_pct)
    hi = g.max()
    if hi - lo < 1e-12:
        norm = np.zeros_like(g)
    else:
        norm = np.clip((g - lo) / (hi - lo), 0.0, 1.0) ** gamma

    rgba = (cm.get_cmap(cmap)(norm) * 255).astype(np.uint8)
    rgba[..., 3] = (norm * 220).astype(np.uint8)   # alpha follows intensity
    return rgba


def overlay_on_frame(frame_bgr, result: ScanResult, fov_deg=(60.0, 45.0),
                     alpha=0.75):
    """
    Composite the acoustic heatmap onto a camera frame.

    fov_deg is the camera's horizontal and vertical field of view - measure it
    once for your webcam (photograph a metre rule at a known distance) and hard
    code it. Getting this wrong is why the hot spot lands next to the fitting
    instead of on it.
    """
    import cv2

    h, w = frame_bgr.shape[:2]
    rgba = heatmap_rgba(result.grid)

    # where the scanned window sits inside the camera frame
    def to_px(az, el):
        x = (0.5 + az / fov_deg[0]) * w
        y = (0.5 - el / fov_deg[1]) * h
        return x, y

    x0, y1 = to_px(result.az_axis[0], result.el_axis[0])
    x1, y0 = to_px(result.az_axis[-1], result.el_axis[-1])
    x0, x1 = int(round(min(x0, x1))), int(round(max(x0, x1)))
    y0, y1 = int(round(min(y0, y1))), int(round(max(y0, y1)))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return frame_bgr

    patch = cv2.resize(rgba, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    rgb = patch[..., :3][..., ::-1].astype(float)     # RGBA -> BGR
    a = (patch[..., 3:4].astype(float) / 255.0) * alpha

    out = frame_bgr.copy().astype(float)
    out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a) + rgb * a

    # crosshair on the peak
    px, py = to_px(result.peak_az, result.peak_el)
    cv2.drawMarker(out, (int(px), int(py)), (255, 255, 255),
                   cv2.MARKER_CROSS, 26, 2)
    return out.astype(np.uint8)
