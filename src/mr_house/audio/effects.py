"""Voice effects chain using Spotify's ``pedalboard``.

Gives Mr. House his characteristic processed, slightly-synthetic broadcast
voice: high/low-pass filtering, parametric EQ, a touch of distortion/grit,
optional bit-crush, and a small plate-ish reverb. The chain is built once from
config and applied to every synthesized chunk before playback.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

try:
    from pedalboard import (
        Pedalboard,
        HighpassFilter,
        LowpassFilter,
        PeakFilter,
        Distortion,
        Reverb,
        Bitcrush,
        Gain,
    )

    _HAVE_PB = True
except Exception as exc:  # pragma: no cover
    _HAVE_PB = False
    log.warning("pedalboard unavailable (%s); voice FX disabled.", exc)


class VoiceEffects:
    def __init__(self, cfg, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.enabled = bool(getattr(cfg, "enabled", True)) and _HAVE_PB
        self._board = None
        # Pad each buffer with trailing silence so the reverb tail can ring out
        # instead of being cut off at the end of every sentence.
        wet = float(getattr(getattr(cfg, "reverb", None), "wet_level", 0.0) or 0.0)
        self._tail_samples = int(sample_rate * 0.45) if wet > 0.01 else 0
        if not self.enabled:
            return
        try:
            self._board = self._build(cfg)
            log.info("Voice FX chain built (%d stages).", len(self._board))
        except Exception as exc:
            log.error("Failed to build voice FX: %s", exc)
            self.enabled = False

    def _build(self, cfg) -> "Pedalboard":
        plugins = []
        if cfg.highpass_hz:
            plugins.append(HighpassFilter(cutoff_frequency_hz=float(cfg.highpass_hz)))
        for band in cfg.eq or []:
            freq, gain, q = (list(band) + [1.0, 0.0, 1.0])[:3]
            plugins.append(PeakFilter(cutoff_frequency_hz=float(freq),
                                      gain_db=float(gain), q=float(q)))
        if cfg.distortion_db:
            plugins.append(Distortion(drive_db=float(cfg.distortion_db)))
        if getattr(cfg, "bitcrush_depth", 0):
            plugins.append(Bitcrush(bit_depth=int(cfg.bitcrush_depth)))
        if cfg.lowpass_hz:
            plugins.append(LowpassFilter(cutoff_frequency_hz=float(cfg.lowpass_hz)))
        rv = cfg.reverb
        plugins.append(
            Reverb(
                room_size=float(rv.room_size),
                damping=float(rv.damping),
                wet_level=float(rv.wet_level),
                dry_level=float(rv.dry_level),
            )
        )
        # Trim a touch to leave headroom for the reverb tail.
        plugins.append(Gain(gain_db=-1.0))
        return Pedalboard(plugins)

    def apply(self, audio: np.ndarray) -> np.ndarray:
        """Process a float32 mono buffer; returns a processed float32 buffer."""
        if not self.enabled or self._board is None or audio is None:
            return audio
        try:
            buf = audio.astype(np.float32)
            if self._tail_samples:
                buf = np.concatenate([buf, np.zeros(self._tail_samples, dtype=np.float32)])
            out = self._board(buf, self.sample_rate)
            # Guard against clipping introduced by the chain.
            peak = float(np.max(np.abs(out))) if out.size else 0.0
            if peak > 1.0:
                out = out / peak
            return out.astype(np.float32)
        except Exception as exc:
            log.error("Voice FX apply failed: %s", exc)
            return audio

