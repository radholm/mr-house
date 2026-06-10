"""Microphone capture.

A thin wrapper around :mod:`sounddevice` that delivers fixed-size blocks of
``int16`` mono audio onto a thread-safe queue. The same stream feeds both the
wake-word detector (always on) and the utterance recorder (after wake), so we
only ever open the mic once.
"""

from __future__ import annotations

import logging
import queue
from typing import Iterator, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - depends on PortAudio being present
    sd = None
    log.warning("sounddevice unavailable (%s); mic capture disabled.", exc)


def _resolve_device(spec) -> Optional[int]:
    """Accept an int index, a substring of the device name, or None."""
    if spec is None or sd is None:
        return None
    if isinstance(spec, int):
        return spec
    for idx, dev in enumerate(sd.query_devices()):
        if isinstance(spec, str) and spec.lower() in dev["name"].lower():
            return idx
    log.warning("Input device %r not found; using default.", spec)
    return None


class MicrophoneStream:
    """Continuously read mono int16 blocks from the default (or chosen) mic."""

    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 1280,
        device=None,
        max_queue: int = 100,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = _resolve_device(device)
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_queue)
        self._stream = None

    @property
    def available(self) -> bool:
        return sd is not None

    def _callback(self, indata, frames, time_info, status):  # noqa: D401
        if status:
            log.debug("Mic status: %s", status)
        # Copy because sounddevice reuses the buffer.
        mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            # Drop oldest to stay real-time.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(mono)
            except queue.Empty:
                pass

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice/PortAudio not available")
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=self.device,
            channels=1,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        log.info("Microphone started @ %d Hz (device=%s).", self.sample_rate, self.device)

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Block for the next audio block; returns None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def blocks(self) -> Iterator[np.ndarray]:
        """Yield blocks forever (until :meth:`stop`)."""
        while self._stream is not None:
            block = self.read()
            if block is not None:
                yield block

    def flush(self) -> None:
        """Discard any buffered audio (call right before recording)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("Microphone stopped.")

    def __enter__(self) -> "MicrophoneStream":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

