"""Threaded audio playback.

The orchestrator pushes synthesized + effected float32 buffers; a single
background thread writes them to ONE persistent output stream, sequentially.
Using one long-lived :class:`sounddevice.OutputStream` (instead of repeated
``sd.play``/``sd.wait`` calls) guarantees buffers can never overlap — each
blocking ``write`` finishes before the next begins — which is what stops
Mr. House from talking over himself. It also keeps playback off the main thread
so the next sentence can be synthesized while the current one plays.

Exposes ``is_speaking`` so the display can glitch harder while he talks.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover
    sd = None
    log.warning("sounddevice unavailable (%s); playback disabled.", exc)


class AudioPlayer:
    def __init__(self, sample_rate: int = 22050, device=None) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = False
        self._stream = None
        # Number of buffers still queued or currently playing. is_speaking is
        # derived from this so it's accurate across the whole response.
        self._pending = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return sd is not None

    @property
    def is_speaking(self) -> bool:
        return self._pending > 0

    def start(self) -> None:
        if self._running or sd is None:
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
            )
            self._stream.start()
        except Exception as exc:
            log.error("Could not open output stream: %s", exc)
            self._stream = None
            return
        self._running = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="AudioPlayer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                buf = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if buf is None:  # end-of-response sentinel; nothing to play
                continue
            try:
                if self._stream is not None:
                    # Blocking, gapless, strictly sequential write.
                    self._stream.write(buf.reshape(-1, 1))
            except Exception as exc:
                log.error("Playback error: %s", exc)
            finally:
                with self._lock:
                    self._pending = max(0, self._pending - 1)

    def play(self, audio: np.ndarray) -> None:
        """Queue a buffer for playback (non-blocking)."""
        if audio is None or sd is None or self._stream is None:
            return
        buf = np.asarray(audio, dtype=np.float32)
        # Keep within [-1, 1] so the stream never clips.
        peak = float(np.max(np.abs(buf))) if buf.size else 0.0
        if peak > 1.0:
            buf = buf / peak
        with self._lock:
            self._pending += 1
        self._queue.put(buf)

    def mark_end(self) -> None:
        """Marker kept for API compatibility (speaking state is counter-based)."""
        self._queue.put(None)

    def wait_until_done(self, timeout: Optional[float] = None) -> None:
        """Block until everything queued has finished playing."""
        start = time.time()
        while self._pending > 0:
            if timeout is not None and (time.time() - start) > timeout:
                break
            time.sleep(0.02)

    def clear(self) -> None:
        """Drop anything queued and stop current playback (barge-in / stop)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._pending = 0
        if self._stream is not None:
            try:
                self._stream.abort()
                self._stream.start()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()
        # Drain queue but don't restart the stream.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._pending = 0
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._stream is not None:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._running = False

