"""End-of-utterance detection.

After the wake word fires we record until the speaker stops. ``webrtcvad`` gives
us per-frame voiced/unvoiced decisions on 10/20/30 ms frames; we wrap it in a
small state machine that:

  * waits up to ``start_timeout_ms`` for speech to begin,
  * keeps recording while there is voice,
  * stops once ``silence_ms`` of trailing silence is seen,
  * hard-stops at ``max_utterance_ms``.

If ``webrtcvad`` isn't installed we fall back to a simple RMS energy gate so the
assistant still works (just a little less robustly).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import webrtcvad
except Exception as exc:  # pragma: no cover
    webrtcvad = None
    log.warning("webrtcvad unavailable (%s); using RMS energy VAD.", exc)


class UtteranceRecorder:
    """Collect audio until the user finishes speaking."""

    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,
        silence_ms: int = 800,
        max_utterance_ms: int = 15000,
        start_timeout_ms: int = 6000,
        frame_ms: int = 20,
        energy_threshold: float = 500.0,
        start_speech_ms: int = 150,
        min_voiced_ms: int = 350,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_len = int(sample_rate * frame_ms / 1000)
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.max_frames = max_utterance_ms // frame_ms
        self.start_frames = start_timeout_ms // frame_ms
        self.energy_threshold = energy_threshold
        # Require a short run of voiced frames before we accept that speech has
        # begun (rejects single-frame noise blips that trip the VAD).
        self.start_speech_frames = max(1, start_speech_ms // frame_ms)
        # An utterance must contain at least this much *voiced* audio to be kept,
        # otherwise it's treated as noise/silence and discarded (this is what
        # stops Whisper from hallucinating "thank you" out of ambient noise).
        self.min_voiced_frames = max(1, min_voiced_ms // frame_ms)
        self._vad = webrtcvad.Vad(aggressiveness) if webrtcvad else None

    def _is_speech(self, frame: np.ndarray) -> bool:
        if self._vad is not None:
            try:
                return self._vad.is_speech(frame.tobytes(), self.sample_rate)
            except Exception:
                pass
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)) + 1e-9)
        return rms > self.energy_threshold

    def record(self, source, start_timeout_ms: Optional[int] = None) -> Optional[np.ndarray]:
        """Pull frames from *source* (an iterable / MicrophoneStream) until done.

        ``source`` may be a :class:`MicrophoneStream` (we call ``.read``) or any
        iterator yielding int16 blocks. Returns the captured int16 waveform, or
        ``None`` if the speaker never started. ``start_timeout_ms`` overrides how
        long to wait for speech to begin (used for the shorter follow-up window).
        """
        read = getattr(source, "read", None)
        start_frames = (
            (start_timeout_ms // self.frame_ms) if start_timeout_ms is not None
            else self.start_frames
        )

        def next_block() -> Optional[np.ndarray]:
            if read is not None:
                return read(timeout=1.0)
            try:
                return next(source)
            except StopIteration:
                return None

        # Re-frame arbitrary block sizes into fixed VAD frames.
        buffer = np.zeros(0, dtype=np.int16)
        collected: list[np.ndarray] = []
        preroll: list[np.ndarray] = []      # recent frames kept before onset
        started = False
        trailing_silence = 0
        frames_seen = 0
        frames_since_start = 0
        consecutive_voiced = 0
        voiced_frames = 0
        preroll_max = self.start_speech_frames + 2

        def finish() -> Optional[np.ndarray]:
            # Discard utterances that don't contain enough actual speech.
            if not collected or voiced_frames < self.min_voiced_frames:
                log.info(
                    "Discarded non-speech capture (voiced=%d < %d).",
                    voiced_frames, self.min_voiced_frames,
                )
                return None
            return np.concatenate(collected)

        while True:
            block = next_block()
            if block is None:
                if started:
                    break
                continue
            buffer = np.concatenate([buffer, block.astype(np.int16)])

            while len(buffer) >= self.frame_len:
                frame = buffer[: self.frame_len]
                buffer = buffer[self.frame_len :]
                speech = self._is_speech(frame)

                if not started:
                    frames_since_start += 1
                    # Keep a short pre-roll so we don't clip the word onset.
                    preroll.append(frame)
                    if len(preroll) > preroll_max:
                        preroll.pop(0)
                    consecutive_voiced = consecutive_voiced + 1 if speech else 0
                    if consecutive_voiced >= self.start_speech_frames:
                        started = True
                        collected.extend(preroll)        # include the onset
                        voiced_frames += consecutive_voiced
                        preroll = []
                    elif frames_since_start > start_frames:
                        log.info("No speech detected; giving up.")
                        return None
                    continue

                collected.append(frame)
                frames_seen += 1
                if speech:
                    voiced_frames += 1
                    trailing_silence = 0
                else:
                    trailing_silence += 1

                if trailing_silence >= self.silence_frames:
                    log.debug("End of utterance (silence).")
                    return finish()
                if frames_seen >= self.max_frames:
                    log.debug("End of utterance (max length).")
                    return finish()

        return finish()

