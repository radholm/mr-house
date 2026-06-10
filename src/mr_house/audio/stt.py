"""Speech-to-text via faster-whisper (local, CTranslate2 backend).

Fast and accurate; ``int8`` on CPU is good enough for real-time short
utterances. We take an int16 waveform (the recorded question) and return text.
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np

log = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
except Exception as exc:  # pragma: no cover
    WhisperModel = None
    log.warning("faster-whisper unavailable (%s); STT disabled.", exc)


# Phrases Whisper notoriously hallucinates from silence / background noise.
# If the WHOLE transcription reduces to one of these, we treat it as nothing.
_HALLUCINATIONS = {
    "", "you", "thank you", "thank you.", "thanks for watching",
    "thanks for watching!", "thank you for watching", "thank you for watching.",
    "thank you very much", "thank you so much", "thanks", "bye", "bye.",
    "please subscribe", "subscribe", "see you next time", "i'll see you next time",
    "see you in the next video", "okay", "ok", "so", "uh", "um", "yeah",
    "the end", "music", "applause", "silence", "...",
}


class SpeechToText:
    def __init__(
        self,
        model: str = "base.en",
        device: str = "auto",
        compute_type: str = "int8",
        language: str = "en",
        beam_size: int = 1,
        cpu_threads: int = 0,
        vad_filter: bool = False,
    ) -> None:
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model = None
        if WhisperModel is None:
            return
        try:
            self._model = WhisperModel(
                model,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
            log.info(
                "Whisper model '%s' loaded (%s/%s, cpu_threads=%s).",
                model, device, compute_type, cpu_threads or "all",
            )
        except Exception as exc:
            log.error("Failed to load Whisper model '%s': %s", model, exc)
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def transcribe(self, audio_int16: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe an int16 mono waveform to text."""
        if self._model is None or audio_int16 is None or len(audio_int16) == 0:
            return ""
        audio = audio_int16.astype(np.float32) / 32768.0
        if sample_rate != 16000:
            # faster-whisper expects 16k; quick linear resample if needed.
            audio = _resample(audio, sample_rate, 16000)
        dur = len(audio) / 16000.0
        t0 = time.time()
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            without_timestamps=True,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
        )
        parts: list[str] = []
        for seg in segments:
            nsp = getattr(seg, "no_speech_prob", 0.0) or 0.0
            alp = getattr(seg, "avg_logprob", 0.0) or 0.0
            # Drop segments the model itself thinks are non-speech / low quality.
            if nsp > 0.6 and alp < -0.4:
                log.debug("Dropping low-confidence segment (no_speech=%.2f, logprob=%.2f): %r",
                          nsp, alp, seg.text)
                continue
            parts.append(seg.text.strip())
        text = " ".join(p for p in parts if p).strip()
        elapsed = time.time() - t0

        # Reject the whole result if it's just a known hallucination phrase.
        normalized = re.sub(r"[^\w\s']", "", text).strip().lower()
        if normalized in _HALLUCINATIONS:
            log.info("STT (%.2fs audio in %.2fs): discarded likely hallucination %r",
                     dur, elapsed, text)
            return ""

        log.info("STT (%.2fs audio in %.2fs, %.1fx): %r", dur, elapsed, dur / max(elapsed, 1e-3), text)
        return text


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio
    n = int(round(len(audio) * dst / src))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)

