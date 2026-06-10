"""Text-to-speech via Piper (local neural TTS, custom voices).

Piper is fast (real-time on CPU) and supports a large library of voices plus
custom-trained ones — perfect for giving Mr. House a distinct voice. We
synthesize one chunk (usually a sentence) at a time so the orchestrator can
stream speech as the LLM produces it.

Targets the Piper >= 1.3 API: ``PiperVoice.synthesize(text, SynthesisConfig)``
yields :class:`AudioChunk` objects exposing ``audio_int16_array``. Returns
float32 mono at the voice's native sample rate; voice FX and the player take it
from there.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    from piper import PiperVoice
    from piper.config import SynthesisConfig

    _HAVE_PIPER = True
except Exception as exc:  # pragma: no cover
    PiperVoice = None
    SynthesisConfig = None
    _HAVE_PIPER = False
    log.warning("piper-tts unavailable (%s); TTS disabled.", exc)


class TextToSpeech:
    def __init__(
        self,
        voice_path: str,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        sentence_silence: float = 0.15,
    ) -> None:
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.sentence_silence = sentence_silence  # kept for API compatibility
        self.sample_rate = 22050
        self._voice = None
        self._syn_config = None

        if not _HAVE_PIPER:
            return
        path = Path(voice_path)
        if not path.exists():
            log.error(
                "Piper voice not found at %s. Download one from "
                "https://github.com/rhasspy/piper/blob/master/VOICES.md",
                path,
            )
            return
        try:
            self._voice = PiperVoice.load(str(path))
            self.sample_rate = self._voice.config.sample_rate
            self._syn_config = SynthesisConfig(
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w_scale=noise_w,
                normalize_audio=True,
            )
            log.info("Piper voice loaded: %s (%d Hz).", path.name, self.sample_rate)
        except Exception as exc:
            log.error("Failed to load Piper voice %s: %s", path, exc)
            self._voice = None

    @property
    def available(self) -> bool:
        return self._voice is not None

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synthesize *text* to a float32 [-1, 1] mono waveform."""
        if self._voice is None or not text.strip():
            return None
        try:
            chunks: list[np.ndarray] = []
            for audio_chunk in self._voice.synthesize(text, self._syn_config):
                arr = np.asarray(audio_chunk.audio_int16_array, dtype=np.int16)
                if arr.size:
                    chunks.append(arr)
            if not chunks:
                return None
            pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
            return pcm
        except Exception as exc:
            log.error("TTS synthesis failed: %s", exc)
            return None

