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
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Non-verbal interjections / onomatopoeia the neural voice mangles or can't say.
# Stripped (with any trailing punctuation) before synthesis.
_INTERJECTIONS = {
    "ugh", "ugch", "mm", "mmm", "mmmm", "mhm", "hm", "hmm", "hmmm", "hmph",
    "tch", "tsk", "grr", "grrr", "argh", "ngh", "uh", "uhh", "um", "umm",
    "uhm", "er", "err", "erm", "eh", "ahem", "huh", "pff", "pfft", "psh",
    "meh", "bah", "hrm", "hngh", "ack", "blegh", "ew",
}
_INTERJ_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_INTERJECTIONS, key=len, reverse=True)) + r")\b[\s.,!?;:…-]*",
    re.IGNORECASE,
)
# Stage directions wrapped in asterisks/brackets, e.g. *sighs*, (groans), [sigh].
_STAGE_RE = re.compile(r"\*[^*\n]*\*|\([^)\n]*\)|\[[^\]\n]*\]")
# Leftover markdown / symbol characters the voice would try to vocalise.
_SYMBOL_RE = re.compile(r"[*_`~#>|^]+")


def sanitize_for_speech(text: str) -> str:
    """Strip anything the TTS can't pronounce: markdown symbols, asterisk/bracket
    stage directions, and non-verbal interjections like 'ugh', 'mm', 'hmph'."""
    if not text:
        return ""
    t = _STAGE_RE.sub(" ", text)      # drop *sighs* / (groans) / [sigh]
    t = _SYMBOL_RE.sub(" ", t)        # drop stray markdown symbols
    t = _INTERJ_RE.sub("", t)         # drop "ugh"/"mm"/etc. + trailing punctuation
    t = re.sub(r"\s+", " ", t)        # collapse whitespace
    t = re.sub(r"\s+([,.;:!?…])", r"\1", t)      # no space before punctuation
    t = re.sub(r"^[\s,.;:!?…\-–—]+", "", t)      # trim orphan leading punctuation
    t = re.sub(r"([,.;:!?])\1{1,}", r"\1", t)    # collapse duplicated punctuation
    return t.strip()


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
        noise_scale: float = 0.85,
        noise_w: float = 0.95,
        expressiveness: float = 0.12,
        sentence_silence: float = 0.15,
    ) -> None:
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        # How much to randomly vary prosody per sentence so a multi-sentence
        # reply doesn't come out flat/monotone. 0 disables the jitter.
        self.expressiveness = max(0.0, expressiveness)
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

    def _config_for(self, text: str):
        """A SynthesisConfig for *text*, jittered for more expressive delivery.

        Each call nudges pitch/prosody (``noise_scale``), cadence
        (``noise_w_scale``) and pace (``length_scale``) by a small random amount
        so consecutive sentences in a reply don't sound identically flat. The
        jitter is gentle and clamped to keep the voice recognisable.
        """
        if self.expressiveness <= 0.0:
            return self._syn_config

        j = self.expressiveness
        # Pitch/prosody variability — the biggest lever against monotone.
        noise_scale = self.noise_scale * (1.0 + random.uniform(-j, j))
        # Cadence/rhythm variability.
        noise_w = self.noise_w * (1.0 + random.uniform(-j, j))
        # Pace — only ever vary toward *slower*, never faster, so he never
        # rattles a sentence off. (Larger length_scale = slower speech.)
        length_scale = self.length_scale * (1.0 + random.uniform(0.0, j) * 0.5)

        noise_scale = float(np.clip(noise_scale, 0.3, 1.2))
        noise_w = float(np.clip(noise_w, 0.4, 1.4))
        # Floor at the configured pace so jitter can't make him faster.
        length_scale = float(np.clip(length_scale, self.length_scale, 1.4))

        return SynthesisConfig(
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w,
            normalize_audio=True,
        )

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synthesize *text* to a float32 [-1, 1] mono waveform."""
        if self._voice is None:
            return None
        # Strip anything the voice can't pronounce (markdown, *stage directions*,
        # interjections like "ugh"/"mm") before synthesis.
        text = sanitize_for_speech(text)
        if not text.strip():
            return None
        try:
            chunks: list[np.ndarray] = []
            syn_config = self._config_for(text)
            for audio_chunk in self._voice.synthesize(text, syn_config):
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

