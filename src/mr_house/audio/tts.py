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

# References to the tools/sources that should never be spoken. We remove whole
# attribution clauses ("according to the Fallout wiki, ...") and bare tool names.
_SOURCE_CLAUSE_RE = re.compile(
    r"\b(?:according to|based on|as (?:found|stated|noted|per)(?: on| in)?|"
    r"sourced from|using|via|per|from|in)\b[^.,;:!?]{0,50}?"
    r"(?:fallout\s*(?:fandom\s*)?wiki|fallout[_ ]?lore|wikipedi[ao]|"
    r"web[_ ]?search(?:\s+results?)?|search\s+results?|my\s+(?:tools?|sources?|records?))"
    r"[^.,;:!?]{0,25}?[,:]?\s*",
    re.IGNORECASE,
)
_BARE_SOURCE_RE = re.compile(
    r"\bthe\s+(?:information\s+from\s+the\s+)?"
    r"(?:fallout\s*(?:fandom\s*)?wiki|web[_ ]?search(?:\s+results?)?|wikipedi[ao]|"
    r"search\s+results?)\b"
    # optionally swallow a following reporting verb so we don't leave "says ..."
    r"(?:\s+(?:says?|shows?|states?|indicates?|reports?|notes?|reveals?|"
    r"confirms?|mentions?|reads?|tells?\s+(?:me|us)))?[,:]?\s*",
    re.IGNORECASE,
)
_TOOL_NAME_RE = re.compile(
    r"\b(?:the\s+)?(?:web[_ ]?search|fallout[_ ]?lore|get[_ ]?self[_ ]?info|"
    r"get[_ ]?weather|get[_ ]?time)\b(?:\s+(?:tool|function|results?))?"
    r"(?:\s+(?:says?|shows?|states?|indicates?|reports?|notes?|reveals?|"
    r"confirms?|mentions?))?",
    re.IGNORECASE,
)


def _strip_tool_references(t: str) -> str:
    """Remove any mention of the tools/sources so House speaks as himself."""
    t = _SOURCE_CLAUSE_RE.sub(" ", t)
    t = _BARE_SOURCE_RE.sub(" ", t)
    t = _TOOL_NAME_RE.sub(" ", t)
    return t
# Title abbreviations expanded to full words so the voice (a) pronounces them
# correctly and (b) doesn't read the trailing '.' as a sentence break — which is
# why "Mr. House" came out as two separate sentences.
_ABBREVIATIONS = [
    (re.compile(r"\bMr\.\s+", re.IGNORECASE), "Mister "),
    (re.compile(r"\bMrs\.\s+", re.IGNORECASE), "Missus "),
    (re.compile(r"\bMs\.\s+", re.IGNORECASE), "Miss "),
    (re.compile(r"\bDr\.\s+", re.IGNORECASE), "Doctor "),
    (re.compile(r"\bProf\.\s+", re.IGNORECASE), "Professor "),
    (re.compile(r"\bSt\.\s+", re.IGNORECASE), "Saint "),
    (re.compile(r"\bMt\.\s+", re.IGNORECASE), "Mount "),
    (re.compile(r"\bvs\.\s+", re.IGNORECASE), "versus "),
    (re.compile(r"\betc\.", re.IGNORECASE), "etcetera"),
]


def sanitize_for_speech(text: str) -> str:
    """Strip anything the TTS can't pronounce: markdown symbols, asterisk/bracket
    stage directions, and non-verbal interjections like 'ugh', 'mm', 'hmph'.
    Also expands title abbreviations ('Mr.' -> 'Mister') so the voice doesn't
    treat their trailing period as a sentence break."""
    if not text:
        return ""
    t = _STAGE_RE.sub(" ", text)      # drop *sighs* / (groans) / [sigh]
    t = _SYMBOL_RE.sub(" ", t)        # drop stray markdown symbols
    t = _strip_tool_references(t)     # remove "the web_search"/"Fallout wiki" etc.
    for pat, repl in _ABBREVIATIONS:  # Mr. -> Mister, Dr. -> Doctor, etc.
        t = pat.sub(repl, t)
    t = _INTERJ_RE.sub("", t)         # drop "ugh"/"mm"/etc. + trailing punctuation
    t = re.sub(r"\s+", " ", t)        # collapse whitespace
    t = re.sub(r"\s+([,.;:!?…])", r"\1", t)      # no space before punctuation
    t = re.sub(r"^[\s,.;:!?…\-–—]+", "", t)      # trim orphan leading punctuation
    # Collapse duplicated punctuation, but NOT '.' — an ellipsis "..." is kept so
    # the voice can pause on it.
    t = re.sub(r"([,;:!?])\1{1,}", r"\1", t)
    t = t.strip()
    # If stripping an attribution left the sentence starting lowercase, fix it.
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t


# Pause-creating punctuation. Em/en dashes and a spaced hyphen act as a dash;
# a comma is kept in the spoken text (so the voice keeps its intonation) with a
# little extra silence after it.
_DASH_DELIMS = ("—", "–", " - ")


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
        ellipsis_pause: float = 0.35,
        dash_pause: float = 0.2,
        comma_pause: float = 0.1,
    ) -> None:
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        # How much to randomly vary prosody per sentence so a multi-sentence
        # reply doesn't come out flat/monotone. 0 disables the jitter.
        self.expressiveness = max(0.0, expressiveness)
        self.sentence_silence = sentence_silence  # kept for API compatibility
        # Seconds of silence inserted at "...", dashes, and commas (0 disables).
        self.ellipsis_pause = max(0.0, ellipsis_pause)
        self.dash_pause = max(0.0, dash_pause)
        self.comma_pause = max(0.0, comma_pause)
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
        # interjections like "ugh"/"mm") and expand "Mr." etc. before synthesis.
        text = sanitize_for_speech(text)
        if not text.strip():
            return None
        # Break the sentence at pause punctuation ("...", dashes, commas) so we
        # can drop a real silence there. It is still spoken as ONE continuous
        # utterance, and the model's context/memory keeps the full text intact.
        segments = self._segment_for_pauses(text)
        if not segments:
            return None
        try:
            pieces: list[np.ndarray] = []
            for seg_text, pause_after in segments:
                seg_pcm = self._synth_one(seg_text)
                if seg_pcm is None or seg_pcm.size == 0:
                    continue
                pieces.append(seg_pcm)
                if pause_after > 0:
                    pieces.append(
                        np.zeros(int(self.sample_rate * pause_after), dtype=np.float32)
                    )
            if not pieces:
                return None
            return np.concatenate(pieces)
        except Exception as exc:
            log.error("TTS synthesis failed: %s", exc)
            return None

    def _segment_for_pauses(self, text: str) -> list[tuple[str, float]]:
        """Split *text* into (segment, pause_after_seconds) pairs at pause
        punctuation. Commas are kept in the spoken text; ellipses and dashes are
        replaced by the silence. Only punctuation with a positive pause is split
        on, so disabling a pause falls back to the voice's own handling."""
        delims: list[str] = []
        if self.ellipsis_pause > 0:
            delims.append(r"\.\.\.+|…")
        if self.dash_pause > 0:
            delims.append(r"—|–| - ")
        if self.comma_pause > 0:
            delims.append(r",")
        if not delims:
            return [(text.strip(), 0.0)] if text.strip() else []

        parts = re.split("(" + "|".join(delims) + ")", text)
        segments: list[list] = []
        buf = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                buf += part
                continue
            delim = part
            if delim == ",":
                buf += ","                 # keep the comma so the voice intonates
                pause = self.comma_pause
            elif delim in _DASH_DELIMS:
                pause = self.dash_pause     # dash dropped, replaced by silence
            else:
                pause = self.ellipsis_pause  # ellipsis dropped, replaced by silence
            seg = buf.strip()
            if seg:
                segments.append([seg, pause])
            elif segments:
                segments[-1][1] = max(segments[-1][1], pause)
            buf = ""
        if buf.strip():
            segments.append([buf.strip(), 0.0])
        if segments:
            segments[-1][1] = 0.0  # no trailing silence at the very end
        return [(s, p) for s, p in segments]

    def _synth_one(self, text: str) -> Optional[np.ndarray]:
        """Synthesize a single segment (no pause handling) to float32 PCM."""
        chunks: list[np.ndarray] = []
        syn_config = self._config_for(text)
        for audio_chunk in self._voice.synthesize(text, syn_config):
            arr = np.asarray(audio_chunk.audio_int16_array, dtype=np.int16)
            if arr.size:
                chunks.append(arr)
        if not chunks:
            return None
        return np.concatenate(chunks).astype(np.float32) / 32768.0

