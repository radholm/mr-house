"""Configuration loading.

Reads ``config.yaml`` into nested dataclasses with sane defaults so the rest of
the codebase can use typed attribute access (``cfg.audio.sample_rate``) instead
of dictionary spelunking. Missing keys fall back to the defaults defined here,
which means a partial / hand-edited config still works.

Environment variables of the form ``${VAR}`` inside string values are expanded
(used by MCP server definitions for API keys).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml
from dotenv import load_dotenv

# Repo root = three levels up from this file (src/mr_house/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` references in strings."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
#  Dataclasses                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class AudioConfig:
    sample_rate: int = 16000
    block_size: int = 1280
    input_device: Any = None
    output_device: Any = None
    output_sample_rate: int = 22050


@dataclass
class WakeWordConfig:
    enabled: bool = True
    models: list[str] = field(default_factory=lambda: ["hey_jarvis"])
    threshold: float = 0.5
    vad_threshold: float = 0.3


@dataclass
class VADConfig:
    aggressiveness: int = 2
    silence_ms: int = 800
    max_utterance_ms: int = 15000
    start_timeout_ms: int = 6000
    start_speech_ms: int = 150     # sustained voice needed before "speech started"
    min_voiced_ms: int = 350       # min voiced audio to keep (else = noise)


@dataclass
class STTConfig:
    enabled: bool = True
    model: str = "small.en"
    device: str = "auto"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 5
    cpu_threads: int = 0       # 0 = use all available cores
    vad_filter: bool = False   # we already trim with our own VAD


@dataclass
class TTSConfig:
    enabled: bool = True
    engine: str = "piper"
    voice: str = "src/mr_house/assets/voices/en_US-ryan-high.onnx"
    length_scale: float = 1.06
    noise_scale: float = 0.85
    noise_w: float = 0.95
    expressiveness: float = 0.12   # per-sentence prosody jitter (0 = off)
    ellipsis_pause: float = 0.35   # seconds of pause at "..." (0 = off)
    sentence_silence: float = 0.15


@dataclass
class ReverbConfig:
    room_size: float = 0.18
    damping: float = 0.6
    wet_level: float = 0.12
    dry_level: float = 0.9


@dataclass
class VoiceFXConfig:
    enabled: bool = True
    highpass_hz: float = 90
    lowpass_hz: float = 7000
    eq: list[list[float]] = field(default_factory=list)
    distortion_db: float = 4.0
    reverb: ReverbConfig = field(default_factory=ReverbConfig)
    bitcrush_depth: int = 0


@dataclass
class BrainConfig:
    provider: str = "ollama"
    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"
    temperature: float = 0.95
    top_p: float = 0.95
    repeat_penalty: float = 1.15
    max_history_turns: int = 12
    num_ctx: int = 4096
    filler_after_ms: int = 450
    max_tool_iterations: int = 4
    persist_memory: bool = True
    memory_file: str = "data/memory.json"


@dataclass
class ConversationConfig:
    # After Mr. House answers, keep listening this long for a follow-up so you
    # don't have to repeat the wake word. 0 disables follow-up mode.
    followup_enabled: bool = True
    followup_window_ms: int = 30000
    max_followups: int = 0        # 0 = unlimited while you keep talking


@dataclass
class MCPConfig:
    enabled: bool = True
    servers: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShaderConfig:
    scanline_intensity: float = 0.85
    scanline_count: float = 400.0
    chromatic_aberration: float = 0.0025
    vignette: float = 0.35
    flicker: float = 0.04
    glitch_amount: float = 0.1
    glitch_speaking_boost: float = 2.5
    curvature: float = 0.20


@dataclass
class DisplayConfig:
    enabled: bool = True
    image: str = "src/mr_house/assets/house.png"
    width: int = 900
    height: int = 1100
    fullscreen: bool = False
    fps: int = 60
    shader: ShaderConfig = field(default_factory=ShaderConfig)


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    voice_fx: VoiceFXConfig = field(default_factory=VoiceFXConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    log_level: str = "INFO"

    def resolve_path(self, p: str | os.PathLike) -> Path:
        """Resolve a possibly-relative path against the repo root."""
        path = Path(p)
        return path if path.is_absolute() else (REPO_ROOT / path)


# --------------------------------------------------------------------------- #
#  Construction helpers                                                        #
# --------------------------------------------------------------------------- #
def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Build a (possibly nested) dataclass from a plain dict, ignoring extras."""
    if not is_dataclass(cls):
        return data
    # ``from __future__ import annotations`` stores field types as strings, so
    # resolve them to real classes here (needed to detect nested dataclasses).
    try:
        type_hints = get_type_hints(cls)
    except Exception:
        type_hints = {f.name: f.type for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = type_hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(ftype, value)  # type: ignore[arg-type]
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load configuration from ``config.yaml`` (or *path*)."""
    load_dotenv(REPO_ROOT / ".env")

    cfg_path = Path(path) if path else (REPO_ROOT / "config.yaml")
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    raw = _expand_env(raw)
    return _from_dict(Config, raw)



