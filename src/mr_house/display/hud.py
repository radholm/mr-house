"""HUD overlay system.

Renders overlay elements (clock, notifications, etc.) onto a pygame Surface that
is composited on top of the portrait BEFORE the CRT shader pass. This means HUD
text gets the full CRT treatment (scanlines, curvature, glitch) so it looks like
part of the monitor output — zero extra GL complexity.

Designed for extensibility: subclass ``HUDElement`` and register instances in
the ``HUD`` to add new overlay widgets in the future.

Notification bus
----------------
Other threads (e.g. the brain/tool thread) can push temporary messages to the
HUD via the module-level ``push_notification()`` function. The ``NotificationElement``
picks them up (thread-safe) and displays them for a configurable duration.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pygame
import pygame.freetype

from mr_house.config import REPO_ROOT

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Global notification bus (thread-safe)
# ---------------------------------------------------------------------------
@dataclass
class _Notification:
    text: str
    created: float
    duration: float  # seconds; 0 or negative = persistent (never expires)
    key: Optional[str] = None  # optional unique key for persistent notifications
    sound: Optional[str] = None  # path to a sound file to play on appearance
    volume: float = 1.0  # 0.0–1.0 playback volume for the sound
    _sound_played: bool = False  # internal: whether sound has been triggered


_notification_queue: deque[_Notification] = deque(maxlen=20)
_persistent_notifications: dict[str, _Notification] = {}  # key -> notification
_notification_lock = threading.Lock()
_default_duration: float = 8.0  # overridden from config at HUD init
_default_sound: Optional[str] = None  # overridden from config at HUD init
_default_volume: float = 0.7  # overridden from config at HUD init


def _resolve_sound_path(sound: Optional[str]) -> Optional[str]:
    """Resolve a sound path relative to REPO_ROOT."""
    if not sound:
        return None
    p = Path(sound)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return str(p) if p.exists() else None


def push_notification(
    text: str,
    duration: float | None = None,
    key: str | None = None,
    persistent: bool = False,
    sound: str | None = None,
    volume: float | None = None,
) -> None:
    """Push a message to the HUD from any thread.

    Parameters
    ----------
    text : str
        The message to display.
    duration : float | None
        How long (seconds) the notification stays on screen.
        ``None`` uses the configured default. Ignored when *persistent* is True.
    key : str | None
        A unique identifier for persistent notifications. If a notification with
        the same key already exists, its text is updated in place. Required when
        *persistent* is True; optional for timed notifications.
    persistent : bool
        If True the notification never expires. Remove it later with
        ``remove_notification(key)``.
    sound : str | None
        Path to a sound file to play when the notification appears.
        ``None`` uses the configured default sound (if any).
        Pass ``""`` (empty string) to explicitly suppress sound.
    volume : float | None
        Playback volume for the sound (0.0 silent – 1.0 full).
        ``None`` uses the configured default volume.
    """
    # Resolve which sound to use.
    if sound is None:
        snd = _default_sound  # use configured default
    elif sound == "":
        snd = None  # explicitly silenced
    else:
        snd = sound

    vol = max(0.0, min(1.0, volume if volume is not None else _default_volume))

    with _notification_lock:
        if persistent:
            k = key or text
            notif = _Notification(text=text, created=time.time(), duration=0, key=k, sound=snd, volume=vol)
            _persistent_notifications[k] = notif
        else:
            dur = duration if duration is not None else _default_duration
            notif = _Notification(text=text, created=time.time(), duration=dur, key=key, sound=snd, volume=vol)
            _notification_queue.append(notif)


def remove_notification(key: str) -> None:
    """Remove a persistent notification by its key. No-op if not found."""
    with _notification_lock:
        _persistent_notifications.pop(key, None)


def update_notification(key: str, text: str) -> None:
    """Update the text of an existing persistent notification. No-op if not found."""
    with _notification_lock:
        if key in _persistent_notifications:
            _persistent_notifications[key] = _Notification(
                text=text,
                created=_persistent_notifications[key].created,
                duration=0,
                key=key,
                _sound_played=True,  # don't re-play sound on update
            )


# ---------------------------------------------------------------------------
#  Position helper
# ---------------------------------------------------------------------------
def _position_rect(
    position: str, rect_w: int, rect_h: int, surface_w: int, surface_h: int, padding: int
) -> tuple[int, int]:
    """Compute (x, y) for a rendered rect given a position string."""
    # Vertical
    if "top" in position:
        y = padding
    elif "bottom" in position:
        y = surface_h - rect_h - padding
    else:
        y = (surface_h - rect_h) // 2

    # Horizontal
    if "right" in position:
        x = surface_w - rect_w - padding
    elif "left" in position:
        x = padding
    else:  # center (including top-center, bottom-center)
        x = (surface_w - rect_w) // 2

    return x, y


# ---------------------------------------------------------------------------
#  Base class
# ---------------------------------------------------------------------------
class HUDElement(ABC):
    """A single overlay widget rendered onto the HUD surface each frame."""

    @abstractmethod
    def render(self, surface: pygame.Surface, now: float) -> None:
        """Draw this element onto *surface* (SRCALPHA, same size as portrait)."""


# ---------------------------------------------------------------------------
#  Clock element
# ---------------------------------------------------------------------------
class ClockElement(HUDElement):
    """Displays the current date/time in a configurable format and colour."""

    def __init__(
        self,
        font: "pygame.freetype.Font",
        color: tuple[int, int, int],
        fmt: str = "%H:%M:%S",
        position: str = "top-right",
        padding: int = 20,
    ) -> None:
        self._font = font
        self._color = color
        self._fmt = fmt
        self._position = position
        self._padding = padding

    def render(self, surface: pygame.Surface, now: float) -> None:
        text = time.strftime(self._fmt, time.localtime(now))
        rendered_surf, rect = self._font.render(text, self._color)
        sw, sh = surface.get_size()
        x, y = _position_rect(self._position, rect.width, rect.height, sw, sh, self._padding)
        surface.blit(rendered_surf, (x, y))


# ---------------------------------------------------------------------------
#  Watermark element
# ---------------------------------------------------------------------------
class WatermarkElement(HUDElement):
    """Displays a static text watermark at a configurable position."""

    def __init__(
        self,
        font: "pygame.freetype.Font",
        color: tuple[int, int, int],
        text: str = "MR. HOUSE",
        position: str = "top-left",
        padding: int = 20,
    ) -> None:
        self._font = font
        self._color = color
        self._text = text
        self._position = position
        self._padding = padding

    def render(self, surface: pygame.Surface, now: float) -> None:
        rendered_surf, rect = self._font.render(self._text, self._color)
        sw, sh = surface.get_size()
        x, y = _position_rect(self._position, rect.width, rect.height, sw, sh, self._padding)
        surface.blit(rendered_surf, (x, y))


# ---------------------------------------------------------------------------
#  Notification element (toast messages)
# ---------------------------------------------------------------------------
class NotificationElement(HUDElement):
    """Displays temporary notification messages pushed via push_notification()."""

    def __init__(
        self,
        font: "pygame.freetype.Font",
        color: tuple[int, int, int],
        position: str = "bottom-center",
        padding: int = 20,
        line_spacing: int = 4,
        max_visible: int = 4,
    ) -> None:
        self._font = font
        self._color = color
        self._position = position
        self._padding = padding
        self._line_spacing = line_spacing
        self._max_visible = max_visible

    def render(self, surface: pygame.Surface, now: float) -> None:
        # Collect active notifications: persistent + non-expired timed.
        with _notification_lock:
            active: list[_Notification] = list(_persistent_notifications.values())
            active += [
                n for n in _notification_queue
                if n.duration <= 0 or now - n.created < n.duration
            ]
            # Play sounds for newly appeared notifications.
            for n in active:
                if not n._sound_played and n.sound:
                    n._sound_played = True
                    self._play_sound(n.sound, n.volume)
        if not active:
            return

        # Show only the most recent N.
        visible = active[-self._max_visible:]
        sw, sh = surface.get_size()

        # Render all lines, then position them as a block.
        rendered: list[tuple[pygame.Surface, pygame.Rect]] = []
        total_h = 0
        max_w = 0
        for notif in visible:
            # Split on explicit newlines first, then word-wrap each line.
            for raw_line in notif.text.split("\n"):
                wrapped = self._wrap(raw_line, sw - self._padding * 2)
                for line in wrapped:
                    surf, rect = self._font.render(line, self._color)
                    rendered.append((surf, rect))
                    total_h += rect.height + self._line_spacing
                    max_w = max(max_w, rect.width)

        if not rendered:
            return

        total_h -= self._line_spacing  # no trailing spacing

        # Position the block.
        bx, by = _position_rect(self._position, max_w, total_h, sw, sh, self._padding)

        # Blit each line.
        cy = by
        for surf, rect in rendered:
            # Align each line within the block according to position.
            if "left" in self._position:
                lx = bx
            elif "right" in self._position:
                lx = bx + (max_w - rect.width)
            else:
                lx = bx + (max_w - rect.width) // 2
            surface.blit(surf, (lx, cy))
            cy += rect.height + self._line_spacing

    def _wrap(self, text: str, max_px: int) -> list[str]:
        """Simple word-wrap to fit within max_px width."""
        words = text.split()
        if not words:
            return []
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            test = current + " " + word
            rect = self._font.get_rect(test)
            if rect.width <= max_px:
                current = test
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    @staticmethod
    def _play_sound(path: str, volume: float = 1.0) -> None:
        """Play a notification sound (non-blocking). Resolves relative paths."""
        try:
            p = Path(path)
            if not p.is_absolute():
                p = REPO_ROOT / p
            if not p.exists():
                return
            # pygame.mixer may not be initialised (OpenGL display mode), so
            # initialise it on first use if needed.
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            sound = pygame.mixer.Sound(str(p))
            sound.set_volume(max(0.0, min(1.0, volume)))
            sound.play()
        except Exception as exc:
            log.debug("Could not play notification sound %s: %s", path, exc)


# ---------------------------------------------------------------------------
#  HUD manager
# ---------------------------------------------------------------------------
class HUD:
    """Manages all HUD elements and renders them onto a surface."""

    def __init__(self, hud_cfg, surface_size: tuple[int, int]) -> None:
        global _default_duration, _default_sound, _default_volume

        self._elements: list[HUDElement] = []
        self._surface = pygame.Surface(surface_size, pygame.SRCALPHA)

        if not hud_cfg.enabled:
            return

        # Set global notification defaults from config.
        _default_duration = float(getattr(hud_cfg, "notification_duration", 8.0))
        raw_sound = getattr(hud_cfg, "notification_sound", "")
        _default_sound = _resolve_sound_path(raw_sound) if raw_sound else None
        _default_volume = float(getattr(hud_cfg, "notification_volume", 0.7))

        # Load fonts (per-element sizes, falling back to global font_size).
        base_size = hud_cfg.font_size
        color = tuple(hud_cfg.color[:3])

        clock_size = getattr(hud_cfg, "clock_font_size", 0) or base_size
        watermark_size = getattr(hud_cfg, "watermark_font_size", 0) or base_size

        font_base = self._load_font(hud_cfg.font, base_size)
        font_clock = self._load_font(hud_cfg.font, clock_size) if clock_size != base_size else font_base
        font_watermark = self._load_font(hud_cfg.font, watermark_size) if watermark_size != base_size else font_base

        if hud_cfg.show_clock:
            self._elements.append(
                ClockElement(
                    font=font_clock,
                    color=color,
                    fmt=hud_cfg.clock_format,
                    position=hud_cfg.clock_position,
                    padding=hud_cfg.padding,
                )
            )

        if getattr(hud_cfg, "show_watermark", False):
            self._elements.append(
                WatermarkElement(
                    font=font_watermark,
                    color=color,
                    text=getattr(hud_cfg, "watermark_text", "MR. HOUSE"),
                    position=getattr(hud_cfg, "watermark_position", "top-left"),
                    padding=hud_cfg.padding,
                )
            )

        # Notification toasts (always active when HUD is enabled).
        notification_pos = getattr(hud_cfg, "notification_position", "bottom-center")
        self._elements.append(
            NotificationElement(
                font=font_base,
                color=color,
                position=notification_pos,
                padding=hud_cfg.padding,
            )
        )

    def _load_font(
        self, font_path: str, size: int
    ) -> "pygame.freetype.Font":
        """Load a font file, resolving relative paths against the repo root."""
        p = Path(font_path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists():
            try:
                font = pygame.freetype.Font(str(p), size)
                return font
            except Exception as exc:
                log.warning("Failed to load font %s: %s; using fallback.", p, exc)
        # Fallback: default monospace
        return pygame.freetype.SysFont("monospace", size)

    def resize(self, new_size: tuple[int, int]) -> None:
        """Recreate the internal surface when the display resizes."""
        self._surface = pygame.Surface(new_size, pygame.SRCALPHA)

    def render(self, target: pygame.Surface) -> None:
        """Render all HUD elements and blit onto *target*."""
        if not self._elements:
            return
        self._surface.fill((0, 0, 0, 0))
        now = time.time()
        for element in self._elements:
            element.render(self._surface, now)
        target.blit(self._surface, (0, 0))

