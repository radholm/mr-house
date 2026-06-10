"""The CRT display window.

A single static portrait is uploaded once as a texture and re-rendered every
frame through the CRT/glitch fragment shader. OpenGL must run on the **main
thread** (especially on macOS), so the orchestrator runs in a background thread
and the display owns the main loop. A ``glitch_provider`` callback lets the
display ask "is Mr. House speaking?" so it can intensify the glitch in sync with
his voice.

If ``moderngl``/``pygame`` aren't available the window degrades to a no-op so the
voice pipeline still runs headless.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import pygame
    import moderngl

    _HAVE_GL = True
except Exception as exc:  # pragma: no cover
    _HAVE_GL = False
    log.warning("pygame/moderngl unavailable (%s); display disabled.", exc)

_SHADER_DIR = Path(__file__).parent / "shaders"

# Fullscreen quad: position (x, y) + uv (u, v). Note v is flipped so the image
# is upright (textures are bottom-left origin).
_QUAD = np.array(
    [
        # x,    y,    u,   v
        -1.0, -1.0, 0.0, 1.0,
        1.0, -1.0, 1.0, 1.0,
        -1.0,  1.0, 0.0, 0.0,
        1.0,  1.0, 1.0, 0.0,
    ],
    dtype="f4",
)


def _placeholder_image(w: int, h: int) -> "pygame.Surface":
    """Generate a moody placeholder portrait if no image file is provided."""
    surf = pygame.Surface((w, h))
    for y in range(h):
        shade = int(20 + 30 * (y / h))
        pygame.draw.line(surf, (shade, shade // 2, shade // 3), (0, y), (w, y))
    font = pygame.font.SysFont("Courier", h // 12, bold=True)
    text = font.render("MR. HOUSE", True, (210, 170, 90))
    surf.blit(text, text.get_rect(center=(w // 2, h // 2)))
    sub = pygame.font.SysFont("Courier", h // 32)
    s2 = sub.render("// drop assets/house.png to replace //", True, (120, 100, 60))
    surf.blit(s2, s2.get_rect(center=(w // 2, h // 2 + h // 8)))
    return surf


class CRTDisplay:
    def __init__(self, cfg, repo_root: Path) -> None:
        self.cfg = cfg
        self.repo_root = repo_root
        self.enabled = bool(cfg.enabled) and _HAVE_GL
        self._ctx = None
        self._prog = None
        self._vao = None
        self._tex = None
        self._screen = None
        self._glitch_provider: Optional[Callable[[], bool]] = None
        self._stop = False
        self._smoothed_glitch = 0.0

    # ---------------------------------------------------------------- setup
    def _load_surface(self) -> "pygame.Surface":
        img_path = self.cfg.image
        p = Path(img_path)
        if not p.is_absolute():
            p = self.repo_root / p
        if p.exists():
            try:
                return pygame.image.load(str(p)).convert_alpha()
            except Exception as exc:
                log.error("Failed to load image %s: %s", p, exc)
        log.warning("Portrait %s missing; using placeholder.", p)
        return _placeholder_image(self.cfg.width, self.cfg.height)

    def _setup(self) -> None:
        pygame.init()
        pygame.display.set_caption("Mr. House")

        # macOS only exposes OpenGL 3.3+ via a *forward-compatible Core* profile,
        # and these attributes MUST be set before create the window, otherwise
        # we get a legacy 2.1 context and moderngl reports "got version 0".
        gl = pygame.display
        gl.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        gl.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        gl.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        # The forward-compatible flag is REQUIRED for a Core profile on macOS, but
        # can be needlessly strict on some Windows drivers — so only set it there.
        if sys.platform == "darwin":
            fwd = getattr(pygame, "GL_CONTEXT_FORWARD_COMPATIBLE_FLAG", None)
            if fwd is not None:
                gl.gl_set_attribute(fwd, 1)
        gl.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        gl.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)

        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if self.cfg.fullscreen:
            flags |= pygame.FULLSCREEN
        self._screen = pygame.display.set_mode(
            (self.cfg.width, self.cfg.height), flags
        )
        self._ctx = moderngl.create_context(require=330)

        vert = (_SHADER_DIR / "crt.vert").read_text()
        frag = (_SHADER_DIR / "crt.frag").read_text()
        self._prog = self._ctx.program(vertex_shader=vert, fragment_shader=frag)

        vbo = self._ctx.buffer(_QUAD.tobytes())
        self._vao = self._ctx.vertex_array(
            self._prog, [(vbo, "2f 2f", "in_vert", "in_uv")]
        )

        surface = self._load_surface()
        surface = pygame.transform.smoothscale(surface, (self.cfg.width, self.cfg.height))
        rgb = pygame.image.tostring(surface, "RGB", False)
        self._tex = self._ctx.texture((self.cfg.width, self.cfg.height), 3, rgb)
        self._tex.build_mipmaps()
        self._tex.repeat_x = False
        self._tex.repeat_y = False
        self._tex.use(0)

        # Static uniforms.
        sh = self.cfg.shader
        self._set("u_tex", 0)
        self._set("u_scanline_intensity", sh.scanline_intensity)
        self._set("u_scanline_count", sh.scanline_count)
        self._set("u_chromatic", sh.chromatic_aberration)
        self._set("u_vignette", sh.vignette)
        self._set("u_flicker", sh.flicker)
        self._set("u_curvature", sh.curvature)

    def _set(self, name: str, value) -> None:
        if self._prog is not None and name in self._prog:
            self._prog[name].value = value

    # ---------------------------------------------------------------- loop
    def set_glitch_provider(self, provider: Callable[[], bool]) -> None:
        self._glitch_provider = provider

    def run(self, on_close: Optional[Callable[[], None]] = None) -> None:
        """Blocking render loop. Call from the MAIN thread."""
        if not self.enabled:
            log.info("Display disabled; running headless.")
            self._headless_wait()
            if on_close:
                on_close()
            return
        try:
            self._setup()
        except Exception as exc:
            log.error("Display setup failed (%s); continuing headless (voice still works).", exc)
            self.enabled = False
            self._headless_wait()
            if on_close:
                on_close()
            return

        clock = pygame.time.Clock()
        sh = self.cfg.shader
        start = time.time()
        log.info("Display running. Press ESC or close window to quit.")

        while not self._stop:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._stop = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._stop = True

            speaking = bool(self._glitch_provider and self._glitch_provider())
            target = sh.glitch_amount * (sh.glitch_speaking_boost if speaking else 1.0)
            # Smooth toward target so the glitch eases in/out with speech.
            self._smoothed_glitch += (target - self._smoothed_glitch) * 0.15

            self._set("u_time", time.time() - start)
            self._set("u_glitch", self._smoothed_glitch)

            self._ctx.clear(0.0, 0.0, 0.0)
            self._tex.use(0)
            self._vao.render(moderngl.TRIANGLE_STRIP)
            pygame.display.flip()
            clock.tick(self.cfg.fps)

        pygame.quit()
        if on_close:
            on_close()

    def stop(self) -> None:
        self._stop = True

    def _headless_wait(self) -> None:
        """Keep the main thread alive (so the voice loop runs) until Ctrl-C."""
        try:
            while not self._stop:
                time.sleep(0.2)
        except KeyboardInterrupt:
            self._stop = True

