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


def _make_quad(view_w: int, view_h: int, img_w: int, img_h: int) -> np.ndarray:
    """A screen quad whose position is scaled to preserve the image's aspect
    ratio inside the view (letterbox/pillarbox), so the portrait never stretches
    — important in fullscreen where the screen shape differs from the image."""
    sx, sy = 1.0, 1.0
    if view_w > 0 and view_h > 0 and img_w > 0 and img_h > 0:
        view_aspect = view_w / view_h
        img_aspect = img_w / img_h
        if view_aspect > img_aspect:
            sx = img_aspect / view_aspect      # pillarbox (bars left/right)
        else:
            sy = view_aspect / img_aspect      # letterbox (bars top/bottom)
    return np.array(
        [
            # x,     y,    u,   v   (v flipped so the image is upright)
            -sx, -sy, 0.0, 1.0,
            sx, -sy, 1.0, 1.0,
            -sx,  sy, 0.0, 0.0,
            sx,  sy, 1.0, 0.0,
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
        self._render_w = cfg.width
        self._render_h = cfg.height
        # Runtime fullscreen state (toggleable with F11). Seeded from config.
        self._fullscreen = bool(getattr(cfg, "fullscreen", False))
        self._surface = None          # raw loaded portrait (reused on toggle)
        self._img_w = cfg.width
        self._img_h = cfg.height
        self._tex_w = cfg.width
        self._tex_h = cfg.height

    # ---------------------------------------------------------------- setup
    def _load_surface(self) -> "pygame.Surface":
        img_path = self.cfg.image
        p = Path(img_path)
        if not p.is_absolute():
            p = self.repo_root / p
        if p.exists():
            try:
                # No .convert_alpha() here: this is loaded BEFORE a display mode
                # exists (we need the image size to pick the window size), and
                # convert_alpha() requires an active video mode.
                return pygame.image.load(str(p))
            except Exception as exc:
                log.error("Failed to load image %s: %s", p, exc)
        log.warning("Portrait %s missing; using placeholder.", p)
        return _placeholder_image(self.cfg.width, self.cfg.height)

    def _fit_to_aspect(self, img_w: int, img_h: int) -> tuple[int, int]:
        """Pick a window size matching the image's aspect ratio, fitting the
        configured width/height as a bounding box."""
        if img_w <= 0 or img_h <= 0:
            return self.cfg.width, self.cfg.height
        aspect = img_w / img_h
        max_w, max_h = self.cfg.width, self.cfg.height
        # Start from the configured width, then clamp by height.
        render_w = max_w
        render_h = max(1, round(render_w / aspect))
        if render_h > max_h:
            render_h = max_h
            render_w = max(1, round(render_h * aspect))
        return int(render_w), int(render_h)

    def _setup(self) -> None:
        pygame.init()
        pygame.display.set_caption("Mr. House")

        # Load the portrait once; keep it so we can rebuild the texture when the
        # window is toggled between windowed and fullscreen at runtime.
        self._surface = self._load_surface()
        self._img_w, self._img_h = self._surface.get_size()
        # Texture resolution = the windowed fit size (consistent quality in both
        # modes; the quad handles aspect/letterboxing).
        self._tex_w, self._tex_h = self._fit_to_aspect(self._img_w, self._img_h)

        self._create_gl()

    def _create_gl(self) -> None:
        """(Re)create the window, GL context and resources for the current mode.
        Safe to call again to switch between windowed and fullscreen."""
        # macOS only exposes OpenGL 3.3+ via a *forward-compatible Core* profile,
        # and these attributes MUST be set before creating the window, otherwise
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
        if self._fullscreen:
            flags |= pygame.FULLSCREEN
            size = (0, 0)  # 0,0 = use the desktop resolution
        else:
            size = (self._tex_w, self._tex_h)
        self._screen = pygame.display.set_mode(size, flags)
        view_w, view_h = self._screen.get_size()
        self._render_w, self._render_h = view_w, view_h
        self._ctx = moderngl.create_context(require=330)

        vert = (_SHADER_DIR / "crt.vert").read_text()
        frag = (_SHADER_DIR / "crt.frag").read_text()
        self._prog = self._ctx.program(vertex_shader=vert, fragment_shader=frag)

        quad = _make_quad(view_w, view_h, self._img_w, self._img_h)
        vbo = self._ctx.buffer(quad.tobytes())
        self._vao = self._ctx.vertex_array(
            self._prog, [(vbo, "2f 2f", "in_vert", "in_uv")]
        )

        # Scale the portrait to the texture size (same aspect ratio, no
        # distortion) and upload it.
        surface = pygame.transform.smoothscale(self._surface, (self._tex_w, self._tex_h))
        rgb = pygame.image.tostring(surface, "RGB", False)
        self._tex = self._ctx.texture((self._tex_w, self._tex_h), 3, rgb)
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

    def _release_gl(self) -> None:
        for obj in (self._tex, self._vao, self._prog, self._ctx):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass
        self._tex = self._vao = self._prog = self._ctx = None

    def _toggle_fullscreen(self) -> None:
        """Switch between windowed and fullscreen at runtime (F11)."""
        self._fullscreen = not self._fullscreen
        try:
            self._release_gl()
            self._create_gl()
            log.info("Display: %s", "fullscreen" if self._fullscreen else "windowed")
        except Exception as exc:
            log.error("Failed to toggle fullscreen (%s); reverting.", exc)
            self._fullscreen = not self._fullscreen
            try:
                self._release_gl()
                self._create_gl()
            except Exception:
                self._stop = True

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
        log.info("Display running. F11 toggles fullscreen; ESC or close to quit.")

        while not self._stop:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._stop = True
                    elif event.key == pygame.K_F11 or (
                        event.key == pygame.K_f and (event.mod & pygame.KMOD_META
                                                     or event.mod & pygame.KMOD_CTRL)
                    ):
                        self._toggle_fullscreen()

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

