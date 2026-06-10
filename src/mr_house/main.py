"""Entry point: wires the orchestrator to the display and handles CLI flags.

The display must own the **main thread**, so :class:`MrHouse` runs its loop in a
background thread and we block here on the render loop. ``--check`` prints which
subsystems are available; ``--no-display`` runs headless; ``--text`` skips audio
entirely for quick brain/personality testing.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import tempfile

from .config import load_config, REPO_ROOT
from .app import MrHouse


# Held open for the lifetime of the process to enforce a single instance.
_LOCK_HANDLE = None


def _acquire_single_instance_lock() -> bool:
    """Return True if we got the lock; False if another instance is running.

    Prevents multiple Mr. Houses from running at once — otherwise several
    instances all hear the wake word and answer over each other. Works on both
    POSIX (fcntl) and Windows (msvcrt).
    """
    global _LOCK_HANDLE

    lock_path = os.path.join(tempfile.gettempdir(), "mr_house.lock")
    try:
        fh = open(lock_path, "w")
        if os.name == "nt":  # Windows
            import msvcrt

            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:  # POSIX
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    try:
        fh.write(str(os.getpid()))
        fh.flush()
    except OSError:
        pass
    _LOCK_HANDLE = fh  # keep reference so the lock is held
    return True


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy libs.
    for noisy in ("numba", "faster_whisper", "urllib3", "asyncio", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # MCP servers sometimes print non-protocol text (e.g. package-install logs)
    # to stdout on a cold start, which the client logs as JSONRPC parse errors.
    # They're harmless noise, so suppress them.
    logging.getLogger("mcp.client.stdio").setLevel(logging.CRITICAL)


def _check(cfg) -> int:
    """Report subsystem availability without starting anything heavy."""
    from .audio.capture import MicrophoneStream
    from .audio.wake_word import WakeWordDetector
    from .audio.stt import SpeechToText
    from .audio.tts import TextToSpeech
    from .audio.effects import VoiceEffects
    from .brain.llm import Brain
    from .brain.memory import ConversationMemory
    from .brain.personality import build_system_prompt

    print("Mr. House — subsystem check\n" + "-" * 32)

    mic = MicrophoneStream(cfg.audio.sample_rate, cfg.audio.block_size, cfg.audio.input_device)
    print(f"  microphone : {'ok' if mic.available else 'MISSING (sounddevice/PortAudio)'}")

    wake = WakeWordDetector(cfg.wake_word.models, cfg.wake_word.threshold, repo_root=REPO_ROOT)
    print(f"  wake word  : {'ok' if wake.available else 'MISSING (openwakeword/models)'}")

    stt = SpeechToText(cfg.stt.model, cfg.stt.device, cfg.stt.compute_type)
    print(f"  stt        : {'ok' if stt.available else 'MISSING (faster-whisper)'}")

    tts = TextToSpeech(str(cfg.resolve_path(cfg.tts.voice)))
    print(f"  tts        : {'ok' if tts.available else 'MISSING (piper voice .onnx)'}")

    fx = VoiceEffects(cfg.voice_fx, sample_rate=22050)
    print(f"  voice fx   : {'ok' if fx.enabled else 'MISSING (pedalboard)'}")

    brain = Brain(cfg.brain, ConversationMemory(build_system_prompt()))
    print(f"  brain/llm  : {'ok' if brain.available else 'MISSING (ollama)'}")

    try:
        import moderngl  # noqa: F401
        import pygame  # noqa: F401
        print("  display    : ok")
    except Exception:
        print("  display    : MISSING (pygame/moderngl)")

    print("-" * 32)
    return 0


def _run_text_mode(house: MrHouse) -> int:
    """Type questions instead of speaking — handy for testing the brain."""
    print("Text mode. Type a question (or 'quit').")
    print("-" * 40)
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        for sentence in house.brain.respond(text):
            print(f"house> {sentence}")
    return 0


def _list_devices() -> int:
    """Print available audio devices."""
    try:
        import sounddevice as sd
    except Exception as exc:
        print(f"sounddevice unavailable: {exc}")
        return 1
    print("Audio devices (index: name  [in/out channels]):")
    try:
        default_in, default_out = sd.default.device
    except Exception:
        default_in = default_out = None
    for i, d in enumerate(sd.query_devices()):
        mark = []
        if i == default_in:
            mark.append("DEFAULT-IN")
        if i == default_out:
            mark.append("DEFAULT-OUT")
        tag = ("  <- " + ",".join(mark)) if mark else ""
        print(f"  {i:>2}: {d['name']}  [{d['max_input_channels']}in/{d['max_output_channels']}out]{tag}")
    print("\nSet audio.input_device / audio.output_device in config.yaml to an "
          "index or a name substring.")
    return 0


def _mic_test(cfg) -> int:
    """Live meter of mic level + wake-word score, to diagnose 'it won't react'."""
    import numpy as np

    from .audio.capture import MicrophoneStream
    from .audio.wake_word import WakeWordDetector

    mic = MicrophoneStream(cfg.audio.sample_rate, cfg.audio.block_size, cfg.audio.input_device)
    if not mic.available:
        print("Microphone unavailable (sounddevice/PortAudio missing).")
        return 1
    wake = WakeWordDetector(cfg.wake_word.models, cfg.wake_word.threshold, repo_root=REPO_ROOT)

    print("Mic test. Speak — you should see the level move. Say your wake word")
    print(f"(models: {cfg.wake_word.models}) and watch the score spike past "
          f"{cfg.wake_word.threshold}. Press Ctrl-C to stop.\n")
    if not wake.available:
        print("(wake-word model not loaded; showing level only)\n")

    try:
        mic.start()
    except Exception as exc:
        print(f"Could not open microphone: {exc}")
        print("On macOS, grant mic access: System Settings > Privacy & Security > Microphone.")
        return 1

    seen_sound = False
    try:
        for block in mic.blocks():
            rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)) + 1e-9)
            seen_sound = seen_sound or rms > 30
            bars = int(min(40, rms / 50))
            score = 0.0
            if wake.available:
                wake.process(block)
                if wake.last_scores:
                    score = max(wake.last_scores.values())
            meter = "#" * bars + "-" * (40 - bars)
            hit = "  <== WAKE!" if score >= cfg.wake_word.threshold else ""
            print(f"\rlevel [{meter}] {rms:6.0f}   wake={score:0.2f}{hit}   ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        print()
        if not seen_sound:
            print("\n!! The level never moved — the mic is silent. Most likely your "
                  "terminal/IDE lacks microphone permission on macOS:\n"
                  "   System Settings > Privacy & Security > Microphone -> enable "
                  "your terminal app, then fully restart it.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Mr. House voice assistant")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--check", action="store_true", help="Report subsystem availability and exit")
    parser.add_argument("--no-display", action="store_true", help="Run without the CRT window")
    parser.add_argument("--fullscreen", action="store_true", help="Start the display in fullscreen (toggle at runtime with F11)")
    parser.add_argument("--windowed", action="store_true", help="Force windowed mode (override config fullscreen)")
    parser.add_argument("--text", action="store_true", help="Text-only brain test (no audio/display)")
    parser.add_argument("--list-devices", action="store_true", help="List audio input/output devices and exit")
    parser.add_argument("--mic-test", action="store_true", help="Live mic level + wake-word score meter")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    _setup_logging(cfg.log_level)

    # CLI overrides for the display mode.
    if args.fullscreen:
        cfg.display.fullscreen = True
    elif args.windowed:
        cfg.display.fullscreen = False

    if args.list_devices:
        return _list_devices()

    if args.mic_test:
        return _mic_test(cfg)

    if args.check:
        return _check(cfg)

    if args.text:
        from .brain.mcp_tools import MCPToolManager
        house = MrHouse(cfg)
        if house.tools is not None:
            house.tools.start()
        try:
            return _run_text_mode(house)
        finally:
            if house.tools is not None:
                house.tools.stop()

    # Live audio run: enforce a single instance so multiple Mr. Houses can't
    # listen and answer over each other.
    if not _acquire_single_instance_lock():
        print(
            "Another Mr. House is already running (it holds the microphone).\n"
            "Close that one first, or kill stragglers with:\n"
            "    pkill -f 'run.py'\n"
        )
        return 1

    house = MrHouse(cfg)

    headless = args.no_display or not cfg.display.enabled

    # Build the display up front so the signal handler can ask it to stop. When
    # headless it just runs a keep-alive wait loop on the main thread.
    from .display.window import CRTDisplay

    display = CRTDisplay(cfg.display, REPO_ROOT) if not headless else None

    _interrupts = {"n": 0}

    def _shutdown(*_):
        # Keep the handler LIGHTWEIGHT: just request stop. Heavy cleanup (thread
        # joins, audio/MCP teardown) happens after the main loop returns, so a
        # second Ctrl-C can't deadlock inside the handler.
        _interrupts["n"] += 1
        if _interrupts["n"] >= 2:
            # Second Ctrl-C: don't wait around — exit hard.
            import os

            os._exit(0)
        logging.getLogger(__name__).info("Shutting down... (press Ctrl-C again to force)")
        if display is not None:
            display.stop()
        house.request_stop()

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    house.start()

    if headless:
        # Block the main thread until stop is requested.
        try:
            while not house.stopped:
                import time

                time.sleep(0.2)
        except KeyboardInterrupt:
            house.request_stop()
        house.stop()
        return 0

    # Display owns the main thread; it returns when the window closes / ESC /
    # stop is requested.
    display.set_glitch_provider(lambda: house.is_speaking)
    display.run()
    house.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

