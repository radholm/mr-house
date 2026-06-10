"""Lightweight CI smoke test (no model downloads, no mic/display required).

Imports every module so we catch import-time / syntax / cross-platform errors,
and loads the config. Intentionally does NOT instantiate the heavy models
(Whisper/Piper/Ollama), so it runs fast on a headless CI runner.
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODULES = [
    "mr_house",
    "mr_house.config",
    "mr_house.app",
    "mr_house.main",
    "mr_house.audio.capture",
    "mr_house.audio.wake_word",
    "mr_house.audio.vad",
    "mr_house.audio.stt",
    "mr_house.audio.tts",
    "mr_house.audio.effects",
    "mr_house.audio.player",
    "mr_house.brain.llm",
    "mr_house.brain.personality",
    "mr_house.brain.memory",
    "mr_house.brain.fillers",
    "mr_house.brain.mcp_tools",
    "mr_house.brain.local_tools",
    "mr_house.display.window",
]


def main() -> int:
    failed = []
    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"  ok   {name}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAIL {name}: {exc!r}")
            failed.append((name, exc))

    # Config must load and produce the expected nested structure.
    from mr_house.config import load_config

    cfg = load_config()
    assert cfg.stt.model, "stt.model missing"
    assert cfg.display.shader.scanline_count > 0, "shader config missing"
    print(f"  config loaded: stt.model={cfg.stt.model}, "
          f"wake={cfg.wake_word.models}, llm={cfg.brain.model}")

    # Personality prompt + local tools schema build without errors.
    from mr_house.brain.personality import build_system_prompt
    from mr_house.brain.local_tools import LocalToolRegistry

    assert "Mr. House" in build_system_prompt()
    tools = LocalToolRegistry().openai_tools()
    names = [t["function"]["name"] for t in tools]
    assert "get_weather" in names, "get_weather tool missing"
    print(f"  local tools: {names}")

    if failed:
        print(f"\nSMOKE FAILED: {len(failed)} module(s) failed to import.")
        return 1
    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

