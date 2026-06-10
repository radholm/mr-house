"""Rolling conversation memory, with optional on-disk persistence.

Keeps the last *N* user/assistant turns so Mr. House has short-term context
without blowing the model's context window. The system prompt is stored
separately and always prepended.

If a ``persist_path`` is given, the rolling history is saved to disk after each
exchange and reloaded on startup — so Mr. House remembers previous
conversations across restarts (within the rolling window).
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


class ConversationMemory:
    def __init__(
        self,
        system_prompt: str,
        max_turns: int = 12,
        persist_path: Optional[str | Path] = None,
    ) -> None:
        self.system_prompt = system_prompt
        # Each "turn" is a user msg + assistant msg, so store 2*max_turns msgs.
        self._messages: deque[dict[str, Any]] = deque(maxlen=max_turns * 2)
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path is not None:
            self._load()

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})
        self._save()  # persist after a complete exchange

    def add_raw(self, message: dict[str, Any]) -> None:
        """Append a raw message dict (e.g. tool calls/results) to history."""
        self._messages.append(message)

    def build(self, extra_system: str | None = None) -> list[dict[str, Any]]:
        """Return the full message list for an LLM call."""
        system = self.system_prompt
        if extra_system:
            system = system + "\n\n" + extra_system
        return [{"role": "system", "content": system}, *list(self._messages)]

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self._messages if m.get("role") == "user")

    # -- persistence -------------------------------------------------------- #
    def _load(self) -> None:
        try:
            if self._persist_path and self._persist_path.exists():
                data = json.loads(self._persist_path.read_text(encoding="utf-8"))
                msgs = data.get("messages", []) if isinstance(data, dict) else data
                for m in msgs:
                    if m.get("role") in {"user", "assistant"} and m.get("content"):
                        self._messages.append({"role": m["role"], "content": m["content"]})
                if self._messages:
                    log.info(
                        "Loaded %d remembered messages from %s.",
                        len(self._messages), self._persist_path,
                    )
        except Exception as exc:
            log.warning("Could not load memory from %s: %s", self._persist_path, exc)

    def _save(self) -> None:
        if self._persist_path is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"messages": list(self._messages)}
            self._persist_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("Could not save memory to %s: %s", self._persist_path, exc)

    def clear(self) -> None:
        self._messages.clear()
        self._save()

