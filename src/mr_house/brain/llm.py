"""LLM orchestration via Ollama (local), with streaming, tool-calling and
sentence-level chunking for low latency.

Key design choices for *low latency*:

* We **stream** tokens from the model and emit **complete sentences** as soon as
  they're ready, so TTS can start on sentence 1 while the model writes sentence 2.
* The first content token fires an ``on_first_token`` callback so the
  orchestrator can cancel any "thinking" filler.
* Tool calls are resolved in a loop: when the model requests a tool we run it
  (via :class:`MCPToolManager`), append the result, and continue — then stream
  the final spoken answer.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Iterator, Optional

from .local_tools import LocalToolRegistry

log = logging.getLogger(__name__)

try:
    import ollama
except Exception as exc:  # pragma: no cover
    ollama = None
    log.warning("ollama client unavailable (%s); brain disabled.", exc)


# Sentence boundary: end punctuation followed by *actual whitespace*. We avoid
# anchoring on end-of-buffer ($) during streaming, otherwise a trailing "13."
# at a chunk boundary is mistaken for a sentence end and splits decimals like
# "13.9". The final partial sentence is emitted by SentenceChunker.flush().
_SENTENCE_END = re.compile(r"([.!?…]+)(\s+)")


class SentenceChunker:
    """Accumulate streamed text and yield speakable sentences."""

    def __init__(self, min_chars: int = 12) -> None:
        self._buf = ""
        self.min_chars = min_chars

    def feed(self, text: str) -> list[str]:
        out: list[str] = []
        self._buf += text
        # Emit on newlines too.
        while True:
            match = _SENTENCE_END.search(self._buf)
            nl = self._buf.find("\n")
            cut = None
            if match and match.end() >= self.min_chars:
                cut = match.end()
            if nl != -1 and (cut is None or nl < cut):
                cut = nl + 1
            if cut is None:
                break
            sentence = self._buf[:cut].strip()
            self._buf = self._buf[cut:]
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> Optional[str]:
        rest = self._buf.strip()
        self._buf = ""
        return rest or None


class Brain:
    def __init__(self, cfg, memory, tools=None) -> None:
        self.cfg = cfg
        self.memory = memory
        self.tools = tools                       # MCP tool manager (may be None)
        self.local_tools = LocalToolRegistry()   # built-in tools (weather, time)
        self._client = None
        if ollama is not None:
            try:
                self._client = ollama.Client(host=cfg.host)
            except Exception as exc:
                log.error("Failed to create Ollama client: %s", exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    def _all_tool_schemas(self) -> list[dict[str, Any]]:
        """Combined local + MCP tool definitions for the model."""
        schemas = list(self.local_tools.openai_tools())
        if self.tools and self.tools.available:
            schemas += self.tools.openai_tools()
        return schemas

    def _run_tool(self, name: str, args: dict[str, Any]) -> str:
        """Route a tool call to the local registry or the MCP manager."""
        if self.local_tools.has(name):
            return self.local_tools.call(name, args)
        if self.tools and self.tools.available:
            return self.tools.call(name, args)
        return f"Tool {name} is unavailable."

    def _options(self) -> dict[str, Any]:
        return {
            "temperature": self.cfg.temperature,
            "num_ctx": self.cfg.num_ctx,
        }

    def respond(
        self,
        user_text: str,
        on_first_token: Optional[Callable[[], None]] = None,
        on_tool_call: Optional[Callable[[str], None]] = None,
    ) -> Iterator[str]:
        """Yield speakable sentences for *user_text* (streaming + tools).

        ``on_first_token`` is invoked once, when real content begins (used to
        cancel the thinking filler). ``on_tool_call`` is invoked with the tool
        name when a tool is about to run (used to play a longer filler).
        """
        if self._client is None:
            yield "My cognitive systems are offline at the moment."
            return

        self.memory.add_user(user_text)
        messages = self.memory.build()
        # Only expose tools when the question actually looks like it needs
        # external / live information. Small local models are otherwise very
        # tool-happy and will "fetch the internet" for things like "1 + 1".
        use_tools = _likely_needs_tools(user_text) and (
            self.local_tools.available or (self.tools and self.tools.available)
        )
        tool_schema = self._all_tool_schemas() if use_tools else None
        if use_tools:
            log.info("Tools enabled: %s", [t["function"]["name"] for t in tool_schema])

        chunker = SentenceChunker()
        first_token_sent = False
        full_answer: list[str] = []

        for _ in range(max(1, self.cfg.max_tool_iterations)):
            collected_content = ""
            tool_calls: list[dict[str, Any]] = []

            try:
                stream = self._client.chat(
                    model=self.cfg.model,
                    messages=messages,
                    tools=tool_schema,
                    options=self._options(),
                    stream=True,
                )
            except Exception as exc:
                log.error("Ollama chat failed: %s", exc)
                yield "I seem to be having trouble thinking just now."
                return

            for chunk in stream:
                msg = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", {})
                content = (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")) or ""
                calls = (msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)) or []

                if calls:
                    for c in calls:
                        tool_calls.append(_normalize_tool_call(c))

                if content:
                    if not first_token_sent and on_first_token:
                        on_first_token()
                        first_token_sent = True
                    collected_content += content
                    for sentence in chunker.feed(content):
                        full_answer.append(sentence)
                        yield sentence

            # If the model asked for tools, run them and loop again.
            if tool_calls and not collected_content.strip():
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_to_ollama_call(tc) for tc in tool_calls],
                })
                for tc in tool_calls:
                    name = tc["name"]
                    args = tc["arguments"]
                    if on_tool_call:
                        on_tool_call(name)
                    log.info("Tool call -> %s(%s)", name, args)
                    result = self._run_tool(name, args)
                    messages.append({"role": "tool", "name": name, "content": result})
                continue  # ask the model again now that it has tool output

            # Otherwise we have the final answer; flush trailing text.
            tail = chunker.flush()
            if tail:
                if not first_token_sent and on_first_token:
                    on_first_token()
                    first_token_sent = True
                full_answer.append(tail)
                yield tail
            break

        answer = " ".join(full_answer).strip()
        if answer:
            self.memory.add_assistant(answer)
        else:
            self.memory.add_assistant("(no response)")


# --------------------------------------------------------------------------- #
#  Tool gating                                                                 #
# --------------------------------------------------------------------------- #
# Words/phrases that suggest the question needs fresh, external, or web data.
_TOOL_HINTS = (
    "weather", "temperature", "forecast", "rain", "snow", "wind", "humidity",
    "news", "headline", "today", "tonight", "tomorrow", "currently", "current",
    "right now", "latest", "recent", "this week", "this year", "price", "cost of",
    "stock", "share price", "exchange rate", "crypto", "bitcoin", "ethereum",
    "score", "who won", "election", "release date", "released", "version of",
    "http://", "https://", "www.", ".com", ".org", ".net", "website", "web page",
    "webpage", "online", "search for", "look up", "lookup", "google", "wikipedia",
    "fetch", "download", "article", "url", "population of", "who is the current",
)


def _likely_needs_tools(text: str) -> bool:
    """Cheap heuristic: should the model be offered tools for this question?"""
    if not text:
        return False
    low = text.lower()
    return any(h in low for h in _TOOL_HINTS)


# --------------------------------------------------------------------------- #
#  Tool-call normalisation (handles dict and object shapes from ollama)        #
# --------------------------------------------------------------------------- #
def _normalize_tool_call(call: Any) -> dict[str, Any]:
    func = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
    name = func.get("name") if isinstance(func, dict) else getattr(func, "name", "")
    raw_args = func.get("arguments") if isinstance(func, dict) else getattr(func, "arguments", {})
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except Exception:
            raw_args = {}
    return {"name": name or "", "arguments": raw_args or {}}


def _to_ollama_call(tc: dict[str, Any]) -> dict[str, Any]:
    return {"function": {"name": tc["name"], "arguments": tc["arguments"]}}

