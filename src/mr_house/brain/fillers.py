"""'Thinking' filler lines.

Low-latency trick: the moment we send the question to the LLM we start a timer.
If the first token hasn't arrived by ``filler_after_ms``, we speak one of these
in-character lines so there's never awkward silence while the model (or a tool
call) works. They're written to sound natural mid-thought.
"""

from __future__ import annotations

import random

FILLERS = [
    "Let me consult the data streams.",
    "One moment, cross-referencing.",
    "Hmm, allow me a moment.",
    "Checking the relevant records.",
    "Give me a heartbeat to verify.",
    "Interesting. Let me look into that.",
    "Accessing the appropriate channels.",
    "A moment of computation, if you would.",
    "Let me be certain before I answer.",
    "Querying my systems now.",
]

# Slightly longer ones for when a tool call is clearly going to take a while.
TOOL_FILLERS = [
    "Reaching out to my external sources.",
    "Pulling that from the outside world now.",
    "Let me fetch the latest on that.",
    "Consulting the relevant service.",
]


def random_filler() -> str:
    return random.choice(FILLERS)


def random_tool_filler() -> str:
    return random.choice(TOOL_FILLERS)

