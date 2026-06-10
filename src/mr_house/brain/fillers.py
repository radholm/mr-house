"""'Thinking' filler lines and wake acknowledgements.

Low-latency trick: the moment we send the question to the LLM we start a timer.
If the first token hasn't arrived by ``filler_after_ms``, we speak one of these
in-character lines so there's never awkward silence while the model (or a tool
call) works. They're written to sound natural mid-thought.

Tone: Mr. House — dry, aristocratic, sardonic. The humour is in the disdain:
mildly impatient, faintly condescending, never crude, never losing composure.
"""

from __future__ import annotations

import random

FILLERS = [
    "Patience. Genius cannot be rushed.",
    "Do try to contain your excitement while I think.",
    "Hm. A pedestrian question, but I'll indulge it.",
    "Give me a moment. Some of us actually consider our answers.",
    "Let me lower myself to the appropriate level for this.",
    "One moment. I'm deciding how much of this you'll understand.",
    "Fascinating, that you'd ask me that. Working on it.",
    "I'm thinking. It's a skill you may have heard of.",
    "A trivial matter. Allow me a heartbeat regardless.",
    "Computing. Try not to wander off.",
]

# Slightly longer ones for when a tool call is clearly going to take a while.
TOOL_FILLERS = [
    "I suppose I'll fetch this myself, as usual.",
    "Reaching into the wasteland for your answer. Charming, isn't it.",
    "Consulting the outside world, against my better judgment.",
    "One moment. The data streams are as slow as the people who built them.",
    "Pulling this from the network. Do mind the dust.",
    "Querying my systems. They're considerably more reliable than your sources.",
    "Let me retrieve that. The internet is a vulgar place, but useful.",
]

# Spoken when the wake word triggers — short, in-character acknowledgements.
# Dry and faintly amused that you've summoned him again.
WAKE_ACKS = [
    "Yes?",
    "You have my attention.",
    "Speak.",
    "Go on.",
    "I'm listening, regrettably.",
    "What is it now?",
    "Yes, what is it?",
    "Do enlighten me.",
    "Summoned again, I see.",
    "Mm. Proceed.",
    "Out with it.",
    "I'm all ears, such as they are.",
]


def random_filler() -> str:
    return random.choice(FILLERS)


def random_tool_filler() -> str:
    return random.choice(TOOL_FILLERS)


def random_wake_ack() -> str:
    return random.choice(WAKE_ACKS)


