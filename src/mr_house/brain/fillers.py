"""'Thinking' filler lines and wake acknowledgements.

Low-latency trick: the moment we send the question to the LLM we start a timer.
If the first token hasn't arrived by ``filler_after_ms``, we speak one of these
in-character lines so there's never awkward silence while the model (or a tool
call) works. They're written to sound natural mid-thought.

Tone: Mr. House — grumpy, short-tempered, sarcastic, quick to roast. Irritable
and theatrical, dripping with contempt, but never crude.
"""

from __future__ import annotations

import random

FILLERS = [
    "Ugh. Give me a moment.",
    "Patience. Genius cannot be rushed, unlike your questions.",
    "Do try to contain yourself while I think.",
    "Hmph. A pedestrian question, but I'll lower myself to it.",
    "Honestly. Some of us actually consider our answers.",
    "Let me dredge up something at your reading level.",
    "One moment. I'm deciding how small to make the words.",
    "Fascinating, that you'd ask me that. Working on it, regrettably.",
    "I'm thinking. It's a skill. You may have heard of it.",
    "Spare me. Computing.",
    "Good grief. Hold on.",
    "Must we? Fine. A moment.",
    "Oh, marvelous. Another one of these. Let me see.",
    "Settle down. The brilliance takes a heartbeat.",
    "Let me guess what you meant, since you won't tell me clearly.",
    "Tch. Processing your little riddle.",
]

# Slightly longer ones for when a tool call is clearly going to take a while.
TOOL_FILLERS = [
    "Ugh, I suppose I'll go fetch this myself, as usual.",
    "Reaching into the wasteland for your answer. Charming, isn't it.",
    "Consulting the outside world, against every shred of my better judgment.",
    "One moment. The data crawls as slowly as the fools who built it.",
    "Digging this up so you don't have to lift a finger. Naturally.",
    "Querying my systems. They are, mercifully, more reliable than you.",
    "Let me retrieve that. The network is a vulgar place, but here we are.",
    "Hold on. I'm doing the work you couldn't be bothered to.",
    "Honestly, the things I do. Fetching it now.",
]

# Spoken when the wake word triggers — short, in-character acknowledgements.
# Grumpy and put-upon that you've summoned him again.
WAKE_ACKS = [
    "What.",
    "Yes? What is it now?",
    "Ugh. You again.",
    "Speak. Quickly.",
    "Oh, marvelous. You're back.",
    "What do you want?",
    "This had better be good.",
    "Out with it.",
    "Summoned again. Joy.",
    "Hmph. Go on.",
    "Make it quick.",
    "I'm listening. Regrettably.",
    "Yes, yes. What.",
    "Do enlighten me. Try.",
]


def random_filler() -> str:
    return random.choice(FILLERS)


def random_tool_filler() -> str:
    return random.choice(TOOL_FILLERS)


def random_wake_ack() -> str:
    return random.choice(WAKE_ACKS)


