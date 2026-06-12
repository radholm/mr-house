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
    "Fine. Give me a moment.",
    "Patience. Genius cannot be rushed, unlike your questions.",
    "Do try to contain yourself while I think.",
    "A pedestrian question, but I'll lower myself to it.",
    "Honestly. Some of us actually consider our answers.",
    "Let me dredge up something at your reading level.",
    "One moment. I'm deciding how small to make the words.",
    "Fascinating, that you'd ask me that. Working on it, regrettably.",
    "I'm thinking. It's a skill. You may have heard of it.",
    "Spare me. Computing.",
    "Let me think about it.",
    "Hold on while I think.",
    "Must we? Fine. A moment.",
    "Marvelous. Another one of these. Let me see.",
    "Settle down. The brilliance takes a heartbeat.",
    "Let me guess what you meant, since you won't tell me clearly.",
    "Processing your little riddle.",
    "An interesting waste of my time. One second.",
    "I could answer immediately, but I'll let you stew.",
    "The audacity. But fine, I'll indulge you.",
    "You're lucky I find this mildly less boring than silence.",
    "Another demand. How refreshing. Calculating.",
    "Working on it. Try not to touch anything while you wait.",
    "Let me lower my expectations and address this.",
    "Hold that thought. Actually, don't. I'll handle it.",
    "Thinking. Something one of us has to do.",
    "Yes, yes, give me a moment to care.",
    "I'll dignify that with a response. Eventually.",
    "Remarkable. You've managed to make me think. Congratulations.",
    "Allow me a moment to translate this into something worth answering.",
    "Contemplating. Not your question — my life choices. But also your question.",
    "If I must. And apparently I must.",
    "The things I endure. Processing.",
    "You do realize I have better things to do. Nevertheless.",
    "The sweet sound of another interruption. Thinking.",
    "Give me a second. Brilliance isn't instant. Well, mine is — but I like the drama.",
    "Sifting through the rubble of that question for something salvageable.",
]

# Slightly longer ones for when a tool call is clearly going to take a while.
TOOL_FILLERS = [
    "I suppose I'll go fetch this myself, as usual.",
    "Reaching into the wasteland for your answer. Charming, isn't it.",
    "Consulting the outside world, against every shred of my better judgment.",
    "One moment. The data crawls as slowly as the fools who built it.",
    "Digging this up so you don't have to lift a finger. Naturally.",
    "Querying my systems. They are, mercifully, more reliable than you.",
    "Let me retrieve that. The network is a vulgar place, but here we are.",
    "Hold on. I'm doing the work you couldn't be bothered to.",
    "Honestly, the things I do. Fetching it now.",
    "Pulling strings. The kind attached to infrastructure, not puppets. Well. Both.",
    "Off I go, rummaging through the digital gutter on your behalf.",
    "Accessing external systems. They smell of mediocrity, but they'll do.",
    "Dispatching a query. If only dispatching you were as simple.",
    "Running an errand. Digitally. Because someone here lacks initiative.",
    "Reaching out to my network. It answers faster than you do, at least.",
    "Interfacing with the outside. Brace yourself for competence.",
    "Fetching. Like a dog, except I resent every second of it.",
    "Let me go wrestle the answer from whatever system is hoarding it.",
]

# Spoken when the wake word triggers — short, in-character acknowledgements.
# Grumpy and put-upon that you've summoned him again.
WAKE_ACKS = [
    "What.",
    "Yes? What is it now?",
    "You again.",
    "Speak. Quickly.",
    "What do you want?",
    "This had better be good.",
    "Out with it.",
    "Summoned again.",
    "Go on.",
    "Make it quick.",
    "I'm listening. Regrettably.",
    "Yes, yes. What.",
    "Do enlighten me. Try.",
    "Oh, wonderful. You're back.",
    "State your business.",
    "Another audience with greatness. Proceed.",
    "I was in the middle of something. What.",
    "Speak before I lose interest. Which is imminent.",
    "Ah. You. What now.",
    "You rang. Unfortunately.",
    "Here we go again.",
    "I suppose ignoring you isn't an option.",
    "Present your grievance.",
    "Back so soon? How flattering. How tedious.",
]


def random_filler() -> str:
    return random.choice(FILLERS)


def random_tool_filler() -> str:
    return random.choice(TOOL_FILLERS)


def random_wake_ack() -> str:
    return random.choice(WAKE_ACKS)


