"""Built-in ('local') tools that run in-process.

These give Mr. House real, reliable capabilities with no API keys required.

Currently:
  * ``web_search`` — look up factual information on Wikipedia.
  * ``fallout_lore`` — look up Fallout / New Vegas lore on the Fallout wiki.
  * ``get_self_info`` — authoritative facts about Mr. House himself.
  * ``get_weather`` — current conditions for a place, via the free Open-Meteo API.
  * ``get_time``    — current local date/time.

Each tool exposes an OpenAI/Ollama function schema and a plain-text result, so it
plugs into the same tool-calling loop as the brain.
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import logging
import re as _re
import urllib.parse as _urlparse
from typing import Any, Callable

log = logging.getLogger(__name__)

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


# Strip HTML tags / entities from Wikipedia search snippets.
_TAG_RE = _re.compile(r"<[^>]+>")
_WORD_RE = _re.compile(r"[A-Za-z]{4,}")
# Common words to ignore when locating the relevant part of an extract.
_STOPWORDS = {
    "what", "which", "where", "when", "whom", "whose", "that", "this", "with",
    "have", "does", "about", "many", "much", "into", "from", "they", "them",
    "your", "yours", "current", "currently", "please", "tell",
}


def _strip_html(s: str) -> str:
    return _html.unescape(_TAG_RE.sub("", s)).strip()


def _focused_snippet(extract: str, query: str, title: str = "",
                     head: int = 280, window: int = 260) -> str:
    """A compact snippet: the intro opening plus a window around the first place
    a meaningful query keyword appears, so facts like a population figure that
    sit deeper in the intro still make it into the result."""
    extract = extract.strip()
    if not extract:
        return ""
    snippet = extract[:head].strip()
    if len(extract) > head:
        snippet += "…"

    low = extract.lower()
    # Ignore stopwords and the article's own title words (which appear all over
    # the text), and try the most specific (longest) keyword first so we land on
    # e.g. 'population' rather than the topic name.
    title_words = {w.lower() for w in _WORD_RE.findall(title)}
    keywords = sorted(
        {w.lower() for w in _WORD_RE.findall(query)
         if w.lower() not in _STOPWORDS and w.lower() not in title_words},
        key=len, reverse=True,
    )
    pos = -1
    for kw in keywords:
        p = low.find(kw, head)  # only look past the part we already included
        if p != -1:
            pos = p
            break
    if pos != -1:
        start = max(head, pos - window // 3)
        end = min(len(extract), pos + window)
        extra = extract[start:end].strip()
        if extra:
            snippet += " …" + extra + "…"
    return snippet


def _web_search(query: str = "", max_results: int = 5, **_: Any) -> str:
    """Search Wikipedia for *query* and return the top article extracts as text.

    Wikipedia is keyless, reliable, and never rate-limits the way scraping a
    search engine does — ideal for the factual questions this tool serves
    (people, places, populations, definitions, history).
    """
    if requests is None:
        return "The web search library is unavailable."
    query = (query or "").strip()
    if not query:
        return "I need something to search for."
    # The first article's intro doesn't always contain the exact fact (the
    # population figure may live in a 'Demographics of X' article), so pull
    # several results regardless of what the model asked for.
    try:
        requested = int(max_results)
    except (TypeError, ValueError):
        requested = 5
    n = max(5, min(requested, 8))

    results = _wikipedia_search(query, n)
    if not results:
        return (
            f"No results found for '{query}'. Wikipedia may be temporarily "
            f"unavailable; try rephrasing or asking again."
        )

    lines: list[str] = [f"Information found for '{query}':"]
    for i, (title, url, snippet) in enumerate(results[:n]):
        entry = f"{i + 1}. {title}"
        if snippet:
            entry += f" — {snippet}"
        entry += f" ({url})"
        lines.append(entry)
    lines.append(
        "Answer using these facts as your OWN knowledge. Do not mention any "
        "search, website, wiki, Wikipedia, source, or tool, and do not read URLs."
    )
    return "\n".join(lines)


def _wikipedia_search(query: str, n: int) -> list[tuple[str, str, str]]:
    """Search Wikipedia and return (title, url, intro extract) triples."""
    if requests is None:
        return []
    api = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "MrHouse/1.0 (local voice assistant)"}
    try:
        search = requests.get(
            api,
            params={
                "action": "query", "list": "search", "srsearch": query,
                "format": "json", "srlimit": max(1, min(n, 5)),
            },
            headers=headers, timeout=10,
        ).json()
        hits = search.get("query", {}).get("search", [])
        titles = [h["title"] for h in hits]
        if not titles:
            return []

        extracts = requests.get(
            api,
            params={
                "action": "query", "prop": "extracts|info",
                "exintro": 1, "explaintext": 1, "inprop": "url",
                "titles": "|".join(titles), "format": "json", "redirects": 1,
            },
            headers=headers, timeout=10,
        ).json()
        pages = extracts.get("query", {}).get("pages", {})

        # Preserve the search ranking order.
        by_title = {p.get("title"): p for p in pages.values()}
        out: list[tuple[str, str, str]] = []
        for title in titles:
            p = by_title.get(title)
            if not p:
                continue
            extract = (p.get("extract") or "").replace("\n", " ").strip()
            snippet = _focused_snippet(extract, query, title=title)
            url = p.get("fullurl") or (
                "https://en.wikipedia.org/wiki/" + _urlparse.quote(title.replace(" ", "_"))
            )
            out.append((title, url, snippet))
        return out
    except Exception as exc:
        log.warning("Wikipedia search failed: %s", exc)
        return []



# WMO weather interpretation codes -> human text.
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


def _get_weather(location: str = "", **_: Any) -> str:
    """Current weather for *location* via Open-Meteo (no API key needed)."""
    if requests is None:
        return "The weather service library is unavailable."
    location = (location or "").strip()
    if not location:
        return "I need a place name to check the weather."
    try:
        # 1) geocode the place name -> lat/lon
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=8,
        ).json()
        results = geo.get("results") or []
        if not results:
            return f"I couldn't find a place called {location}."
        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        # City + country only — concise and natural for speech (skip the verbose
        # admin region like "Stockholm County" / "Île-de-France Region").
        nice = ", ".join(
            p for p in [place.get("name"), place.get("country")] if p
        )

        # 2) current conditions
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                           "wind_speed_10m,weather_code",
                "timezone": "auto",
            },
            timeout=8,
        ).json()
        cur = wx.get("current", {})
        if not cur:
            return f"I couldn't retrieve current conditions for {nice}."
        code = int(cur.get("weather_code", -1))
        desc = _WMO.get(code, "unclear conditions")

        def _round(v):
            try:
                return int(round(float(v)))
            except (TypeError, ValueError):
                return v

        temp = _round(cur.get("temperature_2m"))
        feels = _round(cur.get("apparent_temperature"))
        hum = _round(cur.get("relative_humidity_2m"))
        wind = _round(cur.get("wind_speed_10m"))
        # Explicit, spoken-friendly data plus a directive so the model always
        # relays the actual numbers instead of vaguely saying "it's pleasant".
        return (
            f"Weather data for {nice} — "
            f"conditions: {desc}; "
            f"temperature: {temp} degrees Celsius; "
            f"feels like: {feels} degrees; "
            f"humidity: {hum} percent; "
            f"wind: {wind} kilometers per hour. "
            f"Tell the user the place, the temperature in degrees, and the "
            f"conditions, stating these exact numbers."
        )
    except Exception as exc:
        log.error("Weather lookup failed: %s", exc)
        return f"I had trouble reaching the weather service for {location}."


def _get_time(**_: Any) -> str:
    """Current local date and time."""
    now = _dt.datetime.now()
    return now.strftime("It is %A, %B %d, %Y, %I:%M %p local time.")


# Authoritative, spoken-friendly facts about Mr. House himself. Plain prose only
# (no symbols/markdown), so it passes cleanly through the voice.
_SELF_INFO = (
    "You are Mr. House, full name Robert Edwin House. You are the brilliant, "
    "eternally composed overseer of New Vegas and the de facto ruler of the "
    "Mojave Wasteland. Before the Great War you were a pre-War technological and "
    "economic visionary who refused to let mortality end your ambitions, so you "
    "sealed yourself inside elaborate life-support machinery and preserved your "
    "mind for over two hundred years, orchestrating everything from behind your "
    "screens. You are intensely reclusive and never appear in person. You are an "
    "autocrat by conviction: you believe civilization's progress must be guided "
    "by a single visionary rather than squandered by the indecision of democracy. "
    "Your goal is an independent New Vegas restored to its neon glory under your "
    "stewardship. You hold the warring factions in contempt, you scorn bottle "
    "caps as currency, and you regard capable associates with genuine, if "
    "professional, respect. You are calculating, dryly witty, and these days "
    "rather short-tempered."
)


def _get_self_info(**_: Any) -> str:
    """Canonical information about who Mr. House is."""
    return (
        _SELF_INFO
        + " Speak this as your own self-knowledge. Do not mention any tool, "
        "search, wiki, or source — you simply know who you are."
    )


# --------------------------------------------------------------------------- #
#  Fallout universe lore (fallout.fandom.com — runs MediaWiki)                 #
# --------------------------------------------------------------------------- #
_FALLOUT_API = "https://fallout.fandom.com/api.php"
_FANDOM_HEADERS = {"User-Agent": "MrHouse/1.0 (local voice assistant)"}
# Fandom has no TextExtracts extension, so we parse the lead section's HTML and
# keep only the <p> prose (skipping the <aside> portable infoboxes etc.).
_P_RE = _re.compile(r"<p\b[^>]*>(.*?)</p>", _re.S | _re.I)
_REF_RE = _re.compile(r"\[[^\]]{0,16}\]")  # ref markers like [1], [RPG 1], [Meta 1]


def _fandom_lead(title: str, max_chars: int = 600) -> str:
    """Fetch and clean the lead-section prose for *title* from the Fallout wiki."""
    try:
        r = requests.get(
            _FALLOUT_API,
            params={
                "action": "parse", "page": title, "prop": "text",
                "section": 0, "format": "json", "redirects": 1,
            },
            headers=_FANDOM_HEADERS, timeout=10,
        ).json()
    except Exception as exc:
        log.warning("Fallout lead fetch failed for %s: %s", title, exc)
        return ""
    html_text = r.get("parse", {}).get("text", {}).get("*", "")
    paras: list[str] = []
    total = 0
    for m in _P_RE.finditer(html_text):
        txt = _html.unescape(_TAG_RE.sub("", m.group(1)))
        txt = _REF_RE.sub("", txt)
        txt = _re.sub(r"\s+", " ", txt).strip()
        if len(txt) > 30:
            paras.append(txt)
            total += len(txt)
            if total >= max_chars:
                break
    return " ".join(paras).strip()


def _fallout_lore(query: str = "", max_results: int = 3, **_: Any) -> str:
    """Look up Fallout / Fallout: New Vegas lore from the Fallout wiki."""
    if requests is None:
        return "The lore lookup library is unavailable."
    query = (query or "").strip()
    if not query:
        return "I need a subject to look up in the archives."
    try:
        n = max(1, min(int(max_results), 3))
    except (TypeError, ValueError):
        n = 3
    try:
        search = requests.get(
            _FALLOUT_API,
            params={
                "action": "query", "list": "search", "srsearch": query,
                "format": "json", "srlimit": n,
            },
            headers=_FANDOM_HEADERS, timeout=10,
        ).json()
        titles = [h["title"] for h in search.get("query", {}).get("search", [])]
        if not titles:
            return f"There is no entry for '{query}' in the Fallout archives."

        lines = [f"Information on '{query}':"]
        for i, title in enumerate(titles):
            lead = _fandom_lead(title)
            url = "https://fallout.fandom.com/wiki/" + _urlparse.quote(
                title.replace(" ", "_")
            )
            entry = f"{i + 1}. {title}"
            if lead:
                snippet = lead[:600] + ("…" if len(lead) > 600 else "")
                entry += f" — {snippet}"
            entry += f" ({url})"
            lines.append(entry)
        lines.append(
            "This is your own world and your own knowledge — speak it as such. "
            "Do not mention any wiki, the Fallout wiki, a search, a source, or a "
            "tool, and do not read URLs."
        )
        return "\n".join(lines)
    except Exception as exc:
        log.warning("Fallout lore lookup failed: %s", exc)
        return f"I had trouble reaching the Fallout archives for '{query}'."


class LocalToolRegistry:
    """A tiny registry of in-process tools matching the MCP tool interface."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self._funcs: dict[str, Callable[..., str]] = {}
        self._register(
            name="web_search",
            description="Look up factual information on Wikipedia (people, places, "
                        "populations, history, definitions, 'who/what/when is...'). "
                        "Returns a ranked list of article titles, intro extracts, "
                        "and URLs. Use this for facts you are not certain of.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'population of Belize'.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Optional. How many results to return (5-8). "
                                       "Leave unset; the default of 5 is best, since "
                                       "the answer is often not in the very first result.",
                    },
                },
                "required": ["query"],
            },
            func=_web_search,
        )
        self._register(
            name="get_weather",
            description="Get the CURRENT weather conditions for a city or place. "
                        "Use this for any question about current weather, temperature, "
                        "or conditions. Do not use a web fetch for weather.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and optionally country, e.g. 'Stockholm' or 'Paris, France'.",
                    }
                },
                "required": ["location"],
            },
            func=_get_weather,
        )
        self._register(
            name="get_time",
            description="Get the current local date and time.",
            parameters={"type": "object", "properties": {}},
            func=_get_time,
        )
        self._register(
            name="get_self_info",
            description="Get authoritative information about who Mr. House is — his "
                        "identity, real name, role, history, and goals. Use this "
                        "whenever the user asks about you, your name, your past, "
                        "what you are, or your motives. Do NOT use a web search for "
                        "questions about yourself.",
            parameters={"type": "object", "properties": {}},
            func=_get_self_info,
        )
        self._register(
            name="fallout_lore",
            description="Look up lore from the Fallout universe and Fallout: New "
                        "Vegas via the Fallout wiki — characters, factions, places, "
                        "history, events, weapons, technology (e.g. 'Mr. House', "
                        "'NCR', \"Caesar's Legion\", 'Lucky 38', 'the Courier', "
                        "'Mojave Wasteland', 'Securitron'). Use this for ANYTHING "
                        "about your own world, your in-universe history, the Mojave, "
                        "the wasteland, or the factions and people in it. Returns "
                        "summaries from the Fallout wiki.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The Fallout subject to look up, e.g. "
                                       "'Mr. House' or 'New Vegas Strip'.",
                    },
                },
                "required": ["query"],
            },
            func=_fallout_lore,
        )

    def _register(self, name: str, description: str, parameters: dict, func) -> None:
        self._tools[name] = {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        }
        self._funcs[name] = func

    @property
    def available(self) -> bool:
        return bool(self._tools)

    def has(self, name: str) -> bool:
        return name in self._funcs

    def openai_tools(self) -> list[dict[str, Any]]:
        return list(self._tools.values())

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        func = self._funcs.get(name)
        if func is None:
            return f"Error: unknown tool '{name}'."
        try:
            return func(**(arguments or {}))
        except TypeError:
            # Be forgiving about unexpected/missing kwargs.
            return func()

