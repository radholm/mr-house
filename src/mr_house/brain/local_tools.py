"""Built-in ('local') tools that run in-process.

These give Mr. House real, reliable capabilities with no API keys required.

Currently:
  * ``web_search`` — look up factual information on Wikipedia.
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

    lines: list[str] = [f"Wikipedia results for '{query}':"]
    for i, (title, url, snippet) in enumerate(results[:n]):
        entry = f"{i + 1}. {title}"
        if snippet:
            entry += f" — {snippet}"
        entry += f" ({url})"
        lines.append(entry)
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
        nice = ", ".join(
            p for p in [place.get("name"), place.get("admin1"), place.get("country")] if p
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
        temp = cur.get("temperature_2m")
        feels = cur.get("apparent_temperature")
        hum = cur.get("relative_humidity_2m")
        wind = cur.get("wind_speed_10m")
        return (
            f"Current weather in {nice}: {desc}, {temp} degrees Celsius "
            f"(feels like {feels}), humidity {hum} percent, wind {wind} kilometers per hour."
        )
    except Exception as exc:
        log.error("Weather lookup failed: %s", exc)
        return f"I had trouble reaching the weather service for {location}."


def _get_time(**_: Any) -> str:
    """Current local date and time."""
    now = _dt.datetime.now()
    return now.strftime("It is %A, %B %d, %Y, %I:%M %p local time.")


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

