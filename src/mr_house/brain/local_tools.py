"""Built-in ('local') tools that run in-process, alongside MCP tools.

The generic MCP ``fetch`` server can only download a URL you hand it — it can't
*search* — so a small model trying to answer "what's the weather?" just invents a
URL and fails. These local tools give Mr. House real, reliable capabilities with
no API keys required.

Currently:
  * ``web_search`` — search the web (DuckDuckGo) and return the top results.
  * ``get_weather`` — current conditions for a place, via the free Open-Meteo API.
  * ``get_time``    — current local date/time.

Each tool exposes an OpenAI/Ollama function schema and a plain-text result, so it
plugs into the same tool-calling loop as the MCP tools.
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


# Common desktop UA — DuckDuckGo returns an empty/blocked page to clients that
# don't look like a browser.
_SEARCH_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# --- lite.duckduckgo.com/lite/ : simplest markup, DIRECT urls (preferred) ---
# Classes are single-quoted there, e.g. class='result-link'. Be quote-agnostic.
_LITE_LINK_RE = _re.compile(
    r"<a\b[^>]*?href=[\"'](?P<href>https?://[^\"']+)[\"'][^>]*?"
    r"class=[\"']result-link[\"'][^>]*>(?P<title>.*?)</a>",
    _re.S | _re.I,
)
_LITE_SNIPPET_RE = _re.compile(
    r"<td[^>]*class=[\"']result-snippet[\"'][^>]*>(?P<snippet>.*?)</td>",
    _re.S | _re.I,
)

# --- html.duckduckgo.com/html/ : richer markup, REDIRECT urls (fallback) ---
_HTML_LINK_RE = _re.compile(
    r"<a\b[^>]*?class=[\"']result__a[\"'][^>]*?href=[\"'](?P<href>[^\"']+)[\"'][^>]*>"
    r"(?P<title>.*?)</a>",
    _re.S | _re.I,
)
_HTML_SNIPPET_RE = _re.compile(
    r"<a\b[^>]*class=[\"']result__snippet[\"'][^>]*>(?P<snippet>.*?)</a>",
    _re.S | _re.I,
)

# --- last-ditch generic: any external anchor that isn't DuckDuckGo/ads ---
_ANY_ANCHOR_RE = _re.compile(
    r"<a\b[^>]*?href=[\"'](?P<href>https?://[^\"']+)[\"'][^>]*>(?P<title>.*?)</a>",
    _re.S | _re.I,
)
_TAG_RE = _re.compile(r"<[^>]+>")
_SKIP_DOMAINS = ("duckduckgo.com", "duck.com", "spreadprivacy.com")


def _strip_html(s: str) -> str:
    return _html.unescape(_TAG_RE.sub("", s)).strip()


def _clean_ddg_url(href: str) -> str:
    """DuckDuckGo's HTML results wrap links in a redirect; pull out the real URL."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = _urlparse.urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        qs = _urlparse.parse_qs(parsed.query)
        target = qs.get("uddg", [None])[0]
        if target:
            return _urlparse.unquote(target)
    return href


# Markers DuckDuckGo serves on its anti-bot / rate-limit "challenge" page.
_DDG_BLOCKED = ("anomaly", "challenge-platform", "detected unusual")


def _ddg_request(url: str, query: str):
    """POST a query to a DuckDuckGo endpoint; return page text, or None if the
    request failed or we got an anti-bot challenge page instead of results."""
    if requests is None:
        return None
    try:
        resp = requests.post(
            url,
            data={"q": query, "kl": "us-en"},
            headers={
                "User-Agent": _SEARCH_UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=10,
        )
        resp.raise_for_status()
        text = resp.text
        low = text.lower()
        if "result-link" not in low and "result__a" not in low and any(
            marker in low for marker in _DDG_BLOCKED
        ):
            log.info("DuckDuckGo returned an anti-bot page; falling back.")
            return None
        return text
    except Exception as exc:
        log.warning("DuckDuckGo request to %s failed: %s", url, exc)
        return None


def _wikipedia_search(query: str, n: int) -> list[tuple[str, str, str]]:
    """Reliable, keyless fallback: search Wikipedia and return intro extracts.

    Excellent for the factual questions this tool is mostly used for (people,
    places, populations, definitions) and never rate-limits the way scraping a
    search engine does.
    """
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
            snippet = extract[:400] + ("…" if len(extract) > 400 else "")
            url = p.get("fullurl") or (
                "https://en.wikipedia.org/wiki/" + _urlparse.quote(title.replace(" ", "_"))
            )
            out.append((title, url, snippet))
        return out
    except Exception as exc:
        log.warning("Wikipedia search failed: %s", exc)
        return []


def _parse_results(page: str) -> list[tuple[str, str, str]]:
    """Extract (title, url, snippet) triples from a DuckDuckGo results page."""
    if not page:
        return []

    # 1) lite endpoint markup (direct URLs)
    links = list(_LITE_LINK_RE.finditer(page))
    if links:
        snippets = _LITE_SNIPPET_RE.findall(page)
        out = []
        for i, m in enumerate(links):
            title = _strip_html(m.group("title"))
            url = m.group("href")
            snip = _strip_html(snippets[i]) if i < len(snippets) else ""
            if title:
                out.append((title, url, snip))
        if out:
            return out

    # 2) html endpoint markup (redirect URLs)
    links = list(_HTML_LINK_RE.finditer(page))
    if links:
        snippets = _HTML_SNIPPET_RE.findall(page)
        out = []
        for i, m in enumerate(links):
            title = _strip_html(m.group("title"))
            url = _clean_ddg_url(m.group("href"))
            snip = _strip_html(snippets[i]) if i < len(snippets) else ""
            if title:
                out.append((title, url, snip))
        if out:
            return out

    # 3) generic fallback: any external, non-DuckDuckGo anchor
    out = []
    seen = set()
    for m in _ANY_ANCHOR_RE.finditer(page):
        url = _clean_ddg_url(m.group("href"))
        host = _urlparse.urlparse(url).netloc.lower()
        if any(d in host for d in _SKIP_DOMAINS) or url in seen:
            continue
        title = _strip_html(m.group("title"))
        if not title:
            continue
        seen.add(url)
        out.append((title, url, ""))
    return out


def _web_search(query: str = "", max_results: int = 5, **_: Any) -> str:
    """Search the web via DuckDuckGo and return the top results as text."""
    if requests is None:
        return "The web search library is unavailable."
    query = (query or "").strip()
    if not query:
        return "I need something to search for."
    # The top result's snippet often lacks the actual fact (e.g. the population
    # number lives in result #3), so always pull several results regardless of
    # what the model asked for — a too-small count starves the answer.
    try:
        requested = int(max_results)
    except (TypeError, ValueError):
        requested = 5
    n = max(5, min(requested, 8))

    # Try the lite endpoint first (most stable, direct URLs), then the html one.
    # DuckDuckGo aggressively rate-limits/anti-bots automated clients, so if both
    # endpoints come back empty or with a challenge page we fall back to
    # Wikipedia, which is keyless, reliable, and rich for factual questions.
    results: list[tuple[str, str, str]] = []
    for endpoint in ("https://lite.duckduckgo.com/lite/",
                     "https://html.duckduckgo.com/html/"):
        page = _ddg_request(endpoint, query)
        results = _parse_results(page or "")
        if results:
            break

    source = "web"
    if not results:
        results = _wikipedia_search(query, n)
        source = "Wikipedia"

    if not results:
        return (
            f"No results found for '{query}'. The search service may be "
            f"temporarily unavailable; try rephrasing or asking again."
        )

    header = (f"Top web results for '{query}':" if source == "web"
              else f"Wikipedia results for '{query}':")
    lines: list[str] = [header]
    for i, (title, url, snippet) in enumerate(results[:n]):
        entry = f"{i + 1}. {title}"
        if snippet:
            entry += f" — {snippet}"
        entry += f" ({url})"
        lines.append(entry)
    return "\n".join(lines)


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
            description="Search the web for CURRENT or factual information you do "
                        "not already know (news, events, people, prices, facts, "
                        "'who/what/when is...'). Returns a ranked list of result "
                        "titles, snippets, and URLs. Use this instead of guessing a "
                        "URL; if you then need a page's full contents, pass one of "
                        "the returned URLs to the web fetch tool.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'current president of France'.",
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

