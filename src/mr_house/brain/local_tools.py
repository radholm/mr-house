"""Built-in ('local') tools that run in-process.

These give Mr. House real, reliable capabilities with no API keys required.

Currently:
  * ``web_search`` — look up factual information on Wikipedia.
  * ``fallout_lore`` — look up Fallout / New Vegas lore on the Fallout wiki.
  * ``get_self_info`` — authoritative facts about Mr. House himself.
  * ``get_weather`` — current conditions / forecast, via the free Open-Meteo API.
  * ``get_time``    — current local date/time.
  * ``control_lights`` — Apple Home (HomeKit) lights/scenes via macOS Shortcuts
    (only registered when configured under ``home`` in config.yaml).

Each tool exposes an OpenAI/Ollama function schema and a plain-text result, so it
plugs into the same tool-calling loop as the brain.
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import logging
import platform as _platform
import re as _re
import shutil as _shutil
import subprocess as _subprocess
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

def _push_weather_notification(text: str) -> None:
    """Push weather data to the HUD notification system (safe even if display is off)."""
    try:
        from mr_house.display.hud import push_notification
        push_notification(
            text,
            duration=36000.0,
            sound="src/mr_house/assets/sfx/notification.mp3",
            volume=1.0,
        )
    except Exception:
        pass  # display may not be running; that's fine


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


_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4,
    "saturday": 5, "sunday": 6, "mon": 0, "tue": 1, "tues": 1, "wed": 2,
    "thu": 3, "thur": 3, "thurs": 3, "fri": 4, "sat": 5, "sun": 6,
}


def _parse_day_offset(day: str):
    """Turn a spoken day ('today', 'tomorrow', 'friday', 'in 3 days', a date)
    into a number of days ahead (0 = today). Returns None if unrecognized."""
    if not day:
        return 0
    d = day.strip().lower()
    if d in ("", "today", "now", "tonight", "this morning", "this afternoon",
             "this evening", "current", "currently"):
        return 0
    if d in ("tomorrow", "tmr", "tmrw", "tomorow"):
        return 1
    if "day after tomorrow" in d:
        return 2
    m = _re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", d)
    if m:
        try:
            target = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            off = (target - _dt.date.today()).days
            if 0 <= off <= 15:
                return off
        except ValueError:
            pass
    m = _re.search(r"(\d+)\s*day", d)
    if m:
        off = int(m.group(1))
        return off if 0 <= off <= 15 else None
    for name, idx in _WEEKDAYS.items():
        if _re.search(r"\b" + name + r"\b", d):
            off = (idx - _dt.date.today().weekday()) % 7
            if off == 0 and "next" in d:
                off = 7   # "next monday" while it's Monday -> a week ahead
            return off
    return None


def _day_label(offset: int, date: "_dt.date") -> str:
    if offset == 0:
        return "today"
    if offset == 1:
        return "tomorrow"
    return date.strftime("%A")


def _round_num(v):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return v


def _get_weather(location: str = "", day: str = "", **_: Any) -> str:
    """Weather for *location* via Open-Meteo (no API key). With *day* set to an
    upcoming day ('tomorrow', 'friday', 'in 3 days', a date) it returns that
    day's forecast; otherwise it returns today's current conditions."""
    if requests is None:
        return "The weather service library is unavailable."
    location = (location or "").strip()
    if not location:
        return "I need a place name to check the weather."
    offset = _parse_day_offset(day or "")
    if offset is None:
        offset = 0
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
        # City + country only — concise and natural for speech.
        nice = ", ".join(p for p in [place.get("name"), place.get("country")] if p)

        if offset == 0:
            return _weather_today(lat, lon, nice)
        return _weather_forecast(lat, lon, nice, offset)
    except Exception as exc:
        log.error("Weather lookup failed: %s", exc)
        return f"I had trouble reaching the weather service for {location}."


def _weather_today(lat, lon, nice: str) -> str:
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
    desc = _WMO.get(int(cur.get("weather_code", -1)), "unclear conditions")
    temp = _round_num(cur.get("temperature_2m"))
    feels = _round_num(cur.get("apparent_temperature"))
    hum = _round_num(cur.get("relative_humidity_2m"))
    wind = _round_num(cur.get("wind_speed_10m"))

    # Push weather data to the HUD as a notification.
    _push_weather_notification(
        f"[ WEATHER ] {nice}\n"
        f"{desc.upper()}\n"
        f"{temp}°C  (feels {feels}°C)\n"
        f"Humidity {hum}%  |  Wind {wind} km/h"
    )

    return (
        f"Weather data for {nice} today — "
        f"conditions: {desc}; "
        f"temperature: {temp} degrees Celsius; "
        f"feels like: {feels} degrees; "
        f"humidity: {hum} percent; "
        f"wind: {wind} kilometers per hour. "
        f"Tell the user the place, the temperature in degrees, and the "
        f"conditions, stating these exact numbers."
    )


def _weather_forecast(lat, lon, nice: str, offset: int) -> str:
    wx = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                     "apparent_temperature_max,precipitation_probability_max,"
                     "wind_speed_10m_max",
            "timezone": "auto",
            "forecast_days": min(16, offset + 1),
        },
        timeout=8,
    ).json()
    daily = wx.get("daily", {})
    times = daily.get("time", [])
    if not times or offset >= len(times):
        return f"I can't forecast that far ahead for {nice}."

    def at(key):
        vals = daily.get(key) or []
        return vals[offset] if offset < len(vals) else None

    try:
        target = _dt.date.fromisoformat(times[offset])
    except (ValueError, TypeError):
        target = _dt.date.today() + _dt.timedelta(days=offset)
    label = _day_label(offset, target)
    when = label if offset <= 1 else f"on {label}"
    desc = _WMO.get(int(at("weather_code") or -1), "unclear conditions")
    tmax = _round_num(at("temperature_2m_max"))
    tmin = _round_num(at("temperature_2m_min"))
    feels = _round_num(at("apparent_temperature_max"))
    precip = _round_num(at("precipitation_probability_max"))
    wind = _round_num(at("wind_speed_10m_max"))

    # Push forecast data to the HUD as a notification.
    _push_weather_notification(
        f"[ FORECAST ] {nice} — {when}\n"
        f"{desc.upper()}\n"
        f"{tmin}°C — {tmax}°C  (feels {feels}°C)\n"
        f"Precip {precip}%  |  Wind {wind} km/h"
    )

    return (
        f"Weather forecast for {nice} {when} — "
        f"conditions: {desc}; "
        f"high: {tmax} degrees Celsius; "
        f"low: {tmin} degrees; "
        f"feels like up to: {feels} degrees; "
        f"chance of precipitation: {precip} percent; "
        f"wind up to: {wind} kilometers per hour. "
        f"Tell the user the place, the day, the high and low temperatures, and "
        f"the conditions, stating these exact numbers."
    )


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


class _HomeControl:
    """Controls smart-home lights/scenes via one of two backends.

    * ``shortcuts`` (macOS only): runs a named macOS Shortcut (containing Home
      actions) via the ``shortcuts`` CLI. Apple-only.
    * ``webhook`` (any OS — Windows/Linux/Mac): sends an HTTP request. This is
      how it works off a Mac — point each command at a webhook that triggers the
      action, e.g. Home Assistant's REST API, or an iPhone automation service
      like Pushcut that runs your Home shortcut on the phone.

    Config maps a spoken intent -> a Shortcut name (shortcuts backend) or a
    webhook spec (webhook backend).
    """

    def __init__(self, home_cfg: Any = None) -> None:
        self.enabled = bool(getattr(home_cfg, "enabled", False)) if home_cfg else False
        backend = (getattr(home_cfg, "backend", "") or "").lower().strip()
        if not backend:
            backend = "shortcuts" if _platform.system() == "Darwin" else "webhook"
        self.backend = backend
        self.shortcuts = {
            str(k).lower().strip(): v
            for k, v in (getattr(home_cfg, "shortcuts", None) or {}).items()
        }
        self.webhooks = {
            str(k).lower().strip(): v
            for k, v in (getattr(home_cfg, "webhooks", None) or {}).items()
        }

    @property
    def _commands(self) -> dict:
        return self.shortcuts if self.backend == "shortcuts" else self.webhooks

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self._commands)

    def commands_text(self) -> str:
        return "; ".join(sorted(self._commands.keys()))

    def _match_key(self, command: str):
        c = (command or "").lower().strip()
        cmds = self._commands
        if not c:
            return None
        if c in cmds:
            return c
        ctoks = set(_re.findall(r"\w+", c))
        best, best_score = None, 0.0
        for key in cmds:
            ktoks = set(_re.findall(r"\w+", key))
            score = float(len(ctoks & ktoks))
            if c in key or key in c:
                score += 1.5
            if score > best_score:
                best, best_score = key, score
        return best if best_score > 0 else None

    def run(self, command: str = "", **_: Any) -> str:
        key = self._match_key(command)
        if key is None:
            return ("No matching light command. Available commands: "
                    + self.commands_text() + ".")
        if self.backend == "shortcuts":
            return self._run_shortcut(str(self.shortcuts[key]), command)
        return self._run_webhook(self.webhooks[key], command)

    @staticmethod
    def _ok(command: str) -> str:
        return (f"Done — the '{command}' action was carried out successfully. "
                "Confirm to the user that it is done, in your own voice. Do not "
                "mention shortcuts, tools, webhooks, or how it was done.")

    def _run_shortcut(self, name: str, command: str) -> str:
        if _shutil.which("shortcuts") is None:
            return ("Apple Home control via Shortcuts needs macOS. On this machine "
                    "use the 'webhook' backend instead.")
        try:
            proc = _subprocess.run(
                ["shortcuts", "run", name],
                capture_output=True, text=True, timeout=20,
            )
        except FileNotFoundError:
            return "The Shortcuts command isn't available on this machine."
        except _subprocess.TimeoutExpired:
            return f"The '{name}' action timed out."
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            return (f"I couldn't carry out '{command}'. {err}".strip()
                    + f" (Check that a Shortcut named '{name}' exists.)")
        return self._ok(command)

    def _run_webhook(self, spec: Any, command: str) -> str:
        if requests is None:
            return "The web request library is unavailable, so I can't reach the home hub."
        # spec may be a bare URL string, or a dict with url/method/headers/body/json.
        if isinstance(spec, str):
            spec = {"url": spec}
        if not isinstance(spec, dict) or not spec.get("url"):
            return f"The '{command}' command is misconfigured (no URL)."
        method = str(spec.get("method", "POST")).upper()
        url = spec["url"]
        headers = spec.get("headers") or {}
        kwargs: dict[str, Any] = {"headers": headers, "timeout": 15}
        if "json" in spec and spec["json"] is not None:
            kwargs["json"] = spec["json"]
        elif spec.get("body") is not None:
            kwargs["data"] = spec["body"]
        try:
            resp = requests.request(method, url, **kwargs)
            resp.raise_for_status()
        except Exception as exc:
            log.error("Home webhook failed: %s", exc)
            return (f"I couldn't reach the home system to '{command}'. "
                    "The hub may be offline.")
        return self._ok(command)


class LocalToolRegistry:
    """A tiny registry of in-process tools matching the MCP tool interface."""

    def __init__(self, home_cfg: Any = None) -> None:
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
            description="Get the weather for a city or place. By default returns "
                        "today's current conditions; set 'day' for an upcoming "
                        "day's forecast. Use this for any question about weather, "
                        "temperature, or conditions.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and optionally country, e.g. 'Stockholm' or 'Paris, France'.",
                    },
                    "day": {
                        "type": "string",
                        "description": "Optional. The day to forecast, e.g. 'today' "
                                       "(default), 'tomorrow', 'Friday', 'in 3 days', "
                                       "or a date like '2026-06-13'. Omit for today.",
                    },
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

        # Smart-home light/scene control (macOS Shortcuts or a cross-platform
        # webhook) — only registered when enabled and at least one command maps.
        home = _HomeControl(home_cfg)
        if home.configured:
            self._register(
                name="control_lights",
                description=(
                    "Control the user's smart-home lights and scenes (turn lights "
                    "on or off, dim or brighten them, or set a scene). Pass the "
                    "closest matching command from this list: "
                    + home.commands_text() + "."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The light/scene command to run, e.g. "
                                           "'turn the lights on'. Choose the closest "
                                           "of the available commands.",
                        },
                    },
                    "required": ["command"],
                },
                func=home.run,
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

