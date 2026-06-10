"""Built-in ('local') tools that run in-process, alongside MCP tools.

The generic MCP ``fetch`` server can only download a URL you hand it — it can't
*search* — so a small model trying to answer "what's the weather?" just invents a
URL and fails. These local tools give Mr. House real, reliable capabilities with
no API keys required.

Currently:
  * ``get_weather`` — current conditions for a place, via the free Open-Meteo API.
  * ``get_time``    — current local date/time.

Each tool exposes an OpenAI/Ollama function schema and a plain-text result, so it
plugs into the same tool-calling loop as the MCP tools.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Callable

log = logging.getLogger(__name__)

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


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

