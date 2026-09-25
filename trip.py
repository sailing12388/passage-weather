"""Trips: a passage from one named place to another, and what stage its planning is at.

A trip lives in trips/<slug>/trip.json:
  origin, dest    name, lat, lon, IANA time zone
  earliest        first local date we could leave (origin time zone)
  days            how many days from earliest to screen, one departure per day at local noon
  status          planning -> watching -> done
  watch_depart    chosen departure (origin local time) once watching
  exit_hours      time from the dock to open water, added to every ETA

Routes are a great circle between the two places. Estimated guide only. The captain develops the detailed route for every passage.
"""
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

HERE = Path(__file__).parent
TRIPS = HERE / "trips"
PLACES_FILE = HERE / "places.json"

def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_places():
    """Only places saved on purpose with `passage.py place add`. Nothing is prefilled."""
    return json.loads(PLACES_FILE.read_text()) if PLACES_FILE.exists() else {}


def save_place(key, rec):
    saved = json.loads(PLACES_FILE.read_text()) if PLACES_FILE.exists() else {}
    saved[key] = rec
    PLACES_FILE.write_text(json.dumps(saved, indent=1, sort_keys=True))


def tz_for_position(lat, lon):
    """IANA time zone for a position, from Open-Meteo's timezone=auto."""
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon, "timezone": "auto", "current": "pressure_msl"}, timeout=30).json()
    tz = r.get("timezone")
    if not tz or tz == "GMT":
        raise ValueError(f"no time zone found for {lat}, {lon}")
    return tz


NUM = r"[-+]?\d+(?:\.\d+)?"
HEMI = r"[NnSsEeWw]"
# 19.5277 S | 19° 31.66' S | S 19 31 40 | -19.5277
PART = rf"(?:({HEMI})\s*)?({NUM})\s*(?:°|º|d|deg)?\s*(?:({NUM})\s*(?:'|′|m|min)?\s*(?:({NUM})\s*(?:\"|″|s|sec)?)?)?\s*(?:({HEMI}))?"
POSITION = re.compile(rf"^\s*{PART}\s*(?:[,;/]\s*|\s+){PART}\s*$")


def _one(pre, deg, minutes, seconds, post):
    hemi = (pre or post or "").upper()
    value = abs(float(deg)) + (float(minutes or 0) / 60) + (float(seconds or 0) / 3600)
    if float(deg) < 0:
        value = -value
    if hemi in ("S", "W"):
        value = -abs(value)
    elif hemi in ("N", "E"):
        value = abs(value)
    return value, hemi


def parse_position(text):
    """A lat/lon pair in the usual forms: '-19.53,169.50', '19.5277° S, 169.4959° E',
    '19 31.66 S 169 29.75 E', 'S19°31'40" E169°29'45"'. Returns (lat, lon) or None."""
    m = POSITION.match(text.replace("’", "'").replace("“", '"').replace("”", '"'))
    if not m:
        return None
    a, a_h = _one(*m.group(1, 2, 3, 4, 5))
    b, b_h = _one(*m.group(6, 7, 8, 9, 10))
    if a_h in ("E", "W") or b_h in ("N", "S"):      # written longitude first
        a, b, a_h, b_h = b, a, b_h, a_h
    if not (-90 <= a <= 90 and -180 <= b <= 360):
        return None
    return a, b


def resolve_place(text, country=None, tz=None):
    """A place from a position, the places file, or the geocoder. 'Name, Country' or 'Name, CC' narrows
    the search. Returns (record, note); the note says where the position came from."""
    text = text.strip()
    pos = parse_position(text)
    if pos:
        lat, lon = pos
        tz = tz or tz_for_position(lat, lon)
        return ({"name": f"{abs(lat):.3f}{'S' if lat < 0 else 'N'} {abs(lon):.3f}{'W' if lon < 0 else 'E'}",
                 "lat": round(lat, 4), "lon": round(lon, 4), "tz": tz, "source": "typed position"},
                f"typed position {lat:.3f}, {lon:.3f}, time zone {tz}")
    key = text.lower()
    places = load_places()
    if key in places:
        rec = dict(places[key])
        if tz:
            rec["tz"] = tz
        return rec, f"places file ({rec.get('source', 'saved')})"
    cm = re.fullmatch(r"\s*(.+?)\s*,\s*([A-Za-z][A-Za-z .'-]+)\s*", text)
    if cm and not country:
        text, country = cm.group(1), cm.group(2)      # "Port Resolution, Vanuatu" or "Nadi, FJ"
        if text.lower() in places:                    # a saved place written with its country
            rec = dict(places[text.lower()])
            if tz:
                rec["tz"] = tz
            return rec, f"places file ({rec.get('source', 'saved')})"
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": text, "count": 20, "language": "en"}, timeout=30).json()
    results = r.get("results") or []
    if country:
        c = country.strip().lower()
        results = [x for x in results if c in (x.get("country_code", "").lower(), x.get("country", "").lower())]
    if not results:
        raise ValueError(f"no match for '{text}'" + (f" in {country}" if country else "")
                         + ". Type a position instead, like \"19.53 S, 169.50 E\"")

    def rank(x):   # prefer the Pacific, then capitals and larger towns
        return (not str(x.get("timezone", "")).startswith("Pacific/"),
                x.get("feature_code", "") != "PPLC",
                -(x.get("population") or 0))
    best = sorted(results, key=rank)[0]
    rec = {"name": best["name"], "lat": round(best["latitude"], 3), "lon": round(best["longitude"], 3),
           "tz": tz or best.get("timezone"), "source": f"Open-Meteo geocoder: {best['name']}, "
           f"{best.get('admin1') or ''} {best.get('country', '')} ({best.get('feature_code', '')})".replace("  ", " ")}
    others = [f"{x['name']}, {x.get('country', '')} ({x['latitude']:.2f}, {x['longitude']:.2f})"
              for x in sorted(results, key=rank)[1:4]]
    note = ("GUESS, check it: " + rec["source"] + f" at {rec['lat']}, {rec['lon']}"
            + (f". Other matches: {'; '.join(others)}" if others else "")
            + ". Save a correct position with: passage.py place add")
    return rec, note


@dataclass
class Trip:
    slug: str
    origin: dict
    dest: dict
    earliest: str                 # YYYY-MM-DD, origin local
    days: int = 4
    status: str = "planning"
    watch_depart: str | None = None   # YYYY-MM-DD HH:MM, origin local (noon unless set by hand)
    exit_hours: float = 1.0
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"))
    page_url: str | None = None       # published options page
    brief_url: str | None = None      # published full brief
    last_alert: str | None = None     # signature of the last alert sent through Home Assistant
    last_headline: str | None = None
    notes: list = field(default_factory=list)

    # ------------------------------------------------------------ storage
    @property
    def dir(self):
        return TRIPS / self.slug

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "trip.json").write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, slug):
        return cls(**json.loads((TRIPS / slug / "trip.json").read_text()))

    @classmethod
    def all(cls):
        return [cls.load(p.parent.name) for p in sorted(TRIPS.glob("*/trip.json"))]

    # ------------------------------------------------------------ time
    @property
    def origin_tz(self):
        return ZoneInfo(self.origin["tz"])

    @property
    def dest_tz(self):
        return ZoneInfo(self.dest["tz"])

    DEPART_HOUR = 12      # one departure per day, at local noon

    def window(self):
        """First and last departures screened (local noon each day), as aware origin-local datetimes."""
        d0 = date.fromisoformat(self.earliest)
        first = datetime(d0.year, d0.month, d0.day, self.DEPART_HOUR, 0, tzinfo=self.origin_tz)
        return first, first + timedelta(days=self.days - 1)

    def watch_time(self):
        if not self.watch_depart:
            return None
        return datetime.strptime(self.watch_depart, "%Y-%m-%d %H:%M").replace(tzinfo=self.origin_tz)

    def title(self):
        return f"{self.origin['name']} to {self.dest['name']}"

    # ------------------------------------------------------------ route
    def route_key(self):
        return f"trip_{self.slug}"

    def use_route(self):
        import route
        route.define(self.route_key(), [(self.origin["name"], self.origin["lat"], self.origin["lon"]),
                                        (self.dest["name"], self.dest["lat"], self.dest["lon"])])
        route.use(self.route_key())
        route.EXIT_HOURS = self.exit_hours
        return route


def tz_abbrev(dt, tz):
    """Short zone label like FJT or VUT when the zone database has one, else UTC+11."""
    name = dt.astimezone(tz).tzname() or ""
    if re.fullmatch(r"[+-]\d+", name):
        off = dt.astimezone(tz).utcoffset()
        hours = off.total_seconds() / 3600
        return f"UTC{hours:+g}"
    return name
