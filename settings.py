"""Settings the GUI owns, stored in settings.json. Defaults match what the code used before it existed.

Anything here can be changed from the Settings page: the limits, the motor figures, which polar files
make the conservative envelope and its cruise factor, the performance polar and its adjustments, and
whether the Home Assistant pages and alerts are used at all (off by default in the standalone app).
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
FILE = HERE / "settings.json"

DEFAULTS = {
    "limits": {
        "tws_warn": 25.0, "tws_no": 30.0,      # sustained true wind, kt
        "gust_warn": 35.0, "gust_no": 40.0,    # kt, estimated from the sustained wind
        "hs": 3.0,                             # m, over this forward of the beam is a warning
        "hs_no": 4.0,                          # m, over this from any direction is a no-go
        # The feet rule, judged on the dominant swell's period: period in seconds per foot of
        # wave height. 3:1 or better and there is nothing to think about; under 2:1 is worth
        # thinking hard about; 1:1 is a square sea, and a reason to stay in.
        # Between 2:1 and 3:1 it only warns once the seas are big, because a small steep chop
        # is not a reason to stay in port.
        "feet_clear": 3.0,
        "feet_warn": 2.0,
        "feet_no": 1.0,
        "feet_min_hs": 1.5,      # below this height the ratio is ignored: it is chop, not a hazard
        "feet_big_hs": 2.5,      # at or above this, the 2:1 to 3:1 band warns too
        # Steepness, the physical cross-check forecasters use: wavelength over height, as 1 in N.
        # Waves break near 1 in 7; 1 in 20 is the critical limit quoted for small craft, and
        # 1 in 40 is a steep sea. Smaller N is steeper. This catches the short steep chop that
        # sits under the feet rule's height floor.
        "steep_warn_n": 40,
        "steep_no_n": 20,
        "steep_min_hs": 1.0,
    },
    "tier_share": 0.10,                        # share of ensemble passages before a tier counts
    "gust_factor": 1.23,                       # WMO at-sea 3-second gust against the 10-minute mean
    "motor": {"below_kt": 4.0, "speed_kt": 5.5, "fuel_gph": 1.0},
    "polar": {
        "sources": ["sabado.pol", "sabado.csv", "sabado_polar_v2.csv", "sabado(1).csv"],
        "cruise_factor": 0.90,
        "performance_file": "lagoon42_performance.txt",
        "performance": {"upwind": 0.90, "downwind": 0.85, "night": 0.75},
    },
    # A report says how old its data is, and refuses to build on data older than these.
    # The limits allow for the lag between a model cycle and its data being published:
    # ECMWF's ensemble runs 00/12 UTC and lands about eight hours later, GFS-Wave runs four
    # times a day, and the SMOC currents once.
    "freshness": {
        "max_fetch_age_h": 12,
        "max_run_age_h": {"default": 24, "ecmwf_ifs025": 26, "gfs_seamless": 20,
                          "ecmwf_wam025": 20, "ncep_gfswave025": 20, "currents": 48},
    },
    "boat": {"name": "", "kind": ""},        # shown on the pages, e.g. "S/V Sabado", "Lagoon 42"
    "trip_defaults": {"days": 4, "exit_hours": 1.0},
    # Optional: copy the pages to a Home Assistant box and send alerts through it.
    # Off by default; fill these in on the Settings page if you use HA.
    "home_assistant": {
        "enabled": False,
        "ssh": "root@homeassistant.local",       # an SSH login that can write to /homeassistant/www
        "www_dir": "/homeassistant/www/passage",
        "url": "http://homeassistant.local:8123/local/passage",
        "notify_service": "",                    # e.g. mobile_app_your_phone, blank sends no alerts
    },
}


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


_cache = None


def get():
    global _cache
    if _cache is None:
        saved = json.loads(FILE.read_text()) if FILE.exists() else {}
        _cache = _merge(DEFAULTS, saved)
    return _cache


def save(values):
    """Store a full settings dict (merged over the defaults) and forget the cache."""
    global _cache
    FILE.write_text(json.dumps(_merge(DEFAULTS, values), indent=1))
    _cache = None
    return get()


def reload():
    global _cache
    _cache = None
    return get()


def polar_files(kind="sources"):
    """Polar files that exist, as paths."""
    s = get()["polar"]
    names = s["sources"] if kind == "sources" else [s["performance_file"]]
    return [HERE / n for n in names if (HERE / n).exists()]


def available_polars():
    """Every polar-looking file in the project folder, for the Settings page to offer."""
    return sorted(p.name for p in HERE.iterdir()
                  if p.suffix.lower() in (".pol", ".csv", ".txt") and p.name != "settings.json"
                  and not p.name.startswith("test"))
