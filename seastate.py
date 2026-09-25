"""The sea along a passage: wave trains, their periods and directions, and what they mean aboard.

Read at the boat's median position along its track (from the ensemble passages) every hour:
  ECMWF WAM      combined sea: height, mean period, peak period, direction
  GFS-Wave       combined sea, plus wind sea, primary swell and secondary swell separately

For each wave train:
  wavelength     L = g T^2 / 2 pi (deep water), about 1.56 T^2 m
  steepness      H / L, given as "1 in N"
  wave speed     c = g T / 2 pi
  relative side  where it comes from relative to the heading (bow, forward quarter, aft quarter, astern)
  encounter      how often the boat meets it (comfort.encounter_period)

Rules of thumb used in the discussion (not calibrated against a logged passage):
  steepness      gentle flatter than 1 in 30, moderate to 1 in 18, steep to 1 in 12, very steep beyond
                 (a wave breaks near 1 in 7)
  cross sea      wind sea and swell both 1 m or more, arriving 60 deg or more apart, and the swell shorter
                 than 11 s (a long swell from another angle adds a slow roll, not a confused sea)
  seas slowing   a wave train 1.5 m or more, moderate or steeper, on the bow or forward quarter
"""
import json
import math
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np

import comfort
import route

G = 9.81
MS_TO_KT = 1.943844
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
SIDE_WORDS = {"bow": "on the bow", "fwd_quarter": "on the forward quarter", "aft_quarter": "on the aft quarter",
              "astern": "from astern"}
TRAINS = {"wind": "wind sea", "swell": "swell", "swell2": "second swell"}


def compass(deg):
    return COMPASS[int((deg % 360) / 22.5 + 0.5) % 16]


def wavelength(period_s):
    return G * period_s ** 2 / (2 * math.pi)


def steepness_n(height_m, period_s):
    """N in 'one in N': wavelength over height."""
    return wavelength(period_s) / height_m if height_m > 0 else math.inf


def steepness_words(n):
    if n >= 30:
        return "gentle"
    if n >= 18:
        return "moderate"
    if n >= 12:
        return "steep"
    return "very steep"


CROSS_SWELL_MAX_PERIOD = 11


def angle_apart(a, b):
    return abs((a - b + 180) % 360 - 180)


def cross_sea(wind, swell):
    """Wind sea and swell both 1 m or more, 60 deg or more apart, and the swell short enough to matter."""
    if not wind or not swell or wind["h"] < 1.0 or swell["h"] < 1.0:
        return False
    return angle_apart(wind["from"], swell["from"]) >= 60 and swell["t"] < CROSS_SWELL_MAX_PERIOD


# ------------------------------------------------------------------ data

def load(run_dir):
    """Hourly arrays [time, point] per field, or None where a field wasn't fetched."""
    out = {}
    for model, fields in (("ecmwf_wam025", ("wave_height", "wave_period", "wave_peak_period", "wave_direction")),
                          ("ncep_gfswave025", ("wave_height", "wave_period", "wave_direction",
                                               "wind_wave_height", "wind_wave_period", "wind_wave_direction",
                                               "swell_wave_height", "swell_wave_period", "swell_wave_direction",
                                               "secondary_swell_wave_height", "secondary_swell_wave_period",
                                               "secondary_swell_wave_direction"))):
        path = Path(run_dir) / f"{model}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        d = d if isinstance(d, list) else [d]
        from datetime import datetime
        times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in d[0]["hourly"]["time"]]
        arrs = {}
        for f in fields:
            if f in d[0]["hourly"]:
                arrs[f] = np.array([[np.nan if x is None else x for x in p["hourly"][f]] for p in d], dtype=float).T
        out[model] = dict(times=times, **arrs)
    return out


def _at(model, field, t, p):
    if model is None or field not in model:
        return math.nan
    i = int((t - model["times"][0]).total_seconds() // 3600)
    if i < 0 or i >= model[field].shape[0]:
        return math.nan
    return float(model[field][i, p])


def _train(h, t_s, frm, heading, boat_kt):
    if any(map(math.isnan, (h, t_s, frm))) or h <= 0.05 or t_s <= 0:
        return None
    rel = comfort.relative_sea(frm, heading)
    return dict(h=h, t=t_s, **{"from": frm}, rel=rel, side=comfort.sea_side(rel),
                enc=comfort.encounter_period(t_s, boat_kt, rel), n=steepness_n(h, t_s),
                speed_kt=G * t_s / (2 * math.pi) * MS_TO_KT)


def along_track(seas, dist_at, depart_utc, arrive_utc, step_h=1):
    """One row per step at the boat's median position."""
    rows = []
    idx_nm = [p[0] for p in route.sample_points()]
    t = depart_utc + timedelta(hours=route.EXIT_HOURS)
    ec, gf = seas.get("ecmwf_wam025"), seas.get("ncep_gfswave025")
    while t <= arrive_utc:
        nm = dist_at(t)
        boat_kt = max((dist_at(t + timedelta(hours=1)) - nm), 3.0) if nm < route.total_nm() else 5.0
        _, _, heading = route.position_at(nm)
        p = int(np.argmin([abs(nm - x) for x in idx_nm]))
        row = dict(t=t, nm=nm, heading=heading, boat_kt=boat_kt)
        row["ecmwf"] = _train(_at(ec, "wave_height", t, p), _at(ec, "wave_period", t, p),
                              _at(ec, "wave_direction", t, p), heading, boat_kt)
        if row["ecmwf"]:
            row["ecmwf"]["tp"] = _at(ec, "wave_peak_period", t, p)
        row["gfs"] = _train(_at(gf, "wave_height", t, p), _at(gf, "wave_period", t, p),
                            _at(gf, "wave_direction", t, p), heading, boat_kt)
        row["wind"] = _train(_at(gf, "wind_wave_height", t, p), _at(gf, "wind_wave_period", t, p),
                             _at(gf, "wind_wave_direction", t, p), heading, boat_kt)
        row["swell"] = _train(_at(gf, "swell_wave_height", t, p), _at(gf, "swell_wave_period", t, p),
                              _at(gf, "swell_wave_direction", t, p), heading, boat_kt)
        row["swell2"] = _train(_at(gf, "secondary_swell_wave_height", t, p), _at(gf, "secondary_swell_wave_period", t, p),
                               _at(gf, "secondary_swell_wave_direction", t, p), heading, boat_kt)
        row["cross"] = cross_sea(row["wind"], row["swell"])
        rows.append(row)
        t += timedelta(hours=step_h)
    return rows


# ------------------------------------------------------------------ summaries

def _circular_mean(degs):
    s = sum(math.sin(math.radians(d)) for d in degs)
    c = sum(math.cos(math.radians(d)) for d in degs)
    return (math.degrees(math.atan2(s, c)) + 360) % 360


def train_summary(rows, key):
    xs = [r[key] for r in rows if r.get(key)]
    if not xs:
        return None
    sides = [x["side"] for x in xs]
    side = max(set(sides), key=sides.count)
    return dict(h_min=min(x["h"] for x in xs), h_max=max(x["h"] for x in xs),
                t_min=min(x["t"] for x in xs), t_max=max(x["t"] for x in xs),
                t_med=float(np.median([x["t"] for x in xs])),
                tp_med=float(np.nanmedian([x.get("tp", math.nan) for x in xs])) if key == "ecmwf" else math.nan,
                frm=_circular_mean([x["from"] for x in xs]), side=side, side_share=sides.count(side) / len(sides),
                enc_med=float(np.median([x["enc"] for x in xs])), n_min=min(x["n"] for x in xs),
                speed_kt=float(np.median([x["speed_kt"] for x in xs])), hours=len(xs))


def summarize(rows):
    out = {k: train_summary(rows, k) for k in ("ecmwf", "gfs", "wind", "swell", "swell2")}
    out["cross_h"] = sum(1 for r in rows if r["cross"])
    out["hours"] = len(rows)
    return out


def _range(lo, hi, fmt, unit):
    a, b = format(lo, fmt), format(hi, fmt)
    return f"{a} {unit}" if a == b else f"{a} to {b} {unit}"


SIDE_SHORT = {"bow": "on the bow", "fwd_quarter": "forward quarter", "aft_quarter": "aft quarter", "astern": "from astern"}


def _train_text(x, name):
    lo, hi = format(x["t_min"], ".0f"), format(x["t_max"], ".0f")
    period = f"{lo} s" if lo == hi else f"{lo}-{hi} s"
    side = SIDE_SHORT[x["side"]].replace("from astern", "astern")
    return f"{name} to {x['h_max']:.1f} m at {period} from {compass(x['frm'])}, {side}"


def verdict_line(s):
    """A few words on what the sea means aboard."""
    sw, wi = s.get("swell"), s.get("wind")
    slowing = [x for x in (sw, wi) if x and x["side"] in ("bow", "fwd_quarter") and x["h_max"] >= 1.5 and x["n_min"] < 30]
    motion = []
    if sw:
        motion.append("slow roll" if sw["t_med"] >= 11 else "regular rise and fall")
    if wi:
        motion.append({"astern": "following wind sea", "aft_quarter": "some yaw from the quarter",
                       "fwd_quarter": "quick corkscrew", "bow": "pitching and slamming"}[wi["side"]])
    words = ", ".join(motion) if motion else "easy motion"
    return f"{words[0].upper() + words[1:]}; {'expect some speed lost' if slowing else 'little speed lost'}."


def flags(s):
    out = []
    if s.get("cross_h", 0) >= 3:
        out.append(f"Cross sea about {s['cross_h']} h")
    e, g = s.get("ecmwf"), s.get("gfs")
    if e and g and abs(e["h_max"] - g["h_max"]) >= 0.8:
        out.append(f"models differ on height (ECMWF {e['h_max']:.1f} m, GFS {g['h_max']:.1f} m)")
    wi = s.get("wind")
    if wi and wi["n_min"] < 18:
        out.append(f"wind sea {steepness_words(wi['n_min'])} at 1 in {wi['n_min']:.0f}")
    return (out[0][0].upper() + out[0][1:] + ("; " + "; ".join(out[1:]) if out[1:] else "") + ".") if out else ""


def describe(s, short=False):
    """One or two short sentences: the wave trains, what they mean, and any flags."""
    sw, wi = s.get("swell"), s.get("wind")
    trains = [t for t in ((_train_text(sw, "Swell") if sw else None),
                          (_train_text(wi, "wind sea" if sw else "Wind sea") if wi else None)) if t]
    if not trains:
        e = s.get("ecmwf") or s.get("gfs")
        if not e:
            return ""
        return f"Seas {_range(e['h_min'], e['h_max'], '.1f', 'm')} {e['t_med']:.0f} s {compass(e['frm'])}."
    text = "; ".join(trains) + ". " + verdict_line(s)
    f = flags(s)
    return text + (" " + f if f else "")


def by_day(rows, depart_utc):
    days = []
    k = 0
    while True:
        t0, t1 = depart_utc + timedelta(days=k), depart_utc + timedelta(days=k + 1)
        chunk = [r for r in rows if t0 <= r["t"] < t1]
        if not chunk:
            break
        days.append(dict(day=k + 1, start=t0, rows=chunk, summary=summarize(chunk)))
        k += 1
    return days
