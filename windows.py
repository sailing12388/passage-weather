"""Score departure times Nadi to Port Resolution against the latest forecast.

Every ensemble member is sailed along the planning route with the conservative
polar. Waves are deterministic (no wave ensemble had data), so each member's
track is checked against both ECMWF WAM and GFS-Wave for height and for period
against height. A wave model flags a departure when more than 10% of members
meet a breach on their timing.

Wave period is what Open-Meteo serves as wave_period, checked against raw GRIB:
ECMWF WAM mean period (mwp) and GFS-Wave primary wave period (PERPW).

Usage: python3 windows.py [data/<fetch dir>]   (defaults to the newest Port Resolution fetch)
"""
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import comfort
import polar
import route
import settings

HERE = Path(__file__).parent
FJT = timezone(timedelta(hours=12))
VUT = timezone(timedelta(hours=11))
FIRST_DEPARTURE = datetime(2026, 9, 22, 0, 0, tzinfo=FJT)
DEPARTURE_STEP_H = 3
STEP_H = 0.25

# Limits from settings.json (the Settings page edits them). Wind and gusts are tiers:
# below warn is GO, warn to no is WARNING, above no is NO. Waves depend on period and direction too.
LIMITS = dict(settings.get()["limits"], motion_no=comfort.MOTION_BANDS[3])
TIER_SHARE = settings.get()["tier_share"]       # share of ensemble passages before a tier counts
# Gusts are estimated from the sustained wind, not taken from the models (model gust fields ran high:
# see the note below). ECMWF's gust field is the highest in each 3 hours (median 1.38x sustained on this
# route) and GEFS's is instantaneous (1.08x). WMO/TD-1555 (Harper, Kepert and Ginger 2010), Table 1.1,
# at-sea exposure: a 3-second gust is 1.23x the 10-minute mean wind.
GUST_FACTOR = settings.get()["gust_factor"]
WAVE_FLAG_SHARE = 0.10  # share of members meeting a wave breach before a wave model flags
MOTOR_BELOW_KT = settings.get()["motor"]["below_kt"]    # motor when sailing drops under this
MOTOR_KT = settings.get()["motor"]["speed_kt"]          # speed under power
FUEL_GPH = settings.get()["motor"]["fuel_gph"]          # US gallons an hour
DAYLIGHT_ENTRY = (1.0, 1.5)  # enter no earlier than sunrise+1 h, no later than sunset-1.5 h
WIND_MODELS = {"ecmwf_ifs025": "ECMWF", "gfs_seamless": "GEFS"}
# Points of sail, by the angle the boat actually sails to the true wind (Lagoon 42 manual's reefing bands)
# upwind under 75, beam reach 75-110, broad reach 110-150, running over 150
POINTS_OF_SAIL = (("up", 75), ("beam", 110), ("broad", 150), ("run", 181))
TACK_ANGLE = 60             # the conservative polar starts at 60 TWA, so a beat is sailed there
WAVE_MODELS = {"ecmwf_wam025": "ECMWF", "ncep_gfswave025": "GFS"}
MIN_COMPLETE = 0.8      # a departure is scored only if this share of members arrive inside the forecast


def load(run_dir):
    def members(path, var):
        d = json.loads((run_dir / path).read_text())
        times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in d[0]["hourly"]["time"]]
        keys = sorted(k for k in d[0]["hourly"] if k == var or k.startswith(var + "_member"))
        arr = np.array([[[np.nan if x is None else x for x in pt["hourly"][k]] for pt in d] for k in keys],
                       dtype=float)                      # [member, point, time]
        return times, arr.transpose(0, 2, 1)             # [member, time, point]

    wind = {}
    for model in WIND_MODELS:
        t, spd = members(f"{model}.json", "wind_speed_10m")
        _, wdir = members(f"{model}.json", "wind_direction_10m")
        _, gust = members(f"{model}.json", "wind_gusts_10m")
        wind[model] = dict(times=t, spd=spd, dir=wdir, gust=gust)
    waves = {}
    for model in WAVE_MODELS:
        t, h = members(f"{model}.json", "wave_height")
        _, per = members(f"{model}.json", "wave_period")
        _, wdir = members(f"{model}.json", "wave_direction")    # direction the waves come from
        waves[model] = dict(times=t, hs=h[0], period=per[0], dir=wdir[0])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    waves["_currents"] = load_currents(run_dir)
    return wind, waves, manifest


def load_currents(run_dir):
    """Surface current east/north components in knots, [time, point], or None if not fetched.
    Open-Meteo gives velocity in km/h and the direction the current flows towards."""
    path = run_dir / "currents.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    d = d if isinstance(d, list) else [d]
    times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in d[0]["hourly"]["time"]]
    spd = np.array([[np.nan if x is None else x / 1.852 for x in p["hourly"]["ocean_current_velocity"]] for p in d]).T
    towards = np.radians(np.array([[np.nan if x is None else x for x in p["hourly"]["ocean_current_direction"]]
                                   for p in d]).T)
    east, north = spd * np.sin(towards), spd * np.cos(towards)
    # hold the last forecast value past the end of the current forecast; currents change slowly
    for arr in (east, north):
        for k in range(arr.shape[1]):
            col = arr[:, k]
            ok = np.nonzero(np.isfinite(col))[0]
            if ok.size:
                col[ok[-1] + 1:] = col[ok[-1]]
                col[:ok[0]] = col[ok[0]]
            else:
                col[:] = 0.0
    return dict(times=times, east=east, north=north)


def current_effect(stw, course, east, north):
    """Speed over ground along the course when steering to hold the line through a current.
    The boat crabs to cancel the cross-track set, then the along-track part adds or subtracts."""
    c = math.radians(course)
    along = east * math.sin(c) + north * math.cos(c)
    cross = -east * math.cos(c) + north * math.sin(c)
    return math.sqrt(max(stw * stw - cross * cross, 0.0)) + along, along


def route_table():
    """Per nautical mile: course and index of the nearest forecast sample point."""
    pts = route.sample_points()
    total = route.total_nm()
    nm = np.arange(0, math.ceil(total) + 1)
    crs = np.array([route.position_at(min(x, total))[2] for x in nm])
    idx = np.array([int(np.argmin([abs(x - p[0]) for p in pts])) for x in nm])
    return total, crs, idx


def motor_speed(tws, course_twa):
    """MOTOR_KT, reduced into a strong headwind as in a motoring polar (4 kt at 16-20 kt, none above)."""
    if course_twa < 60 and tws > 14:
        return 4.0 if tws <= 20 else 0.0
    return MOTOR_KT


def sun_times(day, lat, lon):
    """Sunrise and sunset (UTC datetimes) for a UTC calendar day, NOAA low-precision formula."""
    n = day.timetuple().tm_yday
    g = 2 * math.pi / 365 * (n - 1)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    ha = math.degrees(math.acos(math.cos(math.radians(90.833)) / (math.cos(math.radians(lat)) * math.cos(decl))
                                - math.tan(math.radians(lat)) * math.tan(decl)))
    midnight = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    rise = midnight + timedelta(minutes=720 - 4 * (lon + ha) - eqt)
    sset = midnight + timedelta(minutes=720 - 4 * (lon - ha) - eqt)
    return rise, sset


def entry_ok(arrive_utc):
    lat, lon = route.WAYPOINTS[-1][1:]
    d0 = arrive_utc.date()
    for d in (d0 - timedelta(days=1), d0, d0 + timedelta(days=1)):
        rise, sset = sun_times(d, lat, lon)
        if rise + timedelta(hours=DAYLIGHT_ENTRY[0]) <= arrive_utc <= sset - timedelta(hours=DAYLIGHT_ENTRY[1]):
            return True
    return False


def is_night(t, lat, lon, _cache={}):
    key = (t.date(), round(lat), round(lon))
    if key not in _cache:
        _cache[key] = [sun_times(t.date() + timedelta(days=k), lat, lon) for k in (-1, 0, 1)]
    return not any(rise <= t <= sset for rise, sset in _cache[key])


def sail(m, depart, wind, waves, vmc, rt, log=None, vmc_night=None):
    """Sail one member from depart. Returns a summary dict, or None if the forecast runs out.
    If log is a list, one dict per time step is appended to it.
    If vmc_night is given it replaces vmc between sunset and sunrise."""
    total, crs_nm, idx_nm = rt
    t0 = wind["times"][0]
    spd, wdir, gust = wind["spd"][m], wind["dir"][m], wind["gust"][m]
    t = depart + timedelta(hours=route.EXIT_HOURS)
    dist = 0.0
    currents = waves.get("_currents")
    waves = {k: v_ for k, v_ in waves.items() if not k.startswith("_")}
    out = dict(max_tws=0.0, max_gust=0.0, beat_h=0.0, motor_h=0.0, comfort_h=[0.0] * len(comfort.LEVELS),
               current_along_nm=0.0,
               **{f"{k}_h": 0.0 for k, _ in POINTS_OF_SAIL}, **{f"sea_{k}_h": 0.0 for k in comfort.SEA_SECTORS},
               **{f"hs_{w}": 0.0 for w in WAVE_MODELS}, **{f"ratio_{w}": math.inf for w in WAVE_MODELS},
               **{f"wave_warn_{w}": False for w in WAVE_MODELS}, **{f"wave_no_{w}": False for w in WAVE_MODELS},
               **{f"wave_why_{w}": set() for w in WAVE_MODELS})
    while dist < total:
        k = min(int(dist), len(crs_nm) - 1)
        p, crs = idx_nm[k], crs_nm[k]
        th = (t - t0).total_seconds() / 3600
        i = int(th)
        if i + 1 >= spd.shape[0]:
            return None
        f = th - i
        s = spd[i, p] * (1 - f) + spd[i + 1, p] * f
        g = s * GUST_FACTOR
        a0, a1 = math.radians(wdir[i, p]), math.radians(wdir[i + 1, p])   # interpolate on the circle
        d = math.degrees(math.atan2(math.sin(a0) * (1 - f) + math.sin(a1) * f,
                                    math.cos(a0) * (1 - f) + math.cos(a1) * f)) % 360
        if any(map(math.isnan, (s, g, d))):
            return None
        course_twa = abs((d - crs + 180) % 360 - 180)   # 0 = wind on the nose
        table = vmc
        if vmc_night is not None:
            lat, lon, _ = route.position_at(dist)
            if is_night(t, lat, lon):
                table = vmc_night
        si = min(s, table.shape[0] - 1.001)
        s0, c = int(si), int(round(course_twa))
        v = table[s0, c] * (1 - (si - s0)) + table[s0 + 1, c] * (si - s0)
        motoring = False
        if v < MOTOR_BELOW_KT:
            vm = motor_speed(s, course_twa)
            if vm > v:
                v, motoring = vm, True
                out["motor_h"] += STEP_H
        if course_twa < 60:
            out["beat_h"] += STEP_H
        tacking = not motoring and course_twa < TACK_ANGLE
        sailing_twa = TACK_ANGLE if tacking else course_twa
        pos = next(k for k, top in POINTS_OF_SAIL if sailing_twa < top)
        out[f"{pos}_h"] += STEP_H
        # the hull's heading and speed: on a beat, both tacks at TACK_ANGLE off the wind
        if tacking:
            headings = [(d + TACK_ANGLE) % 360, (d - TACK_ANGLE) % 360]
            hull_kt = v * math.cos(math.radians(course_twa)) / math.cos(math.radians(TACK_ANGLE))
        else:
            headings, hull_kt = [crs], v
        aws = comfort.apparent_wind(s, sailing_twa, hull_kt)
        sea, accel, rel0 = {}, [], None
        for wm, wv in waves.items():
            wi = int((t - wv["times"][0]).total_seconds() // 3600)
            if wi >= wv["hs"].shape[0]:
                sea[wm] = (np.nan, np.nan, np.nan)
                continue
            hs, per, wdir_from = wv["hs"][wi, p], wv["period"][wi, p], wv["dir"][wi, p]
            sea[wm] = (hs, per, wdir_from)
            if not math.isnan(hs):
                out[f"hs_{wm}"] = max(out[f"hs_{wm}"], hs)
                if hs > 0.1 and not math.isnan(per):
                    out[f"ratio_{wm}"] = min(out[f"ratio_{wm}"], per / hs)
                    if not math.isnan(wdir_from):
                        rels = [comfort.relative_sea(wdir_from, hd) for hd in headings]
                        a_model = float(np.mean([comfort.vertical_accel(hs, per, hull_kt, r) for r in rels]))
                        accel.append(a_model)
                        rel0 = rels[0] if rel0 is None else rel0
                        forward = comfort.sea_side(rels[0]) in ("bow", "fwd_quarter")
                        if hs > LIMITS["hs"] and forward:
                            out[f"wave_warn_{wm}"] = True
                            out[f"wave_why_{wm}"].add(f"over {LIMITS['hs']:.0f} m forward of the beam")
                        if per < LIMITS["period_ratio"] * hs:
                            out[f"wave_warn_{wm}"] = True
                            out[f"wave_why_{wm}"].add("steeper than the feet rule")
                        if hs > LIMITS["hs_no"]:
                            out[f"wave_no_{wm}"] = True
                            out[f"wave_why_{wm}"].add(f"over {LIMITS['hs_no']:.0f} m")
                        if a_model >= LIMITS["motion_no"]:
                            out[f"wave_no_{wm}"] = True
                            out[f"wave_why_{wm}"].add("motion at the Rough band")
        step_accel = max(accel) if accel else 0.0            # the rougher wave model
        step_level = comfort.level(aws, step_accel)
        out["comfort_h"][step_level] += STEP_H
        side = comfort.sea_side(rel0) if rel0 is not None else None
        if side:
            out[f"sea_{side}_h"] += STEP_H
        if log is not None:
            log.append(dict(t=t, dist=dist, tws=s, gust=g, twd=d, course_twa=course_twa, v=v,
                            motoring=motoring, pos=pos, aws=aws, accel=step_accel, comfort=step_level, sea_side=side,
                            **{f"hs_{wm}": sea[wm][0] for wm in waves},
                            **{f"per_{wm}": sea[wm][1] for wm in waves}))
        out["max_tws"] = max(out["max_tws"], s)
        out["max_gust"] = max(out["max_gust"], g)
        sog, along = v, 0.0
        if currents is not None:
            ci = int(np.clip((t - currents["times"][0]).total_seconds() // 3600, 0, currents["east"].shape[0] - 1))
            sog, along = current_effect(v, crs, currents["east"][ci, p], currents["north"][ci, p])
            sog = max(sog, 0.5)
        if log is not None:
            log[-1]["sog"], log[-1]["current_along"] = sog, along
        out["current_along_nm"] += along * STEP_H
        step = sog * STEP_H
        if dist + step >= total and sog > 0:
            t += timedelta(hours=(total - dist) / sog)
            dist = total
        else:
            dist += step
            t += timedelta(hours=STEP_H)
    out["arrive"] = t
    out["hours"] = (t - depart).total_seconds() / 3600
    sea_hours = max(out["hours"] - route.EXIT_HOURS, 1e-6)
    out["current_along_kt"] = out["current_along_nm"] / sea_hours
    sea_h = sum(out[f"{k}_h"] for k, _ in POINTS_OF_SAIL)
    for k, _ in POINTS_OF_SAIL:
        out[f"{k}_share"] = out[f"{k}_h"] / sea_h if sea_h else 0.0
    sides_h = sum(out[f"sea_{k}_h"] for k in comfort.SEA_SECTORS)
    for k in comfort.SEA_SECTORS:
        out[f"sea_{k}_share"] = out[f"sea_{k}_h"] / sides_h if sides_h else 0.0
    out["day_entry"] = entry_ok(t)
    out["breach"] = {
        "tws_warn": out["max_tws"] > LIMITS["tws_warn"], "tws_no": out["max_tws"] > LIMITS["tws_no"],
        "gust_warn": out["max_gust"] > LIMITS["gust_warn"], "gust_no": out["max_gust"] > LIMITS["gust_no"],
    }
    out["wave_breach"] = {wm: out[f"wave_warn_{wm}"] or out[f"wave_no_{wm}"] for wm in WAVE_MODELS}
    return out


def summarize(done, n_all):
    q = lambda key, pct: float(np.percentile([r[key] for r in done], pct))
    pr = lambda fn: sum(1 for r in done if fn(r)) / len(done)
    med = q("hours", 50)
    return dict(
        n=len(done), of=n_all,
        hours_p10=q("hours", 10), hours_p50=med, hours_p90=q("hours", 90),
        arrive_p50=min(done, key=lambda r: abs(r["hours"] - med))["arrive"],
        tws_p50=q("max_tws", 50), tws_p90=q("max_tws", 90), gust_p90=q("max_gust", 90),
        **{f"hs_{wm}_p90": q(f"hs_{wm}", 90) for wm in WAVE_MODELS},
        **{f"ratio_{wm}_p10": q(f"ratio_{wm}", 10) for wm in WAVE_MODELS},
        **{f"p_wave_{wm}": pr(lambda r, wm=wm: r["wave_breach"][wm]) for wm in WAVE_MODELS},
        **{f"p_wave_no_{wm}": pr(lambda r, wm=wm: r[f"wave_no_{wm}"]) for wm in WAVE_MODELS},
        **{f"wave_why_{wm}": sorted(set().union(*[r[f"wave_why_{wm}"] for r in done])) for wm in WAVE_MODELS},
        beat_p50=q("beat_h", 50), motor_p50=q("motor_h", 50), current_kt=q("current_along_kt", 50),
        **{f"{k}_share": float(np.mean([r[f"{k}_share"] for r in done]))
           for k in [k for k, _ in POINTS_OF_SAIL] + [f"sea_{k}" for k in comfort.SEA_SECTORS]},
        comfort_h=[float(np.mean([r["comfort_h"][i] for r in done])) for i in range(len(comfort.LEVELS))],
        p_tws=pr(lambda r: r["breach"]["tws_warn"]), p_tws_no=pr(lambda r: r["breach"]["tws_no"]),
        p_gust=pr(lambda r: r["breach"]["gust_warn"]), p_gust_no=pr(lambda r: r["breach"]["gust_no"]),
        p_any=pr(lambda r: r["breach"]["tws_warn"] or r["breach"]["gust_warn"]),
        p_no=pr(lambda r: r["breach"]["tws_no"] or r["breach"]["gust_no"]),
        p_day=pr(lambda r: r["day_entry"]),
    )


def score(run_dir, first=None, last=None, step_h=DEPARTURE_STEP_H):
    """Score departures from first to last (aware datetimes). Stops early when the
    forecast no longer covers enough members' passages; those departures are left out."""
    wind, waves, manifest = load(run_dir)
    name = manifest.get("route", "port_resolution")
    if name not in route.ROUTES and manifest.get("waypoints"):
        route.define(name, [tuple(w) for w in manifest["waypoints"]])
    route.use(name)
    vmc = polar.vmc_table(polar.conservative_polar())
    rt = route_table()
    rows = []
    depart = (first or FIRST_DEPARTURE).astimezone(timezone.utc)
    while last is None or depart <= last.astimezone(timezone.utc):
        row = dict(depart=depart)
        for model in WIND_MODELS:
            w = wind[model]
            res = [sail(m, depart, w, waves, vmc, rt) for m in range(w["spd"].shape[0])]
            done = [r for r in res if r]
            if len(done) < MIN_COMPLETE * len(res):
                return rows, manifest
            row[model] = summarize(done, len(res))
        rows.append(row)
        depart += timedelta(hours=step_h)
    return rows, manifest


def wave_flags(row, level="warn"):
    """Wave models where more than WAVE_FLAG_SHARE of members meet the warning (or no-go) wave rules."""
    key = "p_wave_" if level == "warn" else "p_wave_no_"
    return [label for wm, label in WAVE_MODELS.items()
            if max(row[m][f"{key}{wm}"] for m in WIND_MODELS) > WAVE_FLAG_SHARE]


def verdict(row):
    """Worst of the wind tier, the gust tier and the wave rules."""
    no = max(row[m]["p_no"] for m in WIND_MODELS)
    warn = max(row[m]["p_any"] for m in WIND_MODELS)
    if no > TIER_SHARE or wave_flags(row, "no"):
        return "NO"
    if warn > TIER_SHARE or wave_flags(row, "warn"):
        return "WARNING"
    return "GO"


def main():
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        run_dir = sorted(d for d in (HERE / "data").iterdir()
                         if (d / "manifest.json").exists() and not d.name.startswith("synoptic_")
                         and json.loads((d / "manifest.json").read_text()).get("route", "port_resolution")
                         == "port_resolution")[-1]
    rows, manifest = score(run_dir)
    print(f"forecast fetched {manifest['fetched_utc']}; runs: "
          + ", ".join(f"{v['label']} {v['run'] or 'n/a'}" for v in manifest["models"].values()))
    print(f"route {route.total_nm():.0f} nm offshore + {route.EXIT_HOURS:.0f} h exit; limits {LIMITS}")
    print("depart FJT     verdict  waves    | model  wind-risk tws>  gst>  beat> | maxTWS p50/p90 gust90 | "
          "Hs90 WAM/GFSW  T/Hs10 WAM/GFSW | hrs p10/50/90 | motor50 dayEntry | ETA p50 VUT")
    out_csv = HERE / "output" / f"windows_{manifest['fetched_utc']}.csv"
    out_csv.parent.mkdir(exist_ok=True)
    keys = ("p_any", "p_no", "p_tws", "p_gust", "tws_p50", "tws_p90", "gust_p90", "hs_ecmwf_wam025_p90",
            "hs_ncep_gfswave025_p90", "ratio_ecmwf_wam025_p10", "ratio_ncep_gfswave025_p10",
            "p_wave_ecmwf_wam025", "p_wave_ncep_gfswave025", "hours_p10", "hours_p50", "hours_p90", "beat_p50", "motor_p50", "p_day")
    with open(out_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["depart_fjt", "verdict", "wave_flags", "model", "members", *keys, "eta_p50_vut"])
        for row in rows:
            v, flags = verdict(row), " ".join(wave_flags(row))
            for j, model in enumerate(WIND_MODELS):
                r = row[model]
                lead = f"{row['depart'].astimezone(FJT):%a %d %H:%M}   {v:8s} {flags:8s}" if j == 0 else " " * 38
                print(f"{lead} | {WIND_MODELS[model]:5s}  {r['p_any']:4.0%}    {r['p_tws']:4.0%}  {r['p_gust']:4.0%}  "
                      f"{r['p_no']:4.0%} |   {r['tws_p50']:4.1f}/{r['tws_p90']:4.1f}   {r['gust_p90']:4.1f} | "
                      f"   {r['hs_ecmwf_wam025_p90']:3.1f}/{r['hs_ncep_gfswave025_p90']:3.1f}     "
                      f"{r['ratio_ecmwf_wam025_p10']:3.1f}/{r['ratio_ncep_gfswave025_p10']:3.1f}     | "
                      f"{r['hours_p10']:4.0f}/{r['hours_p50']:3.0f}/{r['hours_p90']:3.0f}   | "
                      f"  {r['motor_p50']:4.1f}   {r['p_day']:4.0%}   | {r['arrive_p50'].astimezone(VUT):%a %d %H:%M}")
                wr.writerow([f"{row['depart'].astimezone(FJT):%Y-%m-%d %H:%M}", v, flags, model,
                             f"{r['n']}/{r['of']}", *(round(r[k], 2) for k in keys),
                             f"{r['arrive_p50'].astimezone(VUT):%Y-%m-%d %H:%M}"])
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
