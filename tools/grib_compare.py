"""Sail a downloaded GRIB (as used by LuckGrib) through the passage model, to compare with a router.

Usage (needs a venv with eccodes and numpy):
    GRIB=~/Downloads/GFS.20260914.18.grb2 RUN_H=18 python tools/grib_compare.py
    GRIB=~/Downloads/ECMWF.20260914.12.grb2 PER=mwp WDIR=mwd RUN_H=12 python tools/grib_compare.py
Start and target points and the run date are the ones from the 2026-09-15 comparison; edit them for another file.
"""
import sys, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, eccodes
from datetime import datetime, timedelta, timezone

import os
GRIB = os.environ["GRIB"]
PER, WDIR = os.environ.get("PER", "perpw"), os.environ.get("WDIR", "dirpw")
RUN_H = int(os.environ.get("RUN_H", "18"))
fields = {}
with open(GRIB, "rb") as f:
    while (h := eccodes.codes_grib_new_from_file(f)) is not None:
        sn, step = eccodes.codes_get(h, "shortName"), eccodes.codes_get(h, "step")
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        lat0, lon0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"), eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        dlat, dlon = eccodes.codes_get(h, "jDirectionIncrementInDegrees"), eccodes.codes_get(h, "iDirectionIncrementInDegrees")
        scan_neg = eccodes.codes_get(h, "jScansPositively") == 0
        vals = eccodes.codes_get_values(h).reshape(nj, ni)
        miss = eccodes.codes_get(h, "missingValue")
        vals = np.where(vals == miss, np.nan, vals)
        fields.setdefault(sn, {})[step] = vals
        eccodes.codes_release(h)
run = datetime(2026, 9, 14, RUN_H, tzinfo=timezone.utc)
lats = lat0 - np.arange(nj) * dlat if scan_neg else lat0 + np.arange(nj) * dlat
lons = lon0 + np.arange(ni) * dlon
print("lat order", lats[0], lats[-1], "lon", lons[0], lons[-1])

def at(sn, step, lat, lon):
    a = fields[sn][step]
    fi = np.interp(lat, lats[::-1], np.arange(nj)[::-1]) if lats[0] > lats[-1] else np.interp(lat, lats, np.arange(nj))
    fj = np.interp(lon % 360, lons, np.arange(ni))
    i0, j0 = int(fi), int(fj); wi, wj = fi - i0, fj - j0
    return (a[i0, j0] * (1 - wi) * (1 - wj) + a[i0 + 1, j0] * wi * (1 - wj) + a[i0, j0 + 1] * (1 - wi) * wj + a[i0 + 1, j0 + 1] * wi * wj)

# 1) spot check against LuckGrib's first line: Sep 22 12:03 FJT = 00:03 UTC, 18 03.6S 177 08.3E
lat, lon = -(18 + 3.6 / 60), 177 + 8.3 / 60
steps_all = sorted(fields["10u"])
print("steps:", steps_all)
target_h = (datetime(2026, 9, 22, 0, 3, tzinfo=timezone.utc) - run).total_seconds() / 3600
def at_t(sn, h, lat, lon):
    s0 = max(s for s in steps_all if s <= h); s1 = min(s for s in steps_all if s >= h)
    w = 0 if s1 == s0 else (h - s0) / (s1 - s0)
    return at(sn, s0, lat, lon) * (1 - w) + at(sn, s1, lat, lon) * w
u, v = at_t("10u", target_h, lat, lon), at_t("10v", target_h, lat, lon)
print(f"+{target_h:.2f} h at LuckGrib start: TWS {math.hypot(u, v) * 1.943844:.1f} kt from {(math.degrees(math.atan2(-u, -v)) + 360) % 360:.0f} deg, "
      f"waves {at_t('swh', target_h, lat, lon) * 3.28084:.1f} ft, period {at_t(PER, target_h, lat, lon):.1f} s from {at_t(WDIR, target_h, lat, lon):.0f}, "
      f"")

# 2) build hourly arrays along the LuckGrib start/target great circle and sail it
import route, windows, polar, comfort
route.define("lg", [("LuckGrib start", lat, lon), ("LuckGrib target", -(19 + 30.8 / 60), 169 + 29.1 / 60)])
route.use("lg"); route.EXIT_HOURS = 0.0
pts = route.sample_points()
steps = sorted(fields["10u"])
hours = np.arange(0, steps[-1] + 1)
def series(sn, conv=lambda x: x):
    out = np.zeros((len(hours), len(pts)))
    for k, (_, la, lo) in enumerate(pts):
        vals = [at(sn, s, la, lo) for s in steps]
        out[:, k] = np.interp(hours, steps, vals)
    return out
U, V = series("10u"), series("10v")
spd = np.hypot(U, V) * 1.943844
wdir = (np.degrees(np.arctan2(-U, -V)) + 360) % 360
gust = spd * 1.23
times = [run + timedelta(hours=int(hh)) for hh in hours]
wind = dict(times=times, spd=spd[None], dir=wdir[None], gust=gust[None])
# wave direction: interpolating an angle linearly is wrong across 0/360, so interpolate sin/cos
def dir_series(sn):
    s_, c_ = np.zeros((len(hours), len(pts))), np.zeros((len(hours), len(pts)))
    for k, (_, la, lo) in enumerate(pts):
        d = np.radians([at(sn, s, la, lo) for s in steps])
        s_[:, k] = np.interp(hours, steps, np.sin(d)); c_[:, k] = np.interp(hours, steps, np.cos(d))
    return (np.degrees(np.arctan2(s_, c_)) + 360) % 360
waves = {"ncep_gfswave025": dict(times=times, hs=series("swh"), period=series(PER), dir=dir_series(WDIR))}
# the scorer expects both wave model keys; give ECMWF's slot the same GFS waves
waves["ecmwf_wam025"] = waves["ncep_gfswave025"]
rt = windows.route_table()
depart = datetime(2026, 9, 22, 12, 3, tzinfo=windows.FJT).astimezone(timezone.utc)

src = polar.load_sources()
lg_grid = np.full((len(polar.TWA), len(polar.TWS)), np.nan)
for i, a in enumerate(polar.TWA):
    for j, s in enumerate(polar.TWS):
        lg_grid[i, j] = src["sabado.csv"].get((a, s), src["sabado_polar_v2.csv"].get((a, s)))
setups = {
    "LuckGrib settings: sabado.csv 100%, night 80%": (polar.vmc_table(lg_grid), polar.vmc_table(lg_grid * 0.80)),
    "This tool: conservative polar": (polar.vmc_table(polar.conservative_polar()), None),
}
vut = timezone(timedelta(hours=11))
for name, (vmc, vmc_n) in setups.items():
    log = []
    r = windows.sail(0, depart, wind, waves, vmc, rt, log=log, vmc_night=vmc_n)
    tws = [x["tws"] for x in log]; aws = [x["aws"] for x in log]; twa = [x["course_twa"] for x in log]
    hist = np.histogram(twa, bins=[0, 45, 67, 90, 112, 135, 157, 180])[0] / len(twa)
    print(f"\n{name}")
    print(f"  {r['hours']:.1f} h over {route.total_nm():.0f} nm, arrive {r['arrive'].astimezone(windows.FJT):%a %d %H:%M} FJT, "
          f"avg {route.total_nm() / r['hours']:.1f} kt, motoring {r['motor_h']:.1f} h")
    print(f"  TWS {min(tws):.1f}-{max(tws):.1f} kt, AWS {min(aws):.1f}-{max(aws):.1f} kt, waves max {r['hs_ncep_gfswave025'] * 3.28084:.1f} ft, "
          f"estimated gust max {r['max_gust']:.1f} kt, model gust max {max(x['gust'] for x in log):.1f}")
    print("  TWA share, bins 0/45/67/90/112/135/157/180:", " ".join(f"{p:.0%}" for p in hist))
    print("  point of sail: " + ", ".join(f"{r[k+'_share']:.0%} {k}" for k in ("up","beam","broad","run")) + "; seas " + ", ".join(f"{r['sea_'+k+'_share']:.0%} {k}" for k in ("bow","fwd_quarter","aft_quarter","astern")))
    print(f"  "

          f"comfort hours by level {dict(zip(comfort.LEVELS, [round(x,1) for x in r['comfort_h']]))}")
