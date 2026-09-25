"""Day-by-day forecast of one passage: conditions, boat performance, comfort, distance.

Every member of both wind ensembles is sailed from the departure time with two
polars: the conservative envelope polar, and an optional faster performance polar
with its own speed adjustments. Days run 24 hours from departure. Figures are the
median across all 82 members, with a 10th to 90th percentile range where shown.

Comfort uses the Sereno scale (Sick, Rough, Coffee, Easy, Champagne), from wind over
the deck and boat motion in the sea with point of sail allowed for. See comfort.py.
Not tuned against a logged passage.

Usage: python3 daily.py port_vila "2026-09-14 12:00" [data/<fetch dir>]
       departure time is Fiji time
"""
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import comfort
import polar
import route
import windows

HERE = Path(__file__).parent
LEVELS = comfort.LEVELS      # index 0 is worst
DAY_H = 24


def day_stats(log, depart, arrive):
    days = []
    n_days = int(np.ceil((arrive - depart).total_seconds() / 3600 / DAY_H))
    exit_end = depart + timedelta(hours=route.EXIT_HOURS)
    for d in range(n_days):
        t0, t1 = depart + timedelta(hours=DAY_H * d), depart + timedelta(hours=DAY_H * (d + 1))
        steps = [x for x in log if t0 <= x["t"] < t1]
        before = [x for x in log if x["t"] < t0]
        start_nm = before[-1]["dist"] + before[-1].get("sog", before[-1]["v"]) * windows.STEP_H if before else 0.0
        end_nm = route.total_nm() if t1 >= arrive else (steps[-1]["dist"] + steps[-1].get("sog", steps[-1]["v"]) * windows.STEP_H
                                                          if steps else start_nm)
        if not steps:
            days.append(None)
            continue
        hs = np.array([np.nanmax([x["hs_ecmwf_wam025"], x["hs_ncep_gfswave025"]]) for x in steps])
        ratio = np.array([np.nanmin([x["per_ecmwf_wam025"] / x["hs_ecmwf_wam025"],
                                     x["per_ncep_gfswave025"] / x["hs_ncep_gfswave025"]]) for x in steps])
        lv = np.array([x["comfort"] for x in steps])
        hours_at = [float((lv == k).sum() * windows.STEP_H) for k in range(len(LEVELS))]
        cum = np.cumsum(hours_at)                    # hours at this level or worse
        worst3 = int(np.argmax(cum >= 3)) if cum[-1] >= 3 else int(lv.min())
        rough = [x for x in steps if x["comfort"] <= worst3]
        by_wind = sum(1 for x in rough if comfort._band_level(x["aws"], comfort.WIND_BANDS) <= worst3)
        by_sea = sum(1 for x in rough if comfort._band_level(x["accel"], comfort.MOTION_BANDS) <= worst3)
        sides = [x["sea_side"] for x in steps if x["sea_side"]]
        sailing_h = (min(t1, arrive) - max(t0, exit_end)).total_seconds() / 3600
        days.append(dict(
            dist=end_nm - start_nm, end_nm=end_nm,
            speed=(end_nm - start_nm) / sailing_h if sailing_h > 0 else np.nan,
            tws_mean=np.mean([x["tws"] for x in steps]), tws_max=max(x["tws"] for x in steps),
            gust_max=max(x["gust"] for x in steps),
            twd_mean=float(np.degrees(np.arctan2(np.mean([np.sin(np.radians(x["twd"])) for x in steps]),
                                                 np.mean([np.cos(np.radians(x["twd"])) for x in steps]))) % 360),
            twa_mean=np.mean([x["course_twa"] for x in steps]),
            hs_max=float(np.nanmax(hs)), ratio_min=float(np.nanmin(ratio)),
            per_at_max=float(np.nanmax([x["per_ecmwf_wam025"] if x["hs_ecmwf_wam025"] >= x["hs_ncep_gfswave025"]
                                        else x["per_ncep_gfswave025"] for x in steps
                                        if np.nanmax([x["hs_ecmwf_wam025"], x["hs_ncep_gfswave025"]]) >= np.nanmax(hs) - 1e-9]
                                       or [np.nan])),
            motor_h=sum(windows.STEP_H for x in steps if x["motoring"]),
            current_kt=float(np.mean([x.get("current_along", 0.0) for x in steps])),
            fuel_gal=sum(windows.STEP_H for x in steps if x["motoring"]) * windows.FUEL_GPH,
            beat_h=sum(windows.STEP_H for x in steps if x["course_twa"] < 60),
            level_main=int(np.argmax(hours_at)), level_worst3=worst3, hours_at=hours_at,
            aws_max=max(x["aws"] for x in steps), accel_max=max(x["accel"] for x in steps),
            worst_cause=("wind and sea" if by_wind and by_sea else "wind over the deck" if by_wind else "the sea"),
            **{f"sea_{k}": (sides.count(k) / len(sides) if sides else 0.0) for k in comfort.SEA_SECTORS},
        ))
    return days


def run(route_name, depart_fjt, run_dir, which=None):
    """which: list of polar names to run, default both."""
    wind, waves, manifest = windows.load(run_dir)
    assert manifest.get("route") == route_name, f"{run_dir} is for {manifest.get('route')}"
    if route_name not in route.ROUTES and manifest.get("waypoints"):
        route.define(route_name, [tuple(w) for w in manifest["waypoints"]])
    route.use(route_name)
    rt = windows.route_table()
    pw_day, pw_night = polar.pw_polars()
    polars = {"Conservative polar": dict(vmc=polar.vmc_table(polar.conservative_polar()), vmc_night=None)}
    if pw_day is not None:   # the performance polar is optional
        polars["Performance polar"] = dict(vmc=polar.vmc_table(pw_day), vmc_night=polar.vmc_table(pw_night))
    depart = depart_fjt.astimezone(timezone.utc)
    results = {}
    for pname, pv in polars.items():
        if which and pname not in which:
            continue
        members = []
        for model in windows.WIND_MODELS:
            w = wind[model]
            for m in range(w["spd"].shape[0]):
                log = []
                r = windows.sail(m, depart, w, waves, pv["vmc"], rt, log=log, vmc_night=pv["vmc_night"])
                if r:
                    track = [(x["t"], x["dist"]) for x in log[::4]] + [(r["arrive"], route.total_nm())]
                    members.append((model, r, day_stats(log, depart, r["arrive"]), track))
        results[pname] = members
    return results, manifest


def report(route_name, depart_fjt, results, manifest):
    q = lambda xs, p: float(np.percentile(xs, p))
    dest_tz = windows.VUT
    lines = [f"# {route.WAYPOINTS[0][0]} to {route.WAYPOINTS[-1][0]}, departing {depart_fjt:%a %d %b %H:%M} FJT",
             "",
             f"Forecast fetched {manifest['fetched_utc']}. Runs: "
             + ", ".join(f"{v['label']} {v['run'] or 'n/a'}" for v in manifest["models"].values()) + ".",
             f"Route {route.total_nm():.0f} nm plus {route.EXIT_HOURS:g} h to clear the harbor. "
             "Days run 24 h from departure. Medians across both ensembles, 10th to 90th percentile in brackets.",
             ""]
    csv_rows = []
    for pname, members in results.items():
        hours = [r["hours"] for _, r, _, _ in members]
        arrive = sorted(r["arrive"] for _, r, _, _ in members)
        lines += [f"## {pname}", "",
                  f"Passage {q(hours, 50):.0f} h ({q(hours, 10):.0f} to {q(hours, 90):.0f}), "
                  f"{len(members)} of 82 members arrive inside the forecast. "
                  f"Median arrival {arrive[len(arrive) // 2].astimezone(dest_tz):%a %d %H:%M} VUT.", "",
                  "| Day | Ends FJT | Wind kt, mean / max (gust) | From | TWA | Waves max, min T/H | "
                  "Boat kt | Distance nm | Run total nm | Motor h | Comfort, most of day / worst 3 h |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        n_days = max(len(d) for _, _, d, _ in members)
        for k in range(n_days):
            ds = [d[k] for _, _, d, _ in members if k < len(d) and d[k]]
            if len(ds) < 0.5 * len(members):
                break
            g = lambda key, p=50: q([x[key] for x in ds], p)
            main = LEVELS[int(round(np.median([x["level_main"] for x in ds])))]
            worst = LEVELS[int(np.floor(np.median([x["level_worst3"] for x in ds])))]
            end = depart_fjt + timedelta(hours=DAY_H * (k + 1))
            twd = float(np.degrees(np.arctan2(np.median([np.sin(np.radians(x["twd_mean"])) for x in ds]),
                                              np.median([np.cos(np.radians(x["twd_mean"])) for x in ds]))) % 360)
            lines.append(
                f"| {k + 1} | {end:%a %d %H:%M} | {g('tws_mean'):.0f} / {g('tws_max'):.0f} ({g('gust_max'):.0f}) "
                f"| {twd:03.0f} | {g('twa_mean'):.0f} | {g('hs_max'):.1f} m, {g('ratio_min'):.1f} "
                f"| {g('speed'):.1f} | {g('dist'):.0f} ({g('dist', 10):.0f}–{g('dist', 90):.0f}) "
                f"| {g('end_nm'):.0f} | {g('motor_h'):.1f} | {main} / {worst} |"
                + ("" if len(ds) == len(members) else f" {len(ds)} of {len(members)} still sailing"))
            csv_rows.append([pname, k + 1, f"{end:%Y-%m-%d %H:%M}", len(ds)] +
                            [round(g(key, p), 2) for key in ("tws_mean", "tws_max", "gust_max", "twa_mean",
                                                             "hs_max", "ratio_min", "speed", "dist", "end_nm",
                                                             "motor_h") for p in (10, 50, 90)] +
                            [round(twd), main, worst])
        lines.append("")
    lines += ["Comfort levels (Sereno scale): Champagne, flat and steady. Easy, sleeping fine. "
              "Coffee, one hand to move, still cooking. Rough, two hands, no cooking. Sick, nobody is sleeping.",
              "The mapping from forecast to level is a first guess and hasn't been checked against a real passage.",
              "Speeds in the table are over the ground, with the SMOC surface current along the line.", ""]
    stamp = f"{route_name}_{depart_fjt:%Y%m%dT%H%M}_{manifest['fetched_utc']}"
    md = HERE / "output" / f"daily_{stamp}.md"
    md.write_text("\n".join(lines))
    with open(HERE / "output" / f"daily_{stamp}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["polar", "day", "ends_fjt", "members"] +
                    [f"{key}_p{p}" for key in ("tws_mean", "tws_max", "gust_max", "twa_mean", "hs_max",
                                               "ratio_min", "speed", "dist", "end_nm", "motor_h")
                     for p in (10, 50, 90)] + ["twd_mean", "comfort_main", "comfort_worst3h"])
        wr.writerows(csv_rows)
    return md


def main():
    route_name = sys.argv[1]
    depart_fjt = datetime.strptime(sys.argv[2], "%Y-%m-%d %H:%M").replace(tzinfo=windows.FJT)
    if len(sys.argv) > 3:
        run_dir = Path(sys.argv[3])
    else:
        run_dir = sorted(d for d in (HERE / "data").iterdir()
                         if (d / "manifest.json").exists() and not d.name.startswith("synoptic_")
                         and json.loads((d / "manifest.json").read_text()).get("route") == route_name)[-1]
    results, manifest = run(route_name, depart_fjt, run_dir)
    md = report(route_name, depart_fjt, results, manifest)
    print(md.read_text())


if __name__ == "__main__":
    main()
