"""Screen departure options for a trip: a short brief covering each day in the window.

Uses the wind ensembles and wave models along the route (no synoptic grid), so it
stays small enough to run twice a day. Each 3-hourly departure slot is sailed by
all 82 ensemble members with the conservative polar and scored against the limits in Settings.
One departure per day, at local noon at the origin.
"""
import html
import json
from datetime import datetime, timedelta, timezone

import numpy as np

import comfort
import daily
import fetch
import report_html
import route
import trip as T
import windows

VERDICT_RANK = {"GO": 0, "WARNING": 1, "MARGINAL": 1, "NO": 2}          # MARGINAL: history before 2026-09-15
VERDICT_CLASS = {"GO": "c5", "WARNING": "c3", "MARGINAL": "c3", "NO": "c1"}
LEVEL_CLASS = report_html.LEVEL_CLASS


def esc(x):
    return html.escape(str(x))


def slot_summary(row):
    e, g = row["ecmwf_ifs025"], row["gfs_seamless"]
    both = (e, g)
    return dict(
        depart=row["depart"], verdict=windows.verdict(row), flags=windows.wave_flags(row), wave_notes=wave_warnings(row),
        risk=max(r["p_any"] for r in both), p_tws=max(r["p_tws"] for r in both), p_gust=max(r["p_gust"] for r in both),
        p_no=max(r["p_no"] for r in both), p_tws_no=max(r["p_tws_no"] for r in both),
        p_gust_no=max(r["p_gust_no"] for r in both),
        tws50=max(r["tws_p50"] for r in both), tws90=max(r["tws_p90"] for r in both),
        gust90=max(r["gust_p90"] for r in both),
        hs90=max(max(r["hs_ecmwf_wam025_p90"], r["hs_ncep_gfswave025_p90"]) for r in both),
        ratio10=min(min(r["ratio_ecmwf_wam025_p10"], r["ratio_ncep_gfswave025_p10"]) for r in both),
        hours50=np.mean([r["hours_p50"] for r in both]), hours10=min(r["hours_p10"] for r in both),
        hours90=max(r["hours_p90"] for r in both), arrive=e["arrive_p50"], p_day=np.mean([r["p_day"] for r in both]),
        motor50=max(r["motor_p50"] for r in both), current_kt=float(np.mean([r["current_kt"] for r in both])),
        **{f"{k}_share": float(np.mean([r[f"{k}_share"] for r in both])) for k, _ in windows.POINTS_OF_SAIL})


WAVE_NAMES = {"ecmwf_wam025": "ECMWF", "ncep_gfswave025": "GFS"}


def wave_warnings(row):
    """Plain-language reason for each wave model that breaks the wave rules on this departure."""
    out = []
    for wm, name in WAVE_NAMES.items():
        share = max(row[m][f"p_wave_{wm}"] for m in windows.WIND_MODELS)
        if share <= windows.WAVE_FLAG_SHARE:
            continue
        why = sorted(set().union(*[set(row[m].get(f"wave_why_{wm}", [])) for m in windows.WIND_MODELS]))
        hs = max(row[m][f"hs_{wm}_p90"] for m in windows.WIND_MODELS)
        no = max(row[m][f"p_wave_no_{wm}"] for m in windows.WIND_MODELS) > windows.WAVE_FLAG_SHARE
        out.append(f"{name} wave model: {', '.join(why) or 'over your wave rules'} (to {hs:.1f} m)"
                   + (", a no-go" if no else ""))
    return out


POS_NAMES = {"up": "upwind", "beam": "beam reach", "broad": "broad reach", "run": "running"}


def point_of_sail_text(shares):
    """'62% broad reach, 38% beam reach': every band with 5% or more, largest first."""
    parts = sorted(((shares[f"{k}_share"], k) for k, _ in windows.POINTS_OF_SAIL), reverse=True)
    return ", ".join(f"{v:.0%} {POS_NAMES[k]}" for v, k in parts if v >= 0.05) or "not enough data"


def wave_periods(sea):
    """', mean 11 s, peak 15 s; swell 15 s SSW, wind sea 6 s ESE' from a sea summary."""
    if not sea:
        return ""
    import seastate
    bits = []
    e = sea.get("ecmwf")
    if e:
        bits.append(f", mean {e['t_med']:.0f} s" + (f", peak {e['tp_med']:.0f} s" if e['tp_med'] == e['tp_med'] else ""))
    sub = []
    for key, name in (("swell", "swell"), ("wind", "wind sea")):
        x = sea.get(key)
        if x:
            sub.append(f"{name} {x['h_max']:.1f} m {x['t_med']:.0f} s {seastate.compass(x['frm'])}")
    return "".join(bits) + (f'<span class="sub">{"; ".join(sub)}</span>' if sub else "")


def current_text(kt):
    if abs(kt) < 0.05:
        return "no net effect"
    return f"{abs(kt):.1f} kt {'with you' if kt > 0 else 'against you'} on average"


def verdict_reason(slot):
    """Why this day got its verdict, naming the check that decided it."""
    L, share = windows.LIMITS, windows.TIER_SHARE
    wind, waves = [], list(slot.get("wave_notes", []))
    if slot.get("p_tws_no", 0) > share:
        wind.append(f"{slot['p_tws_no']:.0%} of passages reach {L['tws_no']:.0f} kt")
    elif slot.get("p_tws", 0) > share:
        wind.append(f"{slot['p_tws']:.0%} reach {L['tws_warn']:.0f} kt")
    if slot.get("p_gust_no", 0) > share:
        wind.append(f"{slot['p_gust_no']:.0%} gust over {L['gust_no']:.0f} kt")
    elif slot.get("p_gust", 0) > share:
        wind.append(f"{slot['p_gust']:.0%} gust over {L['gust_warn']:.0f} kt")
    if slot["verdict"] == "GO":
        return "Wind and seas inside your limits."
    reasons = []
    if waves:
        reasons.append("waves over your limits, " + "; ".join(w.replace(" wave model:", "") for w in waves))
    if wind:
        reasons.append("wind, with " + " and ".join(wind))
    if not reasons:
        return "No single check decided it."
    lead = "NO on " if slot["verdict"] == "NO" else "Warning on "
    tail = "" if wind else " Wind stays inside your limits."
    return lead + ", and ".join(reasons) + "." + tail


def best_key(s):
    return (VERDICT_RANK[s["verdict"]], round(s["risk"], 2), s["p_day"] < 0.5, s["hours50"])


def screen(t, run_dir=None):
    t.use_route()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    if run_dir is None:
        run_dir = fetch.fetch_into(t.dir / "runs" / stamp, verbose=False)
    first, last = t.window()
    rows, manifest = windows.score(run_dir, first, last, step_h=24)
    slots = [slot_summary(r) for r in rows]

    days = []
    for k in range(t.days):
        d0 = (first + timedelta(days=k)).replace(hour=0)
        in_day = [s for s in slots if d0 <= s["depart"].astimezone(t.origin_tz) < d0 + timedelta(days=1)]
        day = dict(date=d0, slots=in_day, lead_days=(d0 - datetime.now(t.origin_tz)).total_seconds() / 86400)
        if in_day:
            best = min(in_day, key=best_key)
            day["best"] = best
            res, _ = daily.run(route.NAME, best["depart"].astimezone(t.origin_tz), run_dir, which=["Conservative polar"])
            members = res["Conservative polar"]
            try:
                import seastate
                import synoptic_report as SR
                dist_at = SR.median_track(members)
                arrive = sorted(r["arrive"] for _, r, _, _ in members)[len(members) // 2]
                sea_rows = seastate.along_track(seastate.load(run_dir), dist_at, best["depart"], arrive)
                day["sea"] = seastate.summarize(sea_rows)
                day["sea_text"] = seastate.describe(day["sea"], short=True)
            except Exception as e:                      # older fetches have no wave split
                day["sea"], day["sea_text"] = None, ""
            day["comfort_main"], day["comfort_worst"], day["comfort"], day["comfort_text"] = comfort.summarize_passage(members)
            day["fuel"] = float(np.median([r["motor_h"] for _, r, _, _ in members])) * windows.FUEL_GPH
        days.append(day)

    hist_file = t.dir / "history.json"
    history = json.loads(hist_file.read_text()) if hist_file.exists() else []
    # a rebuild from the same forecast replaces its row instead of adding another
    history = [h for h in history if h.get("fetched_utc") != manifest["fetched_utc"]]
    history.append(dict(fetched_utc=manifest["fetched_utc"],
                        runs={v["label"]: v["run"] for v in manifest["models"].values()},
                        days={d["date"].strftime("%Y-%m-%d"): (dict(verdict=d["best"]["verdict"], risk=round(d["best"]["risk"], 3),
                                                                    best=d["best"]["depart"].astimezone(t.origin_tz).strftime("%H:%M"))
                                                               if "best" in d else None) for d in days}))
    hist_file.write_text(json.dumps(history, indent=1))
    t.dir.mkdir(parents=True, exist_ok=True)
    out = t.dir / "options.html"
    out.write_text(render(t, days, manifest, history))
    pdf = report_html.write_pdf(out)
    return out, pdf, days


def recommendation(t, days):
    scored = [d for d in days if "best" in d]
    if not scored:
        return "The forecast doesn't reach far enough to score any day in this window yet."
    best = min(scored, key=lambda d: best_key(d["best"]))
    b = best["best"]
    when = b["depart"].astimezone(t.origin_tz)
    gos = [d for d in scored if d["best"]["verdict"] == "GO"]
    txt = (f"Best day on this forecast: {when:%A %d %B}, leaving at {when:%H:%M}, {b['verdict']}. "
           + verdict_reason(b))
    if len(gos) > 1:
        txt += " GO days: " + ", ".join(f"{d['date']:%a %d}" for d in gos) + "."
    elif not gos:
        txt += " No day in the window clears every limit."
    far = [d for d in scored if d["lead_days"] > 7]
    if far:
        txt += " Days more than a week out carry little forecast skill, so expect them to change."
    return txt


def render(t, days, manifest, history):
    oz, dz = t.origin_tz, t.dest_tz
    first, last = t.window()
    now = datetime.now(timezone.utc)
    ozn, dzn = T.tz_abbrev(first, oz), T.tz_abbrev(first, dz)
    runs = ", ".join(f"{esc(v['label'])} {esc(v['run'] or 'run time not published')}" for v in manifest["models"].values())
    L = windows.LIMITS
    h = [f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(t.title())} options</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Open+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400&display=swap">
<style>{report_html.CSS}{EXTRA_CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow">{esc(report_html.boat_label())} departure options</div>
  <h1>{esc(t.title())}</h1>
  <p class="lede">Leaving at noon any day from {first:%A %d %B} for {t.days} days. {route.total_nm():.0f} nm on the great circle.</p>
  <p class="meta">Forecast fetched {esc(manifest['fetched_utc'])} UTC. Runs: {runs}. Departure times in {ozn} ({esc(t.origin['tz'])}), arrivals in {dzn} ({esc(t.dest['tz'])}).</p>
</header>
<p class="callout"><b>{esc(recommendation(t, days))}</b></p>
<section id="days-grid"><div class="cards">"""]
    for d in days:
        if "best" not in d:
            h.append(f"""<div class="card"><div class="cardhead"><span class="dname">{d['date']:%a %d %b}</span><span class="chip nodata">No forecast</span></div>
<p class="meta">The forecast doesn't cover a full passage from this day yet.</p></div>""")
            continue
        b = d["best"]
        flags = ("Wave warning. " + ". ".join(b["wave_notes"]) + ".") if b["wave_notes"] else ""
        h.append(f"""<div class="card"><div class="cardhead"><span class="dname">{d['date']:%a %d %b}</span>
<span class="chip {VERDICT_CLASS[b['verdict']]}">{b['verdict']}</span></div>
<div class="best">Leave {b['depart'].astimezone(oz):%H:%M}</div>
<p class="why">{esc(verdict_reason(b))}</p>
<dl>
<dt>Wind</dt><dd>over {L['tws_warn']:.0f} kt {b['p_tws']:.0%}, over {L['tws_no']:.0f} kt {b['p_tws_no']:.0%}<span class="sub">gusts over {L['gust_warn']:.0f} kt {b['p_gust']:.0%}, over {L['gust_no']:.0f} kt {b['p_gust_no']:.0%}</span></dd>
<dt>Strongest</dt><dd>{b['tws50']:.0f} kt, 90th pct {b['tws90']:.0f}<span class="sub">gusts {b['gust90']:.0f} kt</span></dd>
<dt>Waves</dt><dd>{b['hs90']:.1f} m{wave_periods(d.get('sea'))}<span class="sub">T/H {b['ratio10']:.1f}</span></dd>
<dt>Point of sail</dt><dd>{point_of_sail_text(b)}</dd>
<dt>Current</dt><dd>{current_text(b['current_kt'])}</dd>
<dt>At sea</dt><dd>{b['hours50']:.0f} h<span class="sub">{b['hours10']:.0f} to {b['hours90']:.0f} h</span></dd>
<dt>Arrive</dt><dd>{b['arrive'].astimezone(dz):%a %H:%M}<span class="sub">daylight {b['p_day']:.0%}</span></dd>
<dt>Comfort</dt><dd>mostly <span class="chip {LEVEL_CLASS[d['comfort_main']]}">{d['comfort_main']}</span><span class="sub">worst 3 h <span class="chip {LEVEL_CLASS[d['comfort_worst']]}">{d['comfort_worst']}</span></span></dd>
<dt>Fuel</dt><dd>{d['fuel']:.0f} gal</dd>
</dl><p class="comfort-note">{esc(d['comfort_text'])}</p>{f'<p class="comfort-note"><b>Sea.</b> {esc(d["sea_text"])}</p>' if d.get('sea_text') else ''}{f'<p class="meta">{esc(flags.strip())}</p>' if flags else ''}{'<p class="meta">More than a week out: low forecast skill.</p>' if d['lead_days'] > 7 else ''}</div>""")
    h.append("</div></section>")

    # trend
    recent = history[-8:]
    day_keys = [d["date"].strftime("%Y-%m-%d") for d in days]
    h.append(f"""<section id="trend"><h2>How the options have moved</h2>
<p class="col">Each row is one forecast update, newest at the bottom. A day that holds its verdict across several updates is the one to trust.</p>
<div class="tablewrap"><table class="slots"><thead><tr><th>Fetched UTC</th>{''.join(f'<th>{datetime.strptime(k, "%Y-%m-%d"):%a %d}</th>' for k in day_keys)}</tr></thead><tbody>""")
    for hrow in recent:
        cells = []
        for k in day_keys:
            v = hrow["days"].get(k)
            cells.append(f'<td><span class="chip {VERDICT_CLASS[v["verdict"]]}">{v["risk"]:.0%}</span><span class="sub">{v["best"]}</span></td>'
                         if v else '<td class="empty">·</td>')
        h.append(f"<tr><td>{esc(hrow['fetched_utc'])}</td>{''.join(cells)}</tr>")
    h.append("</tbody></table></div></section>")

    h.append(f"""<section id="about"><h2>About this screening</h2><div class="col">
<p>Each day's noon departure is sailed by every member of the ECMWF (51) and GEFS (31) ensembles along the great circle, with the conservative polar and motoring at {windows.MOTOR_KT:g} kt when sailing drops under {windows.MOTOR_BELOW_KT:g} kt.</p>
<p><b>Verdict.</b> The worst of three checks, each counting when more than 10% of passages reach it. Sustained wind: GO under {L['tws_warn']:.0f} kt, WARNING {L['tws_warn']:.0f} to {L['tws_no']:.0f} kt, NO over {L['tws_no']:.0f} kt. Gusts: WARNING over {L['gust_warn']:.0f} kt, NO over {L['gust_no']:.0f} kt, estimated as 1.23 times the sustained wind, the WMO at-sea factor for a 3-second gust against the 10-minute mean. Waves: seas over {L['hs']:.0f} m or a period in seconds shorter than the height in feet; one wave model gives a WARNING, both give a NO. The models' own gust figures aren't used: ECMWF's is the highest in each 3 hours and ran about 1.4 times the sustained wind, GEFS's is instantaneous and ran about 1.1 times.</p>
<p><b>Comfort.</b> Every 15 minutes of each passage gets the worse of two ratings. Wind over the deck uses apparent wind: Champagne under 12 kt, Easy under 18, Coffee under 23 (first reef on a reach), Rough under 33, Sick above. Motion uses how often the boat meets the waves, which depends on where they come from: seas from astern overtake slowly, seas on the bow arrive fast. The estimated vertical acceleration is rated against ISO 2631-1 comfort bands (Champagne under 0.15 m/s², Easy under 0.315, Coffee under 0.63, Rough under 1.25, Sick above), with extra for beam seas (catamaran roll) and seas on the bow over 1.5 m (bridgedeck slamming). The same 3 m sea with an 8 second period rates Easy from astern, Coffee on the beam and Rough on the bow. The card shows the level for most of the passage and the worst 3 hours. These bands are a first guess until they can be checked against real passages. Point of sail is the share of time at sea by the angle the boat sails to the true wind: upwind under 75° (including tacking or motoring into it), beam reach 75 to 110°, broad reach 110 to 150°, running over 150°. Seas are sorted the same way by where they come from relative to the heading: bow 0 to 45°, forward quarter 45 to 90°, aft quarter 90 to 135°, astern 135 to 180°. Current is the average surface current along the line, from Météo-France's SMOC model (8 km, hourly, 10 days, tides included). The boat is steered to hold the line, so a cross current costs a little speed and the along-track part adds to or subtracts from speed over ground. Beyond the current forecast the last value is held. Arrival counts as daylight between an hour after sunrise and 90 minutes before sunset. </p>
<p>Squall risk and the upper-air analysis are in the full brief. Pick a day to watch to get it.</p>
<p><b>Estimated guide only. The captain develops the detailed route for every passage.</b></p>
</div></section>
<footer>Generated {now:%Y-%m-%d %H:%M} UTC by <span class="mono">passage.py screen {esc(t.slug)}</span>.</footer></div>""")
    return "\n".join(h)


EXTRA_CSS = """
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px;margin:1rem 0}
.card{border:1px solid var(--rule);border-radius:6px;padding:12px 14px;background:var(--paper)}
.cardhead{display:flex;justify-content:space-between;align-items:center;gap:8px}
.dname{font-weight:600;color:var(--ink-strong)}
.best{font-family:"Playfair Display",Georgia,serif;font-size:1.35rem;color:var(--ink-strong);margin:.35rem 0 .2rem}
.card dl{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;margin:.4rem 0 0;font-size:.85rem}
.card dt{color:var(--muted)} .card dd{margin:0;font-variant-numeric:tabular-nums}
.comfort-note{font-size:.78rem;color:var(--muted);margin:.5rem 0 0;line-height:1.45}
.why{font-size:.82rem;margin:.1rem 0 .4rem;line-height:1.45}
.chip.nodata{color:var(--muted);background:var(--paper-2)}
table.slots{min-width:640px}
table.slots td,table.slots th{text-align:center}
table.slots td:first-child,table.slots th:first-child{text-align:left}
td.empty{color:var(--muted)}
@media print{
  #trend,#about{break-before:auto}
  .cards{grid-template-columns:repeat(2,1fr);gap:8px}
  .comfort-note{font-size:7pt;line-height:1.35}
  .why{font-size:7.5pt;margin:.1rem 0 .3rem}
  .card .meta{font-size:7.5pt;margin:.3rem 0 0}
  .card{break-inside:avoid;padding:8px 10px}
  .best{font-size:12pt}
  .card dl{font-size:7.5pt}
  table.slots{min-width:0}
}
"""
