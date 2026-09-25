"""Build a self-contained HTML passage brief: trip estimates, day by day, the weather story, sources.

Usage: python3 report_html.py port_vila "2026-09-14 12:00" ["Title override"]
       departure time is Fiji time. Writes output/brief_<route>_<departure>.html
"""
import base64
import html
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import comfort
import daily
import polar
import route
import settings
import synoptic as S
import synoptic_report as SR
import windows

HERE = Path(__file__).parent
DEST_TZ = {"port_vila": windows.VUT, "port_resolution": windows.VUT}
DEST_NAME = {"port_vila": "Port Vila", "port_resolution": "Port Resolution"}
LEVEL_CLASS = {"Champagne": "c5", "Easy": "c4", "Coffee": "c3", "Rough": "c2", "Sick": "c1"}
RISK_CLASS = {"LOW": "c4", "ISOLATED": "c3", "MODERATE": "c2", "HIGH": "c1"}


def model_reach(smeta, b):
    """Which deterministic models cover the passage, since shorter forecasts drop out."""
    depart = b["depart_fjt"].astimezone(timezone.utc)
    arrive = max(p["a90"] for p in b["polars"]).astimezone(timezone.utc)
    full, partial, none = [], [], []
    for v in smeta["models"].values():
        last = datetime.fromisoformat(v["last_valid"]).replace(tzinfo=timezone.utc) if v.get("last_valid") else None
        if last is None or last < depart:
            none.append(v["label"])
        elif last < arrive:
            partial.append(f"{v['label']} (to {last.astimezone(b['ctx']['origin_tz']):%a %d %H:%M})")
        else:
            full.append(v["label"])
    parts = [f"Covering the whole passage: {', '.join(full)}." if full else "No model covers the whole passage."]
    if partial:
        parts.append(f"Ending partway: {', '.join(partial)}.")
    if none:
        parts.append(f"Ending before departure, so not shown: {', '.join(none)}.")
    parts.append("The verdict, trip estimates and comfort come from the ECMWF and GEFS ensembles. These six "
                 "single-run models feed only the weather story, the charts and the model agreement check.")
    return " ".join(parts)


def sea_section(b):
    import seastate
    if not b.get("sea_days"):
        return ""
    oz = b["ctx"]["origin_tz"]
    out = [f"""<section id="sea"><h2>The sea</h2>
<p class="col">At the boat's median position. Swell and wind sea from GFS-Wave, combined sea from ECMWF. Steepness: flatter than 1 in 30 is gentle, 1 in 12 is steep.</p>
<p class="callout"><b>Whole passage.</b> {esc(seastate.describe(b['sea_passage']))}</p>
<div class="tablewrap"><table class="sea"><thead><tr><th>Day</th><th>Wave train</th><th>Height</th><th>Period</th><th>From</th><th class="l">Side</th><th>Steepest</th><th class="l">Aboard</th></tr></thead><tbody>"""]
    for d in b["sea_days"]:
        s = d["summary"]
        rows = [(k, n) for k, n in (("ecmwf", "Combined"), ("swell", "Swell"), ("wind", "Wind sea"), ("swell2", "Second swell"))
                if s.get(k) and not (k == "swell2" and s[k]["h_max"] < 0.5)]
        note = seastate.verdict_line(s) + (" " + seastate.flags(s) if seastate.flags(s) else "")
        for i, (key, name) in enumerate(rows):
            x = s[key]
            period = f"{x['t_med']:.0f} s" + (f" / {x['tp_med']:.0f}" if key == "ecmwf" and x["tp_med"] == x["tp_med"] else "")
            lead = (f'<td rowspan="{len(rows)}">Day {d["day"]}<span class="sub">{d["start"].astimezone(oz):%a %d}</span></td>'
                    if i == 0 else "")
            tail = f'<td class="l" rowspan="{len(rows)}">{esc(note)}</td>' if i == 0 else ""
            out.append(f"""<tr>{lead}<td>{name}</td><td>{seastate._range(x['h_min'], x['h_max'], '.1f', 'm')}</td><td>{period}</td>
<td>{seastate.compass(x['frm'])}</td><td class="l">{seastate.SIDE_SHORT[x['side']]}</td><td>1 in {x['n_min']:.0f}</td>{tail}</tr>""")
    out.append("""</tbody></table></div><p class="meta">Combined period is ECMWF mean / peak.</p></section>""")
    return "\n".join(out)


def boat_kind():
    """" for a Lagoon 42", or nothing when the type isn't set."""
    kind = settings.get()["boat"].get("kind")
    return f" for a {kind}" if kind else ""


def boat_label():
    """The boat name for the page headers, falling back to the type, then to a neutral label."""
    b = settings.get()["boat"]
    return b.get("name") or b.get("kind") or "Passage weather"


def screen_current(kt):
    import screen
    return screen.current_text(kt)


def screen_pos(shares):
    import screen
    return screen.point_of_sail_text(shares)


def img(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def esc(x):
    return html.escape(str(x))


def latest_route_run(route_name):
    return sorted(d for d in (HERE / "data").iterdir()
                  if (d / "manifest.json").exists()
                  and json.loads((d / "manifest.json").read_text()).get("route") == route_name)[-1]


def latest_bulletin():
    files = sorted((HERE / "data" / "official").glob("FQPS01_NFFN_*.txt"))
    return files[-1] if files else None


def legacy_ctx(route_name):
    return dict(origin="Nadi", dest=DEST_NAME[route_name], origin_tz=windows.FJT, dest_tz=DEST_TZ[route_name],
                origin_label="Fiji time", dest_label="Vanuatu time", chart_label="FJT")


def build(route_name, depart_fjt, run_dir=None, syn_dir=None, bulletin=None, ctx=None):
    """depart_fjt: aware departure time (any zone). ctx names the places and zones for the text."""
    legacy = ctx is None
    ctx = ctx or legacy_ctx(route_name)
    SR.set_timezone(ctx["origin_tz"], ctx["chart_label"])
    run_dir = run_dir or latest_route_run(route_name)
    results, manifest = daily.run(route_name, depart_fjt, run_dir)
    route.use(route_name)
    sd = syn_dir or S.latest_dir()
    smeta = json.loads((sd / "manifest.json").read_text())
    models, points = S.load_all(sd), S.load_route(sd, route_name)
    depart_utc = depart_fjt.astimezone(timezone.utc)
    q = lambda xs, p: float(np.percentile(xs, p))

    cons = results["Conservative polar"]
    arrive_med = sorted(r["arrive"] for _, r, _, _ in cons)[len(cons) // 2]
    dist_at = SR.median_track(cons)
    track = SR.along_track(models, points, depart_utc, arrive_med, dist_at)
    n_days = max(len(d) for _, _, d, _ in cons)
    story = SR.discussion(models, track, depart_fjt, n_days)
    arrival = SR.arrival_note(track, arrive_med)

    have_charts = SR.charts_available()     # matplotlib is optional; without it the brief has no maps
    charts = (Path(run_dir).parent.parent / "charts") if ctx.get("trip") else HERE / "output" / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    stamp = f"{route_name}_{depart_fjt:%Y%m%dT%H%M}"
    ec = models["ecmwf_ifs025"]
    for d in story:
        target = d["start"] + timedelta(hours=12)
        ti = min(range(len(ec.times)), key=lambda i: abs((ec.times[i] - target).total_seconds()) if ec.valid[i] else 1e12)
        lat, lon, _ = route.position_at(dist_at(ec.times[ti]))
        d["sfc"] = charts / f"{stamp}_d{d['day']}_sfc.png"
        d["upr"] = charts / f"{stamp}_d{d['day']}_upr.png"
        d["chart_time"] = ec.times[ti]
        if have_charts:
            SR.surface_chart(ec, ti, d["sfc"], boat=(lat, lon), route_name=route_name)
            SR.upper_chart(ec, ti, d["upr"], boat=(lat, lon), route_name=route_name)
    track_png = charts / f"{stamp}_track.png"
    if have_charts:
        SR.track_panels(track, track_png, depart_utc, arrive_med)

    # ---------------------------------------------------------- summaries per polar
    polars = []
    for pname, members in results.items():
        hours = [r["hours"] for _, r, _, _ in members]
        arr = sorted(r["arrive"] for _, r, _, _ in members)
        rows = []
        for k in range(max(len(d) for _, _, d, _ in members)):
            ds = [d[k] for _, _, d, _ in members if k < len(d) and d[k]]
            if len(ds) < 0.5 * len(members):
                break
            g = lambda key, p=50: q([x[key] for x in ds], p)
            twd = float(np.degrees(np.arctan2(np.median([np.sin(np.radians(x["twd_mean"])) for x in ds]),
                                              np.median([np.cos(np.radians(x["twd_mean"])) for x in ds]))) % 360)
            rows.append(dict(
                day=k + 1, end=depart_fjt + timedelta(days=k + 1), n=len(ds), of=len(members),
                tws=g("tws_mean"), tws_max=g("tws_max"), gust=g("gust_max"), twd=twd, twa=g("twa_mean"),
                hs=g("hs_max"), per=float(np.nanmedian([x.get("per_at_max", np.nan) for x in ds])), ratio=g("ratio_min"), speed=g("speed"), dist=g("dist"), d10=g("dist", 10),
                d90=g("dist", 90), total=g("end_nm"), motor=g("motor_h"), fuel=g("fuel_gal"),
                main=daily.LEVELS[int(round(np.median([x["level_main"] for x in ds])))],
                worst=daily.LEVELS[int(np.floor(np.median([x["level_worst3"] for x in ds])))],
                aws=g("aws_max"), accel=g("accel_max"),
                cause=max(set(x["worst_cause"] for x in ds), key=[x["worst_cause"] for x in ds].count),
                sea={k: float(np.mean([x[f"sea_{k}"] for x in ds])) for k in comfort.SEA_SECTORS}))
        fuel = [r["motor_h"] * windows.FUEL_GPH for _, r, _, _ in members]
        current = float(np.median([r.get("current_along_kt", 0.0) for _, r, _, _ in members]))
        shares = {f"{k}_share": float(np.mean([r[f"{k}_share"] for _, r, _, _ in members])) for k, _ in windows.POINTS_OF_SAIL}
        c_main, c_worst, _, c_text = comfort.summarize_passage(members)
        polars.append(dict(name=pname, h50=q(hours, 50), h10=q(hours, 10), h90=q(hours, 90),
                           f50=q(fuel, 50), f90=q(fuel, 90), shares=shares, current=current, c_main=c_main, c_worst=c_worst, c_text=c_text,
                           a10=arr[int(0.1 * (len(arr) - 1))], a50=arr[len(arr) // 2], a90=arr[int(0.9 * (len(arr) - 1))],
                           n=len(members), rows=rows))

    try:
        import seastate
        sea_rows = seastate.along_track(seastate.load(run_dir), dist_at, depart_utc, arrive_med)
        sea_days = seastate.by_day(sea_rows, depart_utc)
        sea_passage = seastate.summarize(sea_rows)
    except Exception as e:                  # an older fetch without the wave split
        sea_days, sea_passage = [], None
    if bulletin is None and legacy:
        bulletin = latest_bulletin()
    return dict(route_name=route_name, depart_fjt=depart_fjt, manifest=manifest, smeta=smeta, polars=polars,
                story=story, arrival=arrival, track_png=track_png, bulletin=bulletin, track=track, ctx=ctx,
                sea_days=sea_days, sea_passage=sea_passage, have_charts=have_charts,
                charts_dir=charts)


# ------------------------------------------------------------------ HTML

CSS = """
:root{
  --paper:#ffffff; --paper-2:#f6f7f9; --rule:#e7e7e7; --ink:#333333; --ink-strong:#161a1d; --muted:#656565;
  --accent:#03A9F4; --accent-ink:#016E9E; --plate:#fcfcfb;
  --c5:#1b7f4b; --c5-bg:#e3f4ea; --c4:#2f7d32; --c4-bg:#edf6e6; --c3:#8a5a00; --c3-bg:#fcf0d6;
  --c2:#a8410f; --c2-bg:#fbe5d8; --c1:#a61d24; --c1-bg:#f9dfe0;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#0f171c; --paper-2:#152129; --rule:#243540; --ink:#d5dee3; --ink-strong:#f2f6f8; --muted:#94a6b1;
    --accent:#4fc3f7; --accent-ink:#81d4fa; --plate:#fcfcfb;
    --c5:#7ddba5; --c5-bg:#123224; --c4:#a5d98a; --c4-bg:#1b2e15; --c3:#f2c46b; --c3-bg:#33270e;
    --c2:#f5a077; --c2-bg:#3a2014; --c1:#f28b8f; --c1-bg:#3b1618;
  }
}
:root[data-theme="dark"]{
  --paper:#0f171c; --paper-2:#152129; --rule:#243540; --ink:#d5dee3; --ink-strong:#f2f6f8; --muted:#94a6b1;
  --accent:#4fc3f7; --accent-ink:#81d4fa; --plate:#fcfcfb;
  --c5:#7ddba5; --c5-bg:#123224; --c4:#a5d98a; --c4-bg:#1b2e15; --c3:#f2c46b; --c3-bg:#33270e;
  --c2:#f5a077; --c2-bg:#3a2014; --c1:#f28b8f; --c1-bg:#3b1618;
}
*{box-sizing:border-box}
html{background:var(--paper)}
body{background:var(--paper);color:var(--ink);font:16px/1.65 "Open Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  margin:0;padding-inline:20px;padding-block:0 64px}
.wrap{max-width:1040px;margin:0 auto}
.col{max-width:68ch}
h1,h2,h3{font-family:"Playfair Display",Georgia,"Times New Roman",serif;color:var(--ink-strong);text-wrap:balance;line-height:1.2}
h1{font-size:clamp(2rem,4.5vw,3rem);margin:0 0 .4rem;font-weight:700}
h2{font-size:1.75rem;margin:3.5rem 0 1rem;padding-top:1.25rem;border-top:1px solid var(--rule)}
h3{font-size:1.2rem;margin:2rem 0 .5rem}
a{color:var(--accent-ink);text-underline-offset:3px}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:2px}
.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.9em}
.eyebrow{font-size:.75rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:600}
header{padding-block:48px 8px}
.lede{font-size:1.1rem;color:var(--ink);max-width:62ch}
.meta{color:var(--muted);font-size:.875rem}
nav.toc{margin:1.5rem 0 0;padding:1rem 1.25rem;background:var(--paper-2);border-radius:6px;max-width:68ch}
nav.toc ol{margin:.25rem 0 0;padding-left:1.25rem;columns:2;column-gap:2rem}
nav.toc li{break-inside:avoid;font-size:.95rem}
.estimates{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin:1.5rem 0}
.est{border:1px solid var(--rule);border-radius:6px;padding:18px 20px;background:var(--paper)}
.est .name{font-weight:600;color:var(--ink-strong)}
.est .big{font-family:"Playfair Display",Georgia,serif;font-size:2.1rem;color:var(--ink-strong);line-height:1.1;margin:.4rem 0 .2rem;font-variant-numeric:lining-nums tabular-nums}
.est dl{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;margin:.75rem 0 0;font-size:.9rem}
.est dt{color:var(--muted)} .est dd{margin:0;font-variant-numeric:tabular-nums}
.callout{border-left:3px solid var(--accent);padding:.25rem 0 .25rem 1rem;margin:1.25rem 0;max-width:68ch}
.tablewrap{overflow-x:auto;margin:1rem 0 .5rem;border:1px solid var(--rule);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:.875rem;font-variant-numeric:tabular-nums;min-width:860px}
th,td{padding:9px 10px;text-align:right;border-bottom:1px solid var(--rule);vertical-align:top;white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:.75rem;letter-spacing:.04em;text-transform:uppercase;background:var(--paper-2);text-align:right}
th:first-child,td:first-child,th.l,td.l{text-align:left}
tr:last-child td{border-bottom:0}
.sub{display:block;color:var(--muted);font-size:.8em}
.chip{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.78rem;font-weight:600;line-height:1.6}
.c5{color:var(--c5);background:var(--c5-bg)} .c4{color:var(--c4);background:var(--c4-bg)}
.c3{color:var(--c3);background:var(--c3-bg)} .c2{color:var(--c2);background:var(--c2-bg)} .c1{color:var(--c1);background:var(--c1-bg)}
.day{display:grid;grid-template-columns:minmax(0,1fr);gap:14px;margin:2.25rem 0}
.day .head{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 14px}
.day .head h3{margin:0}
.plates{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
figure{margin:0}
figure img{display:block;width:100%;height:auto;border:1px solid var(--rule);border-radius:4px;background:var(--plate)}
figcaption{font-size:.8rem;color:var(--muted);margin-top:4px}
.track img{max-width:760px}
pre.bulletin{background:var(--paper-2);border:1px solid var(--rule);border-radius:6px;padding:14px 16px;overflow-x:auto;
  font-family:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.8rem;line-height:1.5;color:var(--ink);white-space:pre}
.defs{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px 28px;margin:1rem 0}
.defs div{font-size:.92rem}
.defs b{color:var(--ink-strong)}
ul.tight li{margin:.2rem 0}
footer{margin-top:4rem;color:var(--muted);font-size:.85rem;border-top:1px solid var(--rule);padding-top:1rem}
@media (max-width:560px){nav.toc ol{columns:1} h2{font-size:1.45rem}}
@page{size:Letter;margin:0.6in 0.55in}
@media print{
  :root,:root[data-theme="dark"],:root:not([data-theme="light"]){
    --paper:#ffffff; --paper-2:#f6f7f9; --rule:#dcdcdc; --ink:#333333; --ink-strong:#161a1d; --muted:#5e5e5e;
    --accent:#03A9F4; --accent-ink:#016E9E;
    --c5:#1b7f4b; --c5-bg:#e3f4ea; --c4:#2f7d32; --c4-bg:#edf6e6; --c3:#8a5a00; --c3-bg:#fcf0d6;
    --c2:#a8410f; --c2-bg:#fbe5d8; --c1:#a61d24; --c1-bg:#f9dfe0;
  }
  *{-webkit-print-color-adjust:exact;print-color-adjust:exact}
  body{font-size:10pt;line-height:1.5;padding:0}
  .wrap{max-width:none}
  header{padding-block:0}
  h1{font-size:20pt;margin-bottom:.2rem}
  .lede{font-size:10.5pt;margin:.3rem 0}
  nav.toc{margin:.6rem 0 0;padding:.5rem .9rem}
  nav.toc li{font-size:9pt}
  .est{padding:10px 14px}
  .est .big{font-size:18pt;margin:.2rem 0 0}
  .est dl{margin-top:.4rem;font-size:8.5pt}
  .callout{margin:.6rem 0}
  h2{font-size:17pt;margin:0 0 .6rem;padding-top:0;border-top:0}
  h3{font-size:12pt;margin:1rem 0 .35rem}
  h2,h3,.day .head{break-after:avoid}
  #days,#story,#track,#official,#method{break-before:page}
  a{color:inherit;text-decoration:none}
  nav.toc a{color:var(--accent-ink)}
  .estimates{grid-template-columns:1fr 1fr;gap:10px}
  .est,.callout,figure,pre.bulletin,tr,.defs div{break-inside:avoid}
  .tablewrap{overflow:visible;border:1px solid var(--rule)}
  table{min-width:0;width:100%;font-size:7.5pt}
  th,td{white-space:normal;padding:4px 4px}
  th{font-size:6.5pt;letter-spacing:.02em}
  .chip{font-size:6.5pt;padding:0 5px;margin:1px 0}
  .day{break-inside:avoid;margin:0 0 .5rem;gap:5px}
  .day p{font-size:9pt;line-height:1.4}
  figure img{max-height:3.05in;width:100%;object-fit:contain}
  figcaption{font-size:7pt;margin-top:1px}
  .plates{grid-template-columns:1fr 1fr;gap:8px}
  .track img{max-width:100%;max-height:7.3in;width:auto;margin:0 auto}
  pre.bulletin{white-space:pre-wrap;font-size:7pt}
  footer{margin-top:1.5rem}
}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
"""


def fmt_lat(lat):
    return f"{abs(lat):.2f}°{'S' if lat < 0 else 'N'}"


def fmt_lon(lon):
    lon = lon % 360
    return f"{lon:.2f}°E" if lon <= 180 else f"{360 - lon:.2f}°W"


def render(b, title):
    ctx = b["ctx"]
    dz, oz = ctx["dest_tz"], ctx["origin_tz"]
    dest = ctx["dest"]
    dep = b["depart_fjt"]
    man, smeta = b["manifest"], b["smeta"]
    cons = b["polars"][0]
    wp = route.WAYPOINTS
    esc_ = esc

    def runs(models):
        return ", ".join(f"{esc_(v['label'])} {esc_(v['run'] or 'run time not published')}" for v in models.values())

    h = []
    h.append(f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Open+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow">{esc(boat_label())} passage brief</div>
  <h1>{esc(title)}</h1>
  <p class="lede">{esc(ctx['origin'])} to {esc(dest)}, leaving {dep.astimezone(oz):%A %d %B, %H:%M} {esc(ctx['origin_label'])}. A forecast of each day at sea for a Lagoon 42, built from ensemble and deterministic weather models.</p>
  <p class="meta">Forecast data fetched {esc(man['fetched_utc'])} (UTC). Every time below is local: {esc(ctx['origin_label'])} at departure, {esc(ctx['dest_label'])} at arrival.</p>
  <nav class="toc" aria-label="Contents"><span class="eyebrow">Contents</span><ol>
    <li><a href="#estimates">Trip estimates</a></li>
    <li><a href="#days">Day by day</a></li>
    <li><a href="#sea">The sea</a></li>
    <li><a href="#story">The weather story</a></li>
    <li><a href="#track">Along the track</a></li>
    <li><a href="#official">Official forecast</a></li>
    <li><a href="#method">How this was made</a></li>
  </ol></nav>
</header>

<section id="estimates">
<h2>Trip estimates</h2>
<p class="col">Every member of two weather ensembles (ECMWF, 51 runs, and NOAA GEFS, 31 runs) sailed the route from the departure time. Two boat speed models give a slow and a fast bound. When sailing drops under 4 kt the boat motors at 5.5 kt, burning about 1 US gallon an hour, less into a strong headwind. The range is the 10th to 90th percentile of those 82 passages.</p>
<div class="estimates">""")
    for p in b["polars"]:
        note = (f"Lowest speed in each cell across the chosen polar files, times {polar.CRUISE_FACTOR:g}."
                if p["name"].startswith("Conservative") else
                f"A performance polar{boat_kind()} with the adjustments from Settings: "
                f"{polar.PW_UPWIND:.0%} upwind, {polar.PW_DOWNWIND:.0%} downwind, {polar.PW_NIGHT:.0%} at night.")
        h.append(f"""<div class="est"><div class="name">{esc(p['name'])}</div>
<div class="big">{p['h50']:.0f} h</div><div class="meta">{p['h10']:.0f} to {p['h90']:.0f} h at sea, including {route.EXIT_HOURS:g} h to clear the harbor</div>
<dl><dt>Median arrival</dt><dd>{p['a50'].astimezone(dz):%a %d %b %H:%M}</dd>
<dt>Early to late</dt><dd>{p['a10'].astimezone(dz):%a %d %H:%M} to {p['a90'].astimezone(dz):%a %d %H:%M}</dd>
<dt>Distance</dt><dd>{route.total_nm():.0f} nm offshore</dd>
<dt>Average</dt><dd>{route.total_nm() / (p['h50'] - route.EXIT_HOURS):.1f} kt</dd>
<dt>Point of sail</dt><dd>{screen_pos(p['shares'])}</dd>
<dt>Current</dt><dd>{screen_current(p['current'])}</dd>
<dt>Comfort</dt><dd>mostly {p['c_main']}, worst 3 h {p['c_worst']}</dd>
<dt>Fuel offshore</dt><dd>{p['f50']:.0f} gal, up to {p['f90']:.0f} gal</dd></dl>
<p class="meta" style="margin:.5rem 0 0">{esc(p['c_text'])}</p>
<p class="meta" style="margin:.75rem 0 0">{esc(note)}</p></div>""")
    h.append("</div>")
    if b["arrival"]:
        h.append(f'<p class="callout"><b>Arrival.</b> {esc(b["arrival"])}</p>')
    h.append("</section>")

    # day by day tables
    h.append("""<section id="days"><h2>Day by day</h2>
<p class="col">Each day runs 24 hours from departure. Figures are medians across the ensemble passages. Wind is true wind at 10 m. TWA is the true wind angle to the route. Waves are the higher of the two wave models, with the lowest ratio of period in seconds to height in meters. The limit is """ + f"{windows.LIMITS['period_ratio']:g}" + """, the rule that the period in seconds should be at least the height in feet. Comfort shows the level for most of the day, then the worst level lasting 3 hours or more, with where the seas come from, the strongest apparent wind, the roughest motion and what caused the worst spell.</p>""")
    for p in b["polars"]:
        h.append(f"""<h3>{esc(p['name'])}</h3><div class="tablewrap"><table>
<thead><tr><th>Day</th><th>Wind kt<span class="sub">mean / max</span></th><th>Gust kt</th><th>From</th><th>TWA</th>
<th>Waves<span class="sub">max, period</span></th><th>Boat kt</th><th>Distance nm<span class="sub">10th to 90th</span></th><th>Run nm</th><th>Motor h<span class="sub">fuel gal</span></th><th class="l">Comfort</th></tr></thead><tbody>""")
        for r in p["rows"]:
            partial = "" if r["n"] == r["of"] else f'<span class="sub">{r["n"]} of {r["of"]} still sailing</span>'
            h.append(f"""<tr><td>Day {r['day']}<span class="sub">to {r['end']:%a %d %H:%M}</span></td>
<td>{r['tws']:.0f} / {r['tws_max']:.0f}</td><td>{r['gust']:.0f}</td><td>{r['twd']:03.0f}° {SR.compass(r['twd'])}</td><td>{r['twa']:.0f}°</td>
<td>{r['hs']:.1f} m, {r['per']:.0f} s<span class="sub">T/H {r['ratio']:.1f}</span></td><td>{r['speed']:.1f}</td><td>{r['dist']:.0f}<span class="sub">{r['d10']:.0f} to {r['d90']:.0f}</span>{partial}</td>
<td>{r['total']:.0f}</td><td>{r['motor']:.1f}<span class="sub">{r['fuel']:.0f}</span></td>
<td class="l"><span class="chip {LEVEL_CLASS[r['main']]}">{r['main']}</span> <span class="chip {LEVEL_CLASS[r['worst']]}">{r['worst']}</span>
<span class="sub">seas {comfort.SEA_SECTOR_NAMES[max(r['sea'], key=r['sea'].get)]} {max(r['sea'].values()):.0%}, apparent {r['aws']:.0f} kt, motion {r['accel']:.2f} m/s²</span>
<span class="sub">worst from {r['cause']}</span></td></tr>""")
        h.append("</tbody></table></div>")
    h.append("""<div class="defs">
<div><span class="chip c5">Champagne</span> flat and steady</div><div><span class="chip c4">Easy</span> sleeping fine</div>
<div><span class="chip c3">Coffee</span> one hand to move, still cooking</div><div><span class="chip c2">Rough</span> two hands, no cooking</div>
<div><span class="chip c1">Sick</span> nobody is sleeping</div></div>
<p class="meta col">Comfort is Sereno's five-level scale. Every 15 minutes gets the worse of two ratings. Wind over the deck uses apparent wind: Champagne under 12 kt, Easy under 18, Coffee under 23 (first reef on a reach), Rough under 33. Motion uses how often the boat meets the waves, which depends on where they come from, rated against ISO 2631-1 comfort bands for vertical acceleration: Champagne under 0.15 m/s², Easy under 0.315, Coffee under 0.63, Rough under 1.25. Beam seas and seas on the bow over 1.5 m add extra for catamaran roll and slamming. The same 3 m sea with an 8 second period rates Easy from astern, Coffee on the beam and Rough on the bow. A first guess, not yet checked against a logged passage.</p>
</section>""")

    h.append(sea_section(b))

    # weather story
    h.append(f"""<section id="story"><h2>The weather story</h2>
<p class="col">Deterministic runs from six models, read through the layers a forecaster works with: the surface pressure pattern, 850 hPa wind and temperature advection, dryness at 700 hPa above the trade inversion, troughs at 500 hPa and the jet at 250 hPa. Charts are ECMWF at the middle of each day, with the boat's median position marked. Runs: {runs(smeta['models'])}.</p>
<p class="col meta">{esc(model_reach(smeta, b))}</p>""")
    for d in b["story"]:
        risk = d["risk"]
        plates = "" if not b["have_charts"] else (
            f'<div class="plates"><figure><img src="{img(d["sfc"])}" alt="Surface chart, day {d["day"]}">'
            f'<figcaption>Surface, {d["chart_time"].astimezone(oz):%a %d %H:%M} {esc(ctx["chart_label"])}</figcaption>'
            f'</figure><figure><img src="{img(d["upr"])}" alt="Upper air chart, day {d["day"]}">'
            f'<figcaption>Upper air, same time</figcaption></figure></div>')
        h.append(f"""<article class="day"><div class="head"><h3>Day {d['day']}</h3>
<span class="meta">{d['start'].astimezone(oz):%a %d %H:%M} to {(d['start'] + timedelta(days=1)).astimezone(oz):%a %d %H:%M} {esc(ctx['origin_label'])}</span>
<span class="chip {RISK_CLASS[risk]}">Squall risk {risk.lower()}</span></div>
<p class="col" style="margin:0">{esc(d['text'])}</p>
{plates}
</article>""")
    h.append("</section>")

    track_figure = (f'<figure class="track"><img src="{img(b["track_png"])}" alt="Along-track panels by model"></figure>'
                    if b["have_charts"] else '<p class="meta">Charts are skipped: matplotlib is not installed.</p>')
    h.append(f"""<section id="track"><h2>Along the track</h2>
<p class="col">The same diagnostics read at the boat's median position every 6 hours, one measure per panel. ECMWF, GFS and ICON are in color, the other models in gray. The 24 hour pressure change is used because tropical pressure rises and falls about 2 hPa twice a day with the atmospheric tide.</p>
{track_figure}
<div class="defs">
<div><b>850 hPa wind.</b> About 1,500 m up. A squall downdraft can bring this speed to the surface.</div>
<div><b>Precipitable water.</b> Water in the whole column. Below 35 mm is dry for the tropics, 50 mm or more feeds heavy showers.</div>
<div><b>700 hPa humidity.</b> About 3,000 m up. Very dry air here means the trade inversion is capping showers.</div>
<div><b>K-index.</b> Warmth and moisture low down against cold aloft. Below 15 is stable; 28 or more supports thunderstorms.</div>
<div><b>Temperature advection.</b> Negative means colder air arriving, usually behind a front passing to the south.</div>
<div><b>500 hPa vorticity.</b> Positive is an upper trough, which lifts the air. Negative is a ridge with sinking air.</div>
</div></section>""")

    h.append('<section id="official"><h2>Official forecast</h2>')
    if b["bulletin"]:
        txt = b["bulletin"].read_text(errors="replace").strip()
        h.append(f"""<p class="col">Fiji Meteorological Service is the official high seas authority for this route (equator to 25°S, 160°E to 120°W). This is the latest issue on the NOAA server, header FQPS01 NFFN. NOAA Honolulu stopped its own South Pacific high seas forecast on February 17, 2026.</p>
<pre class="bulletin">{esc(txt)}</pre>""")
    else:
        h.append('<p class="col">No official high seas bulletin is set up for this area. Check the national met service for the waters you cross.</p>')
    h.append("</section>")

    wpt = "".join(f"<li>{esc(n)} <span class='mono'>{fmt_lat(a)} {fmt_lon(o % 360)}</span></li>" for n, a, o in wp)
    h.append(f"""<section id="method"><h2>How this was made</h2>
<div class="col">
<h3>Route</h3><ol class="tight">{wpt}</ol>
<p>{route.total_nm():.0f} nm on this line. Clearing the harbor counts as {route.EXIT_HOURS:g} hours.</p>
<p><b>Estimated guide only. The captain develops the detailed route for every passage.</b></p>
<h3>Weather data</h3>
<ul class="tight">
<li>Wind ensembles: ECMWF ENS and NOAA GEFS through Open-Meteo. Runs: {runs(man['models'])}.</li>
<li>Surface currents: Météo-France SMOC through Open-Meteo, 8 km, hourly, 10 days, tides included. The boat holds the line, so current along the line changes speed over ground and a cross current costs a little speed.</li>
<li>Waves: the ECMWF and GFS wave models, single runs. Open-Meteo's wave ensembles returned no data.</li>
<li>Synoptic grid: {smeta.get('step', 2.5):g}° from {fmt_lat(smeta['lats'][0])} to {fmt_lat(smeta['lats'][-1])}, {fmt_lon(smeta['lons'][0])} to {fmt_lon(smeta['lons'][-1])}, every 6 hours. Surface and column values at the boat come from each model at 0.25° route points, because a coarse grid point near an island can sit on land.</li>
<li>Open-Meteo's wind values were checked against raw GRIB from ECMWF Open Data and NOAA NOMADS on September 15, 2026, and agreed within 0.1 kt and 1°.</li>
</ul>
<h3>What isn't in it</h3>
<ul class="tight">
<li>Individual squalls. The squall call is a risk from the column, not a squall forecast.</li>
<li>Speed lost to waves, and swell in the anchorage.</li>
<li>Forecast skill fades past about 5 days. Rerun daily before departure.</li>
</ul>
</div></section>
<footer>Generated by <span class="mono">report_html.py</span> in <span class="mono">passage_weather</span>. Estimated guide only. The captain develops the detailed route for every passage. Check official forecasts before sailing.</footer>
</div>""")
    return "\n".join(h)


def main():
    route_name = sys.argv[1]
    depart_fjt = datetime.strptime(sys.argv[2], "%Y-%m-%d %H:%M").replace(tzinfo=windows.FJT)
    title = sys.argv[3] if len(sys.argv) > 3 else f"Nadi to {DEST_NAME[route_name]}, {depart_fjt:%-d %B}"
    b = build(route_name, depart_fjt)
    out = HERE / "output" / f"brief_{route_name}_{depart_fjt:%Y%m%dT%H%M}.html"
    out.write_text(render(b, title))
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")
    pdf = write_pdf(out)
    if pdf:
        print(pdf, f"{pdf.stat().st_size / 1e6:.1f} MB")


def write_pdf(html_path):
    """Print the brief to a Letter PDF with headless Brave or Chromium, using the page's print styles."""
    import shutil
    import subprocess
    browser = next((b for b in ("brave-browser", "chromium", "chromium-browser", "google-chrome") if shutil.which(b)), None)
    if not browser:
        print("no Chromium-based browser found, skipping PDF")
        return None
    pdf = html_path.with_suffix(".pdf")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=15000", f"--print-to-pdf={pdf}", html_path.resolve().as_uri()],
                   check=True, capture_output=True, timeout=180)
    return pdf


if __name__ == "__main__":
    main()
