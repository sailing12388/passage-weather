"""Full passage brief for a trip's watched departure.

Reuses the trip's latest ensemble fetch if it's under 6 hours old, fetches the
synoptic grid around the route, and adds Fiji Met's high seas bulletin when the
route lies inside its area (equator to 25S, 160E to 120W).
"""
from datetime import datetime, timedelta, timezone

import fetch
import report_html
import synoptic
import trip as T

BULLETIN_URL = "https://tgftp.nws.noaa.gov/data/raw/fq/fqps01.nffn..txt"
FRESH = timedelta(hours=6)


def _fresh(dirs, fresh=FRESH):
    now = datetime.now(timezone.utc)
    for d in sorted(dirs, reverse=True):
        try:
            stamp = datetime.strptime(d.name[:14], "%Y%m%dT%H%MZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if now - stamp < fresh and (d / "manifest.json").exists():
            return d
    return None


def in_fiji_area(t):
    for p in (t.origin, t.dest):
        lon = p["lon"] % 360
        if not (-25 <= p["lat"] <= 0 and 160 <= lon <= 240):
            return False
    return True


def fetch_bulletin(t):
    import requests
    r = requests.get(BULLETIN_URL, timeout=60)
    r.raise_for_status()
    out = t.dir / "official" / f"FQPS01_NFFN_{datetime.now(timezone.utc):%Y%m%dT%H%MZ}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(r.text)
    return out


def build_for_trip(t, fresh=FRESH):
    """fresh: reuse downloads younger than this. An Update now press passes a shorter window."""
    t.use_route()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    runs = list((t.dir / "runs").glob("*")) if (t.dir / "runs").exists() else []
    run_dir = _fresh(runs, fresh) or fetch.fetch_into(t.dir / "runs" / stamp, verbose=False)
    syns = list((t.dir / "synoptic").glob("*")) if (t.dir / "synoptic").exists() else []
    syn_dir = _fresh(syns, fresh) or synoptic.fetch((t.route_key(),), out=t.dir / "synoptic" / stamp)
    bulletin = None
    if in_fiji_area(t):
        try:
            bulletin = fetch_bulletin(t)
        except Exception as e:
            print(f"Fiji Met bulletin not fetched: {e}")
    depart = t.watch_time()
    ctx = dict(trip=t.slug, origin=t.origin["name"], dest=t.dest["name"], origin_tz=t.origin_tz, dest_tz=t.dest_tz,
               origin_label=f"{T.tz_abbrev(depart, t.origin_tz)} ({t.origin['tz']})",
               dest_label=f"{T.tz_abbrev(depart, t.dest_tz)} ({t.dest['tz']})",
               chart_label=T.tz_abbrev(depart, t.origin_tz))
    b = report_html.build(t.route_key(), depart, run_dir=run_dir, syn_dir=syn_dir, bulletin=bulletin, ctx=ctx)
    title = f"{t.title()}, {depart:%-d %B %H:%M}"
    out = t.dir / "brief.html"
    out.write_text(report_html.render(b, title))
    pdf = report_html.write_pdf(out)
    return out, pdf
