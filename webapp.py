"""Local web GUI: plan trips, pick a day, read the brief, set limits and polars.

No web framework and no server to install: Python's own http.server, with pages built here.
Nothing runs on a schedule; forecasts are fetched when you press a button.

    python3 webapp.py            # http://127.0.0.1:8765
    python3 webapp.py --lan      # also reachable from the boat network
"""
import argparse
import html
import json
import mimetypes
import threading
import traceback
import webbrowser
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import polar
import report_html
import settings
import trip as T

HERE = Path(__file__).parent
PORT = 8765


def esc(x):
    return html.escape(str(x if x is not None else ""))


# ---------------------------------------------------------------- jobs

class Job:
    """One long task at a time (fetching, screening, building a brief), with its log."""

    def __init__(self):
        self.name = None
        self.lines = []
        self.started = None
        self.error = None
        self.thread = None
        self.lock = threading.Lock()

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def log(self, line):
        with self.lock:
            self.lines.append(f"{datetime.now():%H:%M:%S} {line}")
            self.lines[:] = self.lines[-200:]

    def start(self, name, fn):
        if self.running:
            return False
        self.name, self.lines, self.started, self.error = name, [], datetime.now(), None

        def run():
            try:
                fn(self.log)
                self.log("done")
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                self.log("FAILED " + self.error)
                self.log(traceback.format_exc().splitlines()[-1])
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return True

    def state(self):
        return {"name": self.name, "running": self.running, "error": self.error,
                "started": self.started.strftime("%H:%M:%S") if self.started else None,
                "lines": self.lines[-12:]}


JOB = Job()


# ---------------------------------------------------------------- actions

def act_screen(slug):
    def run(log):
        import screen
        t = T.Trip.load(slug)
        log(f"fetching forecasts along {t.title()}")
        out, pdf, days = screen.screen(t)
        log(screen.recommendation(T.Trip.load(slug), days))
    return run


def act_report(slug):
    def run(log):
        import full_report
        t = T.Trip.load(slug)
        if not t.watch_depart:
            raise ValueError("pick a day first")
        log("building the full brief, a few minutes")
        out, pdf = full_report.build_for_trip(t)
        log(f"brief ready: {Path(out).name}")
    return run


def act_new(origin, dest, earliest, days, exit_hours):
    def run(log):
        import screen
        o, onote = T.resolve_place(origin)
        d, dnote = T.resolve_place(dest)
        log(f"from {o['name']} {o['lat']}, {o['lon']} ({onote[:60]})")
        log(f"to {d['name']} {d['lat']}, {d['lon']} ({dnote[:60]})")
        slug = base = T.slugify(f"{o['name']}-{d['name']}-{earliest}")
        n = 2
        while (T.TRIPS / slug / "trip.json").exists():
            slug, n = f"{base}-{n}", n + 1
        t = T.Trip(slug=slug, origin=o, dest=d, earliest=earliest, days=int(days), exit_hours=float(exit_hours))
        t.save()
        log(f"screening {t.title()}")
        _, _, day_list = screen.screen(t)
        log(screen.recommendation(T.Trip.load(slug), day_list))
    return run


# ---------------------------------------------------------------- pages

CSS = report_html.CSS + """
body{padding-block:24px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:.75rem 0}
.btn{display:inline-block;border:1px solid var(--rule);background:var(--paper-2);color:var(--ink-strong);
  border-radius:6px;padding:7px 14px;font:inherit;font-size:.9rem;cursor:pointer;text-decoration:none}
.btn:hover{border-color:var(--accent)}
.btn.go{background:var(--accent);border-color:var(--accent);color:#fff}
.btn.quiet{color:var(--muted)}
form.inline{display:inline}
label{display:block;font-size:.8rem;color:var(--muted);margin-bottom:2px}
input,select{font:inherit;padding:6px 8px;border:1px solid var(--rule);border-radius:5px;background:var(--paper);
  color:var(--ink);max-width:100%}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;align-items:end}
.card2{border:1px solid var(--rule);border-radius:8px;padding:16px 18px;margin:1rem 0;background:var(--paper)}
.job{background:var(--paper-2);border-radius:6px;padding:10px 14px;font-family:"IBM Plex Mono",monospace;
  font-size:.78rem;white-space:pre-wrap;line-height:1.5;max-height:200px;overflow:auto}
.pill{font-size:.75rem;border-radius:999px;padding:1px 9px;background:var(--paper-2);color:var(--muted)}
iframe{width:100%;height:78vh;border:1px solid var(--rule);border-radius:8px;background:#fff}
nav.top{display:flex;gap:16px;margin-bottom:1rem;font-size:.9rem}
.err{color:var(--c1);font-size:.85rem}
"""

PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Open+Sans:wght@400;600&family=IBM+Plex+Mono&display=swap">
<style>{css}</style>
<div class="wrap">
<nav class="top"><a href="/">Trips</a><a href="/settings">Boat and limits</a>
<span class="pill" id="jobpill"></span></nav>
{body}
<div class="job" id="joblog" hidden></div>
</div>
<script>
async function poll() {{
  try {{
    const s = await (await fetch("/job.json", {{cache: "no-store"}})).json();
    const pill = document.getElementById("jobpill"), log = document.getElementById("joblog");
    if (s.running || s.lines.length) {{
      pill.textContent = (s.running ? "Working: " : "Last: ") + (s.name || "");
      log.hidden = false;
      log.textContent = s.lines.join("\\n") + (s.error ? "\\n" + s.error : "");
      log.scrollTop = log.scrollHeight;
    }}
    if (window.wasRunning && !s.running) location.reload();
    window.wasRunning = s.running;
  }} catch (e) {{}}
  setTimeout(poll, 2000);
}}
poll();
</script>
"""


def page(title, body):
    return PAGE.format(title=esc(title), css=CSS, body=body)


def home():
    s = settings.get()
    trips = T.Trip.all()
    rows = []
    for t in sorted(trips, key=lambda x: (x.status == "done", x.earliest), reverse=False):
        state = {"planning": "Screening", "watching": f"Watching {t.watch_depart}", "done": "Done"}[t.status]
        rows.append(f"""<tr><td><a href="/trip/{esc(t.slug)}"><b>{esc(t.title())}</b></a>
<span class="sub">from {esc(t.earliest)}, {t.days} days</span></td><td>{esc(state)}</td>
<td class="l">{esc(t.last_headline or '')}</td></tr>""")
    missing = not settings.polar_files("sources")
    banner = ("" if not missing else
              '<div class="card2"><b>No polar yet.</b> Put your boat\'s polar file (.pol or .csv, the OpenCPN or '
              'qtVlm format) in the app folder, then choose it under <a href="/settings">Boat and limits</a>. '
              'Until then the speeds are meaningless.</div>')
    table = (f"""<div class="tablewrap"><table><thead><tr><th>Trip</th><th>Status</th><th class="l">Latest</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>""" if rows else '<p class="meta">No trips yet.</p>')
    today = datetime.now().date().isoformat()
    return page("Passage weather", f"""
<header><div class="eyebrow">Passage weather</div><h1>Trips</h1>
<p class="lede">Plan a passage, screen a few days, pick one, then build the full brief. Forecasts are fetched when you press a button.</p></header>
{banner}
{table}
<div class="card2"><h3 style="margin-top:0">New trip</h3>
<form method="post" action="/new"><div class="grid">
<div><label>From</label><input name="origin" placeholder="Denarau, or 17.77 S 177.37 E" required></div>
<div><label>To</label><input name="dest" placeholder="Port Resolution" required></div>
<div><label>Earliest departure</label><input type="date" name="earliest" value="{today}" required></div>
<div><label>Days to screen</label><input type="number" name="days" min="1" max="7" value="{s['trip_defaults']['days']}"></div>
<div><label>Hours to clear the harbor</label><input type="number" step="0.5" name="exit_hours" value="{s['trip_defaults']['exit_hours']}"></div>
<div><button class="btn go" type="submit">Plan and screen</button></div>
</div></form>
<p class="meta">Places are looked up when you type them. A name that can't be found, or the wrong match, can be given as a position: <span class="mono">19.53 S, 169.50 E</span>.</p></div>
""")


def trip_page(slug):
    t = T.Trip.load(slug)
    opts, brief = t.dir / "options.html", t.dir / "brief.html"
    hist = json.loads((t.dir / "history.json").read_text()) if (t.dir / "history.json").exists() else []
    days = sorted((hist[-1]["days"] if hist else {}).items())
    picker = "".join(
        f'<option value="{k} {v["best"]}"{" selected" if t.watch_depart == f"{k} {v['best']}" else ""}>'
        f'{datetime.strptime(k, "%Y-%m-%d"):%a %d %b} {v["best"]} · {v["verdict"]} {v["risk"]:.0%}</option>'
        for k, v in days if v)
    watch_form = (f"""<form method="post" action="/trip/{esc(slug)}/watch" class="inline">
<select name="depart">{picker}</select> <button class="btn" type="submit">Watch this day</button></form>""" if picker else "")
    brief_links = (f"""<a class="btn" href="/file/{esc(slug)}/brief.html" target="_blank">Full brief</a>
<a class="btn quiet" href="/file/{esc(slug)}/brief.pdf" target="_blank">PDF</a>""" if brief.exists() else "")
    body = f"""
<header><div class="eyebrow">{esc({"planning": "Screening", "watching": "Watching " + (t.watch_depart or ""), "done": "Done"}[t.status])}</div>
<h1>{esc(t.title())}</h1>
<p class="meta">Leaving any day from {esc(t.earliest)} for {t.days} days. {esc(t.last_headline or '')}</p></header>
<div class="bar">
<form method="post" action="/trip/{esc(slug)}/screen" class="inline"><button class="btn go" type="submit">Update forecast</button></form>
{watch_form}
<form method="post" action="/trip/{esc(slug)}/report" class="inline"><button class="btn" type="submit">Build full brief</button></form>
{brief_links}
<a class="btn quiet" href="/file/{esc(slug)}/options.pdf" target="_blank">Options PDF</a>
<form method="post" action="/trip/{esc(slug)}/done" class="inline"><button class="btn quiet" type="submit">Stop watching</button></form>
</div>
{f'<iframe src="/file/{esc(slug)}/options.html"></iframe>' if opts.exists() else '<p class="meta">No screening yet. Press Update forecast.</p>'}
"""
    return page(t.title(), body)


def settings_page(msg=""):
    s = settings.get()
    files = settings.available_polars()
    checks = "".join(
        f'<label style="display:flex;gap:8px;align-items:center;font-size:.9rem;color:var(--ink)">'
        f'<input type="checkbox" name="sources" value="{esc(f)}"{" checked" if f in s["polar"]["sources"] else ""}>'
        f'{esc(f)}</label>' for f in files)
    perf = "".join(f'<option value="{esc(v)}"{" selected" if v == s["polar"]["performance_file"] else ""}>{esc(lab)}</option>'
                   for v, lab in [("", "(none)")] + [(f, f) for f in files])
    grid = polar.conservative_polar()
    head = "".join(f"<th>{w}</th>" for w in polar.TWS)
    rows = "".join(f"<tr><td>{a}°</td>" + "".join(f"<td>{grid[i][j]:.1f}</td>" for j in range(len(polar.TWS))) + "</tr>"
                   for i, a in enumerate(polar.TWA))
    L = s["limits"]
    num = lambda name, value, step="0.1": f'<input type="number" step="{step}" name="{name}" value="{value}">'
    return page("Boat and limits", f"""
<header><div class="eyebrow">Settings</div><h1>Boat and limits</h1>
{f'<p class="meta">{esc(msg)}</p>' if msg else ''}</header>
<form method="post" action="/settings">
<div class="card2"><h3 style="margin-top:0">Boat</h3>
<p class="meta">The name shown at the top of the screening and brief pages. Both are optional.</p>
<div class="grid">
<div><label>Name</label><input name="boat_name" value="{esc(s['boat']['name'])}" placeholder="S/V Your Boat"></div>
<div><label>Type</label><input name="boat_kind" value="{esc(s['boat']['kind'])}" placeholder="Lagoon 42"></div>
</div></div>

<div class="card2"><h3 style="margin-top:0">Wind and waves</h3>
<p class="meta">A tier counts when more than {s['tier_share']:.0%} of the ensemble passages reach it. Gusts are estimated as {s['gust_factor']} times the sustained wind.</p>
<div class="grid">
<div><label>Wind warning, kt</label>{num("tws_warn", L["tws_warn"], "1")}</div>
<div><label>Wind no-go, kt</label>{num("tws_no", L["tws_no"], "1")}</div>
<div><label>Gust warning, kt</label>{num("gust_warn", L["gust_warn"], "1")}</div>
<div><label>Gust no-go, kt</label>{num("gust_no", L["gust_no"], "1")}</div>
<div><label>Waves forward of the beam, m</label>{num("hs", L["hs"])}</div>
<div><label>Waves any direction, m</label>{num("hs_no", L["hs_no"])}</div>
<div><label>Feet rule: clear at (period s per ft)</label>{num("feet_clear", L["feet_clear"], "0.1")}</div>
<div><label>Feet rule: warn under</label>{num("feet_warn", L["feet_warn"], "0.1")}</div>
<div><label>Feet rule: no-go under</label>{num("feet_no", L["feet_no"], "0.1")}</div>
<div><label>Feet rule ignored below, m</label>{num("feet_min_hs", L["feet_min_hs"], "0.1")}</div>
<div><label>Big seas for the 2:1 to 3:1 band, m</label>{num("feet_big_hs", L["feet_big_hs"], "0.1")}</div>
<div><label>Steepness: warn steeper than 1 in</label>{num("steep_warn_n", L["steep_warn_n"], "1")}</div>
<div><label>Steepness: no-go steeper than 1 in</label>{num("steep_no_n", L["steep_no_n"], "1")}</div>
<div><label>Steepness ignored below, m</label>{num("steep_min_hs", L["steep_min_hs"], "0.1")}</div>
<div><label>Tier share</label>{num("tier_share", s["tier_share"], "0.01")}</div>
<div><label>Gust factor</label>{num("gust_factor", s["gust_factor"], "0.01")}</div>
</div></div>

<div class="card2"><h3 style="margin-top:0">Engine</h3><div class="grid">
<div><label>Motor when sailing drops under, kt</label>{num("below_kt", s["motor"]["below_kt"])}</div>
<div><label>Speed under power, kt</label>{num("speed_kt", s["motor"]["speed_kt"])}</div>
<div><label>Fuel, US gallons an hour</label>{num("fuel_gph", s["motor"]["fuel_gph"])}</div>
</div></div>

<div class="card2"><h3 style="margin-top:0">Polars</h3>
<p class="meta">The conservative polar is the lowest speed in each cell across the files you tick, times the cruise factor. The performance polar is the optional faster comparison, with its own adjustments.</p>
<div class="grid"><div><label>Files in the conservative envelope</label>{checks}</div>
<div><label>Cruise factor</label>{num("cruise_factor", s["polar"]["cruise_factor"], "0.01")}</div>
<div><label>Performance polar</label><select name="performance_file">{perf}</select></div>
<div><label>Upwind</label>{num("perf_upwind", s["polar"]["performance"]["upwind"], "0.01")}</div>
<div><label>Downwind</label>{num("perf_downwind", s["polar"]["performance"]["downwind"], "0.01")}</div>
<div><label>At night</label>{num("perf_night", s["polar"]["performance"]["night"], "0.01")}</div>
</div>
<p class="meta" style="margin-top:1rem">Drop new polar files (.pol or .csv) into <span class="mono">{esc(HERE)}</span> and reload this page.</p>
</div>
<div class="bar"><button class="btn go" type="submit">Save</button></div>
</form>
<div class="card2"><h3 style="margin-top:0">Conservative polar now in use</h3>
<div class="tablewrap"><table><thead><tr><th>TWA</th>{head}</tr></thead><tbody>{rows}</tbody></table></div>
<p class="meta">Boat speed in knots by true wind angle and wind speed, after the cruise factor.</p></div>
""")


# ---------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, code=200, ctype="text/html; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, to):
        self.send_response(303)
        self.send_header("Location", to)
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/":
                return self._send(home())
            if path == "/job.json":
                return self._send(json.dumps(JOB.state()), ctype="application/json")
            if path == "/settings":
                return self._send(settings_page(parse_qs(urlparse(self.path).query).get("msg", [""])[0]))
            if path.startswith("/trip/"):
                return self._send(trip_page(path.split("/")[2]))
            if path.startswith("/file/"):
                _, _, slug, name = path.split("/", 3)
                f = T.TRIPS / slug / name
                if not f.exists() or ".." in name:
                    return self._send("not found", 404, "text/plain")
                ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
                return self._send(f.read_bytes(), ctype=ctype)
            return self._send("not found", 404, "text/plain")
        except Exception as e:
            return self._send(f"<pre>{esc(traceback.format_exc())}</pre>", 500)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        fields = parse_qs(self.rfile.read(length).decode())      # read the body once
        form = {k: v[0] for k, v in fields.items()}
        try:
            if path == "/new":
                JOB.start(f"new trip {form['origin']} to {form['dest']}",
                          act_new(form["origin"], form["dest"], form["earliest"],
                                  form.get("days", 4), form.get("exit_hours", 1)))
                return self._redirect("/")
            if path == "/settings":
                return self._save_settings(fields, form)
            if path.startswith("/trip/"):
                _, _, slug, action = path.split("/", 3)
                t = T.Trip.load(slug)
                if action == "screen":
                    JOB.start(f"updating {t.title()}", act_screen(slug))
                elif action == "report":
                    JOB.start(f"full brief for {t.title()}", act_report(slug))
                elif action == "watch":
                    t.watch_depart, t.status = form["depart"], "watching"
                    t.save()
                elif action == "done":
                    t.status, t.watch_depart = "planning", None
                    t.save()
                return self._redirect(f"/trip/{slug}")
            return self._send("not found", 404, "text/plain")
        except Exception as e:
            return self._send(f"<pre>{esc(traceback.format_exc())}</pre>", 500)

    def _save_settings(self, fields, form):
        s = settings.get()
        sources = fields.get("sources")          # checkboxes: every ticked value
        values = {
            "limits": {k: float(form[k]) for k in ("tws_warn", "tws_no", "gust_warn", "gust_no", "hs", "hs_no",
                                                   "feet_clear", "feet_warn", "feet_no", "feet_min_hs",
                                                   "feet_big_hs", "steep_warn_n", "steep_no_n",
                                                   "steep_min_hs")},
            "tier_share": float(form["tier_share"]), "gust_factor": float(form["gust_factor"]),
            "motor": {k: float(form[k]) for k in ("below_kt", "speed_kt", "fuel_gph")},
            "polar": {"sources": sources if sources is not None else s["polar"]["sources"],
                      "cruise_factor": float(form["cruise_factor"]),
                      "performance_file": form.get("performance_file", s["polar"]["performance_file"]),
                      "performance": {"upwind": float(form["perf_upwind"]), "downwind": float(form["perf_downwind"]),
                                      "night": float(form["perf_night"])}},
            "boat": {"name": form.get("boat_name", "").strip(), "kind": form.get("boat_kind", "").strip()},
        }
        settings.save({**s, **values})
        _reload_modules()
        return self._redirect("/settings?msg=Saved.+New+settings+apply+to+the+next+update.")


def _reload_modules():
    import importlib
    import windows
    settings.reload()
    importlib.reload(polar)
    importlib.reload(windows)


def serve(port=PORT, lan=False, open_browser=True):
    host = "0.0.0.0" if lan else "127.0.0.1"
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Passage weather on {url}" + (f" (and on the boat network, port {port})" if lan else ""))
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--lan", action="store_true", help="listen on the boat network too")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    serve(a.port, a.lan, not a.no_browser)


if __name__ == "__main__":
    main()
