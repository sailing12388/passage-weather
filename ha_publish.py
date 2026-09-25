"""Put trip pages on Home Assistant and send an alert when the picture changes.

Pages are copied to /homeassistant/www/passage/ on the HA host over SSH and served at
the address set in Settings. Anything under /local is readable without logging in by
anyone on that network.

The alert goes through HA's own API from the SSH session (the Supervisor token that
session already has), so no token is stored here and HA's configuration isn't touched.
Alerts go out only when the best day or its verdict changes, or a watched departure's
verdict changes, to the notify service named in Settings.
"""
import html
import json
import subprocess
from datetime import datetime, timezone

import report_html
import settings
import trip as T

_HA = settings.get()["home_assistant"]
HA_SSH = _HA["ssh"]
HA_WWW = _HA["www_dir"]
HA_URL = _HA["url"]
NOTIFY_SERVICE = _HA["notify_service"]
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HA_SSH]


def _guard():
    """Tests set PASSAGE_NO_HA so nothing they run can reach the live Home Assistant."""
    import os
    if os.environ.get("PASSAGE_NO_HA"):
        raise RuntimeError("Home Assistant calls are disabled (PASSAGE_NO_HA is set)")


def _run(cmd, **kw):
    _guard()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} failed: {r.stderr.strip()[-300:]}")
    return r.stdout


def _put(local, remote):
    """Stream a file over SSH. HA's SSH add-on has no SFTP server, so scp fails there."""
    _guard()
    with open(local, "rb") as f:
        r = subprocess.run(SSH + [f"cat > '{remote}.tmp' && mv '{remote}.tmp' '{remote}'"], stdin=f,
                           capture_output=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"copy {local} failed: {r.stderr.decode(errors='replace').strip()[-300:]}")


def copy_pages(t):
    """Copy the trip's pages to HA. Returns the options page URL."""
    _run(SSH + [f"mkdir -p {HA_WWW}/{t.slug}"])
    files = [p for p in (t.dir / "options.html", t.dir / "options.pdf", t.dir / "brief.html", t.dir / "brief.pdf") if p.exists()]
    for f in files:
        _put(f, f"{HA_WWW}/{t.slug}/{f.name}")
    v = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")          # new URL per publish, past HA's 31-day cache
    if t.status == "watching" and (t.dir / "brief.html").exists():
        return f"{HA_URL}/{t.slug}/brief.html?v={v}"
    return f"{HA_URL}/{t.slug}/options.html?v={v}"


SHELL_VERSION = 3      # bump when the index shell changes; the dashboard card loads index.html?v=SHELL_VERSION


def write_index():
    """HA serves /local with a 31-day cache, so index.html is a fixed shell that fetches trips.json
    with no-store, and every link carries a version stamp so a new publish is a new URL."""
    trips = [t for t in T.Trip.all() if t.status != "done"] + [t for t in T.Trip.all() if t.status == "done"][-5:]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    data = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "trips": []}
    for t in trips:
        data["trips"].append(dict(
            slug=t.slug, title=t.title(), earliest=t.earliest, days=t.days, status=t.status,
            watch=t.watch_depart, headline=t.last_headline or "", v=stamp,
            brief=(t.dir / "brief.html").exists() and t.status == "watching"))
    local_json = T.TRIPS / "trips.json"
    local_json.write_text(json.dumps(data, indent=1))
    _put(local_json, f"{HA_WWW}/trips.json")
    shell = T.TRIPS / "index.html"
    shell.write_text(INDEX_SHELL.replace("__CSS__", report_html.CSS)
                                .replace("__BOAT__", html.escape(report_html.boat_label())))
    _put(shell, f"{HA_WWW}/index.html")
    return f"{HA_URL}/index.html?v={SHELL_VERSION}"


INDEX_SHELL = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Passage weather</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Open+Sans:wght@400;600&display=swap">
<style>__CSS__ table{min-width:0} td{white-space:normal} .pages a{margin-right:.6rem}</style>
<div class="wrap"><header><div class="eyebrow">__BOAT__</div><h1>Passage weather</h1>
<p class="meta" id="updated">Loading…</p></header>
<div class="tablewrap"><table><thead><tr><th>Trip</th><th>Status</th><th class="l">Latest</th><th class="l">Pages</th></tr></thead>
<tbody id="rows"><tr><td colspan="4">Loading…</td></tr></tbody></table></div>
<footer>Estimated guide only. The captain develops the detailed route for every passage. Pages open in this tab; use Back to return.</footer></div>
<script>
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
async function load() {
  try {
    const r = await fetch("trips.json?t=" + Date.now(), {cache: "no-store"});
    const d = await r.json();
    document.getElementById("updated").textContent = "Updated " + d.updated + ". Refreshes about 08:30 and 20:30 Fiji time.";
    // target _top: the dashboard's Webpage card is a sandboxed frame, and a PDF opened from inside it is
    // blocked. The card allows top navigation, so the page replaces the Home Assistant tab instead.
    const link = (t, file, label) => `<a href="${t.slug}/${file}?v=${t.v}" target="_top">${label}</a>`;
    document.getElementById("rows").innerHTML = d.trips.length ? d.trips.map(t => {
      const status = t.status === "watching" ? "Watching " + esc(t.watch) : t.status === "planning" ? "Screening" : "Done";
      const pages = [link(t, "options.html", "Options"), link(t, "options.pdf", "PDF")]
        .concat(t.brief ? [link(t, "brief.html", "Full brief"), link(t, "brief.pdf", "PDF")] : []).join(" ");
      return `<tr><td><b>${esc(t.title)}</b><span class="sub">from ${esc(t.earliest)}, ${t.days} days</span></td>
        <td>${status}</td><td class="l">${esc(t.headline)}</td><td class="l pages">${pages}</td></tr>`;
    }).join("") : '<tr><td colspan="4">No trips yet.</td></tr>';
  } catch (e) {
    document.getElementById("updated").textContent = "Couldn't load the trip list: " + e;
  }
}
load();
setInterval(load, 300000);
</script>
"""


def notify(title, message, url):
    """Send through HA's notify service. Returns HA's HTTP status; delivery to the phone isn't confirmed."""
    payload = json.dumps({"title": title, "message": message,
                          "data": {"clickAction": url, "url": url, "tag": "passage-weather"}})
    remote = ("curl -s -o /dev/null -w '%{http_code}' -m 20 -X POST "
              "-H \"Authorization: Bearer $SUPERVISOR_TOKEN\" -H 'Content-Type: application/json' "
              f"--data-binary @- http://supervisor/core/api/services/notify/{NOTIFY_SERVICE}")
    return _run(SSH + [remote], input=payload).strip()


def signature(t, days):
    """What the alert rule compares: best day and verdict, plus the watched slot's verdict."""
    import screen
    scored = [d for d in days if "best" in d]
    if not scored:
        return "no-forecast", "The forecast doesn't reach this window yet."
    best = min(scored, key=lambda d: screen.best_key(d["best"]))
    b = best["best"]
    when = b["depart"].astimezone(t.origin_tz)
    sig = f"best={when:%Y-%m-%d}:{b['verdict']}"
    headline = f"Best {when:%a %d %b} {when:%H:%M}, {b['verdict']}. {screen.verdict_reason(b)}"
    w = t.watch_time()
    if w:
        slots = [s for d in days for s in d["slots"]]
        if slots:
            s = min(slots, key=lambda s: abs((s["depart"] - w).total_seconds()))
            if abs((s["depart"] - w).total_seconds()) <= 2 * 3600:
                sig += f"|watch={s['verdict']}"
                headline = f"Watched {w:%a %d %H:%M}: {s['verdict']}. {screen.verdict_reason(s)} " + headline
            else:
                sig += "|watch=beyond-forecast"
                headline = f"Watched {w:%a %d %H:%M}: beyond the forecast. " + headline
    return sig, headline


def publish(t, days, force_alert=False, alert_prefix="", always_alert=False):
    """Copy pages, refresh the index, alert if the signature changed. Returns a log line."""
    url = copy_pages(t)
    sig, headline = signature(t, days)
    changed = sig != (t.last_alert or "")
    t.last_headline = headline
    msg = f"{t.slug}: pages on HA"
    if changed or force_alert or always_alert:
        code = notify(f"Passage: {t.title()}", alert_prefix + headline, url)
        msg += f", alert sent to {NOTIFY_SERVICE} (HA replied {code})"
        if code.startswith("2"):
            t.last_alert = sig
    else:
        msg += ", no change, no alert"
    t.save()
    write_index()
    try:
        update_select()
    except Exception as e:
        msg += f", dropdown update FAILED {e}"
    return msg


# ------------------------------------------------------------ the "day to watch" dropdown

SELECT = "input_select.passage_watch"
NOT_WATCHING = "Not watching"
STATE_FILE = T.TRIPS / "ha_watch.json"
LOCK_FILE = T.TRIPS / ".ha.lock"


class Lock:
    """One process at a time touches trips and the dropdown. blocking=False returns None if busy."""

    def __init__(self, blocking=True):
        self.blocking = blocking

    def __enter__(self):
        import fcntl
        LOCK_FILE.parent.mkdir(exist_ok=True)
        self.f = open(LOCK_FILE, "w")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | (0 if self.blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            self.f.close()
            return None
        return self

    def __exit__(self, *exc):
        if not self.f.closed:
            self.f.close()


def ha_api(method, path, payload=None):
    """Call HA's REST API from the SSH session. Returns parsed JSON (or None)."""
    remote = (f"curl -s -m 30 -X {method} -H \"Authorization: Bearer $SUPERVISOR_TOKEN\" "
              f"-H 'Content-Type: application/json' "
              + ("--data-binary @- " if payload is not None else "")
              + f"http://supervisor/core/api/{path}")
    out = _run(SSH + [remote], input=json.dumps(payload) if payload is not None else None)
    try:
        return json.loads(out) if out.strip() else None
    except json.JSONDecodeError:
        raise RuntimeError(f"HA API {path}: {out[:200]}")


def _label(t, when, verdict, risk, multi):
    prefix = f"{t.dest['name']} · " if multi else ""
    tail = f"{verdict} {risk:.0%}" if verdict else "watching"
    return f"{prefix}{when:%a %d %b %H:%M} · {tail}"


def update_select():
    """Fill the dropdown from each active trip's latest screening, keeping the watched departure selected."""
    active = [t for t in T.Trip.all() if t.status in ("planning", "watching")]
    multi = len(active) > 1
    mapping, expected = {NOT_WATCHING: None}, NOT_WATCHING
    for t in active:
        hist = json.loads((t.dir / "history.json").read_text()) if (t.dir / "history.json").exists() else []
        watched = t.watch_time()
        seen_watch = False
        for day, v in (hist[-1]["days"].items() if hist else []):
            if not v:
                continue
            when = datetime.strptime(f"{day} {v['best']}", "%Y-%m-%d %H:%M").replace(tzinfo=t.origin_tz)
            lab = _label(t, when, v["verdict"], v["risk"], multi)
            mapping[lab] = [t.slug, when.strftime("%Y-%m-%d %H:%M")]
            if watched and when == watched:
                expected, seen_watch = lab, True
        if watched and not seen_watch:
            lab = _label(t, watched, None, None, multi)
            mapping[lab] = [t.slug, t.watch_depart]
            expected = lab
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    # A pick the user made that the 5-minute check hasn't handled yet must survive the refill. Labels carry the
    # verdict and risk, which change every screening, so match the pick by trip and departure time.
    shown = expected
    live = _state(SELECT)
    old_target = state.get("options", {}).get(live) if live else None
    if live and live != state.get("expected") and old_target:
        match = [lab for lab, tgt in mapping.items() if tgt == old_target]
        if match:
            shown = match[0]
        else:
            lab = _label(T.Trip.load(old_target[0]), datetime.strptime(old_target[1], "%Y-%m-%d %H:%M").replace(
                tzinfo=T.Trip.load(old_target[0]).origin_tz), None, None, multi).replace("watching", "picked")
            mapping[lab] = old_target
            shown = lab
    # write what we expect before touching HA, so the 5-minute check doesn't mistake our own change for the user's
    state.update(options=mapping, expected=expected)          # keep the button records
    STATE_FILE.write_text(json.dumps(state, indent=1))
    ha_api("POST", "services/input_select/set_options", {"entity_id": SELECT, "options": list(mapping)})
    ha_api("POST", "services/input_select/select_option", {"entity_id": SELECT, "option": shown})
    return shown, list(mapping)


NEW_TRIP_BUTTON = "input_button.passage_new_trip"
CANCEL_BUTTON = "input_button.passage_cancel_all"
UPDATE_BUTTON = "input_button.passage_update_now"
UPDATE_COOLDOWN_MIN = 30     # an update uses a good share of Open-Meteo's hourly allowance
FROM_TEXT, TO_TEXT, EARLIEST_DATE = "input_text.passage_from", "input_text.passage_to", "input_datetime.passage_earliest"


def _state(entity):
    return (ha_api("GET", f"states/{entity}") or {}).get("state")


def _end_active(log):
    ended = []
    for t in T.Trip.all():
        if t.status in ("planning", "watching"):
            t.status = "done"
            t.save()
            ended.append(t.slug)
            log(f"{t.slug}: ended from Home Assistant")
    return ended


def _pressed(state, key, entity):
    """True when the button's last-press time differs from the one already handled.
    The first time a button is seen its current time is recorded, not acted on."""
    now = _state(entity)
    if key not in state:
        state[key] = now
        return False
    if now in (None, "unknown", "unavailable") or now == state[key]:
        return False
    state[key] = now
    return True


def start_trip_from_ha(log):
    import screen
    origin, dest, earliest = _state(FROM_TEXT), _state(TO_TEXT), _state(EARLIEST_DATE)
    blank = [n for n, v in (("From", origin), ("To", dest), ("Earliest departure", earliest))
             if v in (None, "", "unknown", "unavailable")]
    if blank:
        raise ValueError(f"fill in {', '.join(blank)} first")
    o, onote = T.resolve_place(origin)
    d, dnote = T.resolve_place(dest)
    ended = _end_active(log)
    slug = base = T.slugify(f"{o['name']}-{d['name']}-{earliest}")
    n = 2
    while (T.TRIPS / slug / "trip.json").exists():
        slug, n = f"{base}-{n}", n + 1
    t = T.Trip(slug=slug, origin=o, dest=d, earliest=earliest)
    t.save()
    log(f"{slug}: planned from Home Assistant. From {onote}. To {dnote}")
    _, _, days = screen.screen(t)
    t = T.Trip.load(slug)
    msg = publish(t, days, force_alert=False, alert_prefix=(
        f"New trip. From {o['name']} ({o['lat']}, {o['lon']}), to {d['name']} ({d['lat']}, {d['lon']}). "
        + ("Ended: " + ", ".join(ended) + ". " if ended else "")), always_alert=True)
    return f"new trip {slug}. {msg}"


def update_now(state, log):
    """Fresh forecasts for every active trip, the full brief for a watched one, and an alert either way."""
    import full_report
    import screen
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    last = state.get("last_manual_update")
    if last and now - datetime.fromisoformat(last) < timedelta(minutes=UPDATE_COOLDOWN_MIN):
        ready = datetime.fromisoformat(last) + timedelta(minutes=UPDATE_COOLDOWN_MIN)
        notify("Passage: update skipped", f"The last update was under {UPDATE_COOLDOWN_MIN} minutes ago. "
               f"Press again after {ready.astimezone(timezone(timedelta(hours=12))):%H:%M} Fiji time.",
               f"{HA_URL}/index.html?v={SHELL_VERSION}")
        return "update skipped, cooldown"
    state["last_manual_update"] = now.isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=1))
    active = [t for t in T.Trip.all() if t.status in ("planning", "watching")]
    if not active:
        notify("Passage: nothing to update", "No trip is being screened or watched.", f"{HA_URL}/index.html?v={SHELL_VERSION}")
        return "update: no active trips"
    done = []
    for t in active:
        try:
            _, _, days = screen.screen(t)
            if t.status == "watching":
                full_report.build_for_trip(T.Trip.load(t.slug), fresh=timedelta(hours=3))
            publish(T.Trip.load(t.slug), days, alert_prefix="Updated on request. ", always_alert=True)
            done.append(t.slug)
            log(f"{t.slug}: updated on request")
        except Exception as e:
            log(f"{t.slug}: update on request FAILED {type(e).__name__}: {e}")
            notify("Passage: update failed", f"{t.title()}: {e}", f"{HA_URL}/index.html?v={SHELL_VERSION}")
    return "updated on request: " + ", ".join(done)


def sync_select(log=print):
    """Act on the Home Assistant form, buttons and dropdown. Returns a short description of what happened."""
    with Lock(blocking=False) as got:
        if got is None:
            return "busy, another update is running"
        state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {"options": {NOT_WATCHING: None},
                                                                                "expected": NOT_WATCHING}
        cancel = _pressed(state, "cancel_seen", CANCEL_BUTTON)
        new = _pressed(state, "new_seen", NEW_TRIP_BUTTON)
        update = _pressed(state, "update_seen", UPDATE_BUTTON)
        STATE_FILE.write_text(json.dumps(state, indent=1))
        if cancel:
            ended = _end_active(log)
            update_select()
            write_index()
            notify("Passage: watches cancelled", f"Ended {len(ended)} trip(s)." if ended else "No trips were active.",
                   f"{HA_URL}/index.html?v={SHELL_VERSION}")
            return "cancelled all watches" + (f": {', '.join(ended)}" if ended else "")
        if update and not (cancel or new):
            return update_now(state, log)
        if new:
            try:
                return start_trip_from_ha(log)
            except Exception as e:
                notify("Passage: new trip not started", f"{e}. Nothing was changed.", f"{HA_URL}/index.html?v={SHELL_VERSION}")
                return f"new trip not started: {e}"
        current = (ha_api("GET", f"states/{SELECT}") or {}).get("state")
        if current is None or current == state["expected"]:
            return "no change"
        if current not in state["options"]:
            return f"unknown option '{current}', ignored"
        target = state["options"][current]
        state["expected"] = current
        STATE_FILE.write_text(json.dumps(state, indent=1))
        if target is None:
            for t in T.Trip.all():
                if t.status == "watching":
                    t.status, t.watch_depart = "planning", None
                    t.save()
                    log(f"{t.slug}: dropdown set to not watching")
            write_index()
            return "stopped watching"
        slug, depart = target
        for t in T.Trip.all():               # one watched departure at a time
            if t.status == "watching" and t.slug != slug:
                t.status, t.watch_depart = "planning", None
                t.save()
        t = T.Trip.load(slug)
        t.watch_depart, t.status = depart, "watching"
        t.save()
        log(f"{slug}: dropdown picked {depart}, building the full brief")
        import full_report
        import screen
        full_report.build_for_trip(t)
        run = sorted((t.dir / "runs").iterdir())[-1]
        _, _, days = screen.screen(T.Trip.load(slug), run_dir=run)
        msg = publish(T.Trip.load(slug), days, force_alert=True)
        return f"watching {slug} {depart}. {msg}"
