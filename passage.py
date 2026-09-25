"""Plan a passage, screen departure options, watch a day, get the full brief.

  python3 passage.py plan "Nadi" "Port Resolution" --earliest 2026-09-22 [--days 4] [--exit-hours 1]
        [--from-country FJ] [--to-country VU] [--from-tz ...] [--to-tz ...]
        Places can be names or "lat,lon" (then give the time zone).
  python3 passage.py list
  python3 passage.py screen <trip>            departure options for each day in the window
  python3 passage.py watch <trip> 2026-09-23    pick a day (noon), or "2026-09-23 09:00" for another time
  python3 passage.py report <trip>            full brief for the watched departure
  python3 passage.py publish <trip> [--alert]  copy pages to Home Assistant, alert if changed
  python3 passage.py done <trip>              stop updating a trip
  python3 passage.py auto                     what the timer runs: screen every active trip,
                                              full brief for watched ones
  python3 passage.py place add "Name" "lat,lon" Pacific/Efate   save a position the geocoder gets wrong
  python3 passage.py sync-ha                  act on the Home Assistant "day to watch" dropdown
  python3 passage.py install-timer            systemd user timers: updates at 08:30 and 20:30 UTC,
                                              dropdown check every 5 minutes

One departure per day is screened, at local noon. Estimated guide only. The captain develops the detailed route for every passage.
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import trip as T

HERE = Path(__file__).parent


def cmd_plan(a):
    o, onote = T.resolve_place(a.origin, a.from_country, a.from_tz)
    d, dnote = T.resolve_place(a.dest, a.to_country, a.to_tz)
    if a.from_name:
        o["name"] = a.from_name
    if a.to_name:
        d["name"] = a.to_name
    slug = a.name or T.slugify(f"{o['name']}-{d['name']}-{a.earliest}")
    t = T.Trip(slug=slug, origin=o, dest=d, earliest=a.earliest, days=a.days, exit_hours=a.exit_hours)
    t.save()
    r = t.use_route()
    print(f"trip {slug}")
    print(f"  from {o['name']} {o['lat']}, {o['lon']} ({o['tz']})  <- {onote}")
    print(f"  to   {d['name']} {d['lat']}, {d['lon']} ({d['tz']})  <- {dnote}")
    print(f"  {r.total_nm():.0f} nm great circle, leaving any day from {a.earliest} for {a.days} days")
    print(f"next: python3 passage.py screen {slug}")


def cmd_list(a):
    for t in T.Trip.all():
        w = f", watching {t.watch_depart}" if t.watch_depart else ""
        print(f"{t.slug:40s} {t.status:9s} {t.title()} from {t.earliest} for {t.days} days{w}"
              + (f"  {t.page_url}" if t.page_url else ""))


def cmd_screen(a):
    import screen
    t = T.Trip.load(a.trip)
    out, pdf, days = screen.screen(t)
    print(screen.recommendation(t, days))
    for d in days:
        b = d.get("best")
        print(f"  {d['date']:%a %d %b}: " + (f"{b['verdict']:8s} leave {b['depart'].astimezone(t.origin_tz):%H:%M}, "
                                            f"wind risk {b['risk']:.0%}, {b['hours50']:.0f} h, comfort {d['comfort_main']}/{d['comfort_worst']}"
                                            if b else "beyond the forecast"))
    print(out)
    if pdf:
        print(pdf)


def cmd_watch(a):
    t = T.Trip.load(a.trip)
    if len(a.depart.strip()) == 10:          # a date alone means noon
        a.depart = f"{a.depart.strip()} {T.Trip.DEPART_HOUR:02d}:00"
    datetime.strptime(a.depart, "%Y-%m-%d %H:%M")
    t.watch_depart, t.status = a.depart, "watching"
    t.save()
    print(f"{t.slug}: watching a departure at {a.depart} {t.origin['tz']}. Next: python3 passage.py report {t.slug}")


def cmd_report(a):
    import full_report
    t = T.Trip.load(a.trip)
    if not t.watch_depart:
        sys.exit(f"{t.slug} has no watched departure. Pick one: python3 passage.py watch {t.slug} 'YYYY-MM-DD HH:MM'")
    out, pdf = full_report.build_for_trip(t)
    print(out)
    if pdf:
        print(pdf)


def cmd_publish(a):
    """Rescreen from the latest fetch, copy pages to Home Assistant, alert if changed (or --alert)."""
    import ha_publish
    import screen
    t = T.Trip.load(a.trip)
    run = sorted((t.dir / "runs").iterdir())[-1]
    _, _, days = screen.screen(t, run_dir=run)
    print(ha_publish.publish(T.Trip.load(t.slug), days, force_alert=a.alert))
    print(f"{ha_publish.HA_URL}/index.html")


def cmd_done(a):
    t = T.Trip.load(a.trip)
    t.status = "done"
    t.save()
    print(f"{t.slug}: done, no more updates")


def cmd_place(a):
    lat, lon = (float(x) for x in a.position.split(","))
    ZoneInfo(a.tz)
    T.save_place(a.name.strip().lower(), {"name": a.name, "lat": lat, "lon": lon, "tz": a.tz, "source": "added by hand"})
    print(f"saved {a.name} {lat}, {lon} {a.tz}")


def cmd_auto(a):
    """Screen active trips; full brief for watched ones. Retire trips once the departure has passed."""
    import ha_publish
    with ha_publish.Lock():
        _auto()


def cmd_sync_ha(a):
    """What the 5-minute timer runs: act on a change to the HA day-to-watch dropdown."""
    import ha_publish
    log = HERE / "trips" / "auto.log"
    result = ha_publish.sync_select(log=lambda m: _log(log, m))
    if result not in ("no change", "busy, another update is running"):
        _log(log, f"dropdown: {result}")
    else:
        print(result)


def _auto():
    import screen
    log = HERE / "trips" / "auto.log"
    log.parent.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    for t in T.Trip.all():
        if t.status == "done":
            continue
        first, last = t.window()
        end = t.watch_time() or last
        if now > end.astimezone(timezone.utc) + timedelta(hours=12):
            t.status = "done"
            t.save()
            _log(log, f"{t.slug}: departure window has passed, marked done")
            continue
        try:
            out, pdf, days = screen.screen(t)
            _log(log, f"{t.slug}: screened. {screen.recommendation(t, days)}")
            if t.status == "watching":
                import full_report
                time.sleep(65)   # let the Open-Meteo minute allowance recover before the grid pull
                rout, rpdf = full_report.build_for_trip(t)
                _log(log, f"{t.slug}: full brief {rout}")
        except Exception as e:
            _log(log, f"{t.slug}: FAILED {type(e).__name__}: {e}")
            continue
        import settings
        if settings.get()["home_assistant"]["enabled"]:
            try:
                import ha_publish
                _log(log, ha_publish.publish(T.Trip.load(t.slug), days))
            except Exception as e:
                _log(log, f"{t.slug}: Home Assistant step FAILED {type(e).__name__}: {e}")


def _log(path, msg):
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%MZ} {msg}"
    print(line)
    with open(path, "a") as f:
        f.write(line + "\n")


UNIT = """[Unit]
Description=Passage weather: screen trips and refresh briefs

[Service]
Type=oneshot
WorkingDirectory={here}
ExecStart={python} {here}/passage.py auto
"""
TIMER = """[Unit]
Description=Passage weather updates after each ECMWF ensemble run

[Timer]
# 08:30 and 20:30 UTC: after the 00Z and 12Z ensemble runs are published.
OnCalendar=*-*-* 08:30:00 UTC
OnCalendar=*-*-* 20:30:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
"""


SYNC_UNIT = """[Unit]
Description=Passage weather: act on the Home Assistant day-to-watch dropdown

[Service]
Type=oneshot
WorkingDirectory={here}
ExecStart={python} {here}/passage.py sync-ha
"""
SYNC_TIMER = """[Unit]
Description=Check the Home Assistant day-to-watch dropdown every 5 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
"""


def cmd_install_timer(a):
    d = Path.home() / ".config" / "systemd" / "user"
    d.mkdir(parents=True, exist_ok=True)
    (d / "passage-weather.service").write_text(UNIT.format(here=HERE, python=sys.executable))
    (d / "passage-weather.timer").write_text(TIMER)
    (d / "passage-weather-sync.service").write_text(SYNC_UNIT.format(here=HERE, python=sys.executable))
    (d / "passage-weather-sync.timer").write_text(SYNC_TIMER)
    for args in (["daemon-reload"], ["enable", "--now", "passage-weather.timer", "passage-weather-sync.timer"]):
        subprocess.run(["systemctl", "--user", *args], check=True)
    subprocess.run(["systemctl", "--user", "list-timers", "passage-weather*"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan")
    s.add_argument("origin"), s.add_argument("dest")
    s.add_argument("--earliest", required=True, help="YYYY-MM-DD, local date at the origin")
    s.add_argument("--days", type=int, default=4)
    s.add_argument("--exit-hours", type=float, default=1.0)
    s.add_argument("--from-country"), s.add_argument("--to-country"), s.add_argument("--from-tz"), s.add_argument("--to-tz")
    s.add_argument("--name"), s.add_argument("--from-name"), s.add_argument("--to-name")
    s.set_defaults(func=cmd_plan)
    sub.add_parser("list").set_defaults(func=cmd_list)
    for name, fn in (("screen", cmd_screen), ("report", cmd_report), ("done", cmd_done)):
        s = sub.add_parser(name)
        s.add_argument("trip")
        s.set_defaults(func=fn)
    s = sub.add_parser("publish")
    s.add_argument("trip"), s.add_argument("--alert", action="store_true", help="send the alert even if nothing changed")
    s.set_defaults(func=cmd_publish)
    s = sub.add_parser("watch")
    s.add_argument("trip"), s.add_argument("depart")
    s.set_defaults(func=cmd_watch)
    sub.add_parser("auto").set_defaults(func=cmd_auto)
    s = sub.add_parser("place")
    s.add_argument("action", choices=["add"]), s.add_argument("name"), s.add_argument("position"), s.add_argument("tz")
    s.set_defaults(func=cmd_place)
    sub.add_parser("sync-ha").set_defaults(func=cmd_sync_ha)
    sub.add_parser("install-timer").set_defaults(func=cmd_install_timer)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
