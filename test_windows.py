"""Checks against synthetic forecasts with known answers. Run: python3 test_windows.py"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
from pathlib import Path
import requests

import polar
import route
import windows

VMC = polar.vmc_table(polar.conservative_polar())
RT = windows.route_table()
T0 = datetime(2026, 9, 20, tzinfo=timezone.utc)
N_T, N_P = 24 * 10, len(route.sample_points())
COURSE = route.gc_bearing(*route.WAYPOINTS[0][1:], *route.WAYPOINTS[1][1:])


def synthetic(tws, twd_from, gust=None, hs=1.5, period=9.0, wave_from=None, current_kt=0.0, current_towards=0.0):
    times = [T0 + timedelta(hours=i) for i in range(N_T)]
    full = lambda v: np.full((N_T, N_P), float(v))
    wind = dict(times=times, spd=full(tws)[None], dir=full(twd_from)[None], gust=full(gust or tws * 1.3)[None])
    wave_from = twd_from if wave_from is None else wave_from
    waves = {wm: dict(times=times, hs=full(hs), period=full(period), dir=full(wave_from)) for wm in windows.WAVE_MODELS}
    c = math.radians(current_towards)
    waves["_currents"] = dict(times=times, east=full(current_kt * math.sin(c)), north=full(current_kt * math.cos(c)))
    return wind, waves


def run(tws, twd_from, **kw):
    wind, waves = synthetic(tws, twd_from, **kw)
    return windows.sail(0, T0, wind, waves, VMC, RT)


def test_steady_trade_passage_time():
    # 16 kt from 140 deg off the course: expected time = distance / VMC + exit hours
    twd = (COURSE + 140) % 360
    r = run(16, twd)
    course_twas = [abs(((twd - c) + 180) % 360 - 180) for c in RT[1]]
    mean_v = np.mean([VMC[16, int(round(a))] for a in course_twas])
    expect = route.total_nm() / mean_v + route.EXIT_HOURS
    assert abs(r["hours"] - expect) < 1.0, (r["hours"], expect)
    assert r["beat_h"] == 0 and r["motor_h"] == 0 and not any(r["breach"].values()), r


def test_headwind_counts_as_beating():
    r = run(16, COURSE)  # wind from dead ahead on the first leg
    assert r["up_share"] > 0.9, r
    # sails make VMC[16, 0] ~2.9 kt dead upwind, below the 4 kt motoring speed into 16 kt, so it motors
    assert VMC[16, 0] < windows.motor_speed(16, 0)
    expect = route.total_nm() / windows.motor_speed(16, 0) + route.EXIT_HOURS
    assert abs(r["hours"] - expect) < 6, (r["hours"], expect)   # last leg is 4 deg off, so sails a little
    assert r["motor_h"] > 0.9 * (r["hours"] - route.EXIT_HOURS), r


def test_tailwind_is_not_beating():
    r = run(16, (COURSE + 180) % 360)
    assert r["beat_h"] == 0, r


def test_strong_wind_breaches():
    # gusts are 1.23 x sustained (WMO at-sea), whatever the model's gust field says
    warn = run(29, (COURSE + 140) % 360, gust=10)            # gust 35.7
    assert warn["breach"]["tws_warn"] and not warn["breach"]["tws_no"], warn["breach"]
    assert warn["breach"]["gust_warn"] and not warn["breach"]["gust_no"], warn["breach"]
    no = run(33, (COURSE + 140) % 360, gust=10)              # gust 40.6
    assert no["breach"]["tws_no"] and no["breach"]["gust_no"], no["breach"]
    calm = run(22, (COURSE + 140) % 360, gust=50)            # a 50 kt model gust is ignored: 27 kt estimated
    assert not any(calm["breach"].values()), calm["breach"]
    assert abs(calm["max_gust"] - 22 * 1.23) < 1e-9, calm["max_gust"]


def test_calm_motors():
    r = run(3, (COURSE + 90) % 360)
    assert r["motor_h"] > 0.9 * (r["hours"] - route.EXIT_HOURS), r
    expect = route.total_nm() / 5.5 + route.EXIT_HOURS    # the default 5.5 kt under power
    assert abs(r["hours"] - expect) < 1.0, (r["hours"], expect)


def test_waves_recorded_per_model():
    r = run(16, (COURSE + 140) % 360, hs=3.4)
    assert all(abs(r[f"hs_{wm}"] - 3.4) < 1e-9 for wm in windows.WAVE_MODELS), r


def test_wave_rules_use_period_and_direction():
    """Height alone isn't a no-go: it depends on the period and where the seas hit."""
    trade = (COURSE + 140) % 360
    astern, ahead = (COURSE + 180) % 360, COURSE
    big_following = run(16, trade, hs=3.2, period=12.0, wave_from=astern)
    big_ahead = run(16, trade, hs=3.2, period=12.0, wave_from=ahead)
    steep = run(16, trade, hs=2.5, period=7.0, wave_from=astern)          # 7 s < 2.5 m in feet (8.2 s)
    huge = run(16, trade, hs=4.5, period=14.0, wave_from=astern)
    calm = run(16, trade, hs=2.0, period=10.0, wave_from=astern)
    wm = "ecmwf_wam025"
    assert not big_following[f"wave_warn_{wm}"] and not big_following[f"wave_no_{wm}"], big_following[f"wave_why_{wm}"]
    assert big_ahead[f"wave_warn_{wm}"] and not big_ahead[f"wave_no_{wm}"], big_ahead[f"wave_why_{wm}"]
    assert "over 3 m forward of the beam" in big_ahead[f"wave_why_{wm}"]
    assert steep[f"wave_warn_{wm}"] and "steeper than the feet rule" in steep[f"wave_why_{wm}"]
    assert huge[f"wave_no_{wm}"] and "over 4 m" in huge[f"wave_why_{wm}"]
    assert not calm[f"wave_warn_{wm}"] and not calm[f"wave_no_{wm}"], calm[f"wave_why_{wm}"]
    # a steep sea on the bow throws the boat around enough to be a no-go on motion alone
    rough = run(16, trade, hs=3.0, period=6.0, wave_from=ahead)
    assert rough[f"wave_no_{wm}"] and "motion at the Rough band" in rough[f"wave_why_{wm}"], rough[f"wave_why_{wm}"]


def test_pw_polar_and_night_factor():
    raw = polar.pw_raw_grid()
    assert abs(raw[polar.TWA.index(120), polar.TWS.index(20)] - 9.86) < 1e-9     # file value, TWS 20 TWA 120
    assert abs(raw[polar.TWA.index(180), polar.TWS.index(20)] - 7.99) < 1e-9     # 180 takes 170
    assert abs(raw[polar.TWA.index(90), polar.TWS.index(22)] - (8.76 + (11.16 - 8.76) * 2 / 5)) < 1e-9  # between 20 and 25
    day, night = polar.pw_polars()
    assert abs(day[polar.TWA.index(60), polar.TWS.index(20)] - 7.34 * 0.90) < 1e-9
    assert abs(day[polar.TWA.index(120), polar.TWS.index(20)] - 9.86 * 0.85) < 1e-9
    vd, vn = polar.vmc_table(day), polar.vmc_table(night)
    w, wv = synthetic(16, (COURSE + 140) % 360)
    t_day = windows.sail(0, T0, w, wv, vd, RT)["hours"]
    t_mix = windows.sail(0, T0, w, wv, vd, RT, vmc_night=vn)["hours"]
    assert t_mix > t_day + 3, (t_day, t_mix)      # about half the passage is at night at 75%


def test_night_detection():
    lat, lon = route.WAYPOINTS[0][1:]
    assert windows.is_night(datetime(2026, 9, 15, 2, tzinfo=windows.FJT).astimezone(timezone.utc), lat, lon)
    assert not windows.is_night(datetime(2026, 9, 15, 12, tzinfo=windows.FJT).astimezone(timezone.utc), lat, lon)
    assert windows.is_night(datetime(2026, 9, 15, 21, tzinfo=windows.FJT).astimezone(timezone.utc), lat, lon)


def test_trip_window_is_origin_local():
    import trip as T
    t = T.Trip(slug="x", origin={"name": "A", "lat": -17.8, "lon": 177.4, "tz": "Pacific/Fiji"},
               dest={"name": "B", "lat": -17.7, "lon": 168.3, "tz": "Pacific/Efate"}, earliest="2026-09-22", days=4)
    first, last = t.window()
    assert first.isoformat() == "2026-09-22T12:00:00+12:00", first     # local noon
    assert last.isoformat() == "2026-09-25T12:00:00+12:00", last
    assert T.tz_abbrev(first, t.dest_tz) == "UTC+11"


def test_synoptic_domain_reaches_poleward():
    import synoptic as S
    la, lo, step = S.domain_for([("A", -17.8, 177.4), ("B", -17.7, 168.3)])
    assert la[0] >= -17.7 + 12 - step and la[-1] <= -17.8 - 20 + step, (la[0], la[-1])
    assert lo[0] <= 168.3 - 15 + step and lo[-1] >= 177.4 + 15 - step, (lo[0], lo[-1])
    assert la.size * lo.size <= 256
    la, lo, step = S.domain_for([("C", 20.0, -157.0), ("D", 33.0, -118.0)])   # northern hemisphere
    assert la[0] >= 33 + 20 - step and la[-1] <= 20 - 12 + step, (la[0], la[-1])


def test_best_slot_prefers_verdict_then_risk():
    import screen
    base = dict(p_day=1.0, hours50=70)
    go = dict(base, verdict="GO", risk=0.09)
    marginal = dict(base, verdict="MARGINAL", risk=0.02)
    no_low = dict(base, verdict="NO", risk=0.31)
    assert min([marginal, no_low, go], key=screen.best_key) is go
    dark = dict(base, verdict="GO", risk=0.09, p_day=0.2)
    assert min([dark, go], key=screen.best_key) is go


def test_ha_alert_only_on_change(tmp_path=None):
    import tempfile
    import trip as T
    import ha_publish
    from datetime import datetime
    sent, old_trips, old_state = [], T.TRIPS, ha_publish.STATE_FILE
    T.TRIPS = Path(tempfile.mkdtemp())
    ha_publish.STATE_FILE = T.TRIPS / "ha_watch.json"
    orig = (ha_publish.copy_pages, ha_publish.write_index, ha_publish.notify, ha_publish.update_select)
    ha_publish.copy_pages = lambda t: "url"
    ha_publish.write_index = lambda: "index"
    ha_publish.update_select = lambda: None
    ha_publish.notify = lambda title, msg, url: sent.append(msg) or "200"
    try:
        t = T.Trip(slug="alert-test", origin={"name": "A", "lat": -17.8, "lon": 177.4, "tz": "Pacific/Fiji"},
                   dest={"name": "B", "lat": -17.7, "lon": 168.3, "tz": "Pacific/Efate"}, earliest="2026-09-22", days=2)
        t.save()
        d0 = datetime(2026, 9, 22, 9, tzinfo=t.origin_tz)
        slot = lambda v, r: dict(depart=d0, verdict=v, risk=r, p_day=1.0, hours50=80)
        days = lambda v, r: [dict(date=d0, slots=[slot(v, r)], best=slot(v, r)), dict(date=d0, slots=[])]
        ha_publish.publish(t, days("MARGINAL", 0.2))                    # first look: alert
        ha_publish.publish(T.Trip.load("alert-test"), days("MARGINAL", 0.25))   # same day and verdict: quiet
        assert len(sent) == 1, sent
        ha_publish.publish(T.Trip.load("alert-test"), days("GO", 0.05))  # verdict changed: alert
        assert len(sent) == 2, sent
        t = T.Trip.load("alert-test"); t.watch_depart, t.status = "2026-09-22 09:00", "watching"; t.save()
        ha_publish.publish(T.Trip.load("alert-test"), days("GO", 0.05))  # now watching: new signature, alert
        assert len(sent) == 3 and "Watched" in sent[-1], sent
        ha_publish.publish(T.Trip.load("alert-test"), days("GO", 0.06))  # nothing changed: quiet
        assert len(sent) == 3, sent
    finally:
        ha_publish.copy_pages, ha_publish.write_index, ha_publish.notify, ha_publish.update_select = orig
        T.TRIPS, ha_publish.STATE_FILE = old_trips, old_state


def test_dropdown_pick_sets_watch():
    import json, sys, tempfile, types
    import trip as T
    import ha_publish as H
    tmp = Path(tempfile.mkdtemp())
    saved = (T.TRIPS, H.STATE_FILE, H.LOCK_FILE, H.ha_api, H.publish, H.write_index)
    T.TRIPS, H.STATE_FILE, H.LOCK_FILE = tmp, tmp / "ha_watch.json", tmp / ".lock"
    ha = {"state": "Not watching", "options": []}
    def fake_api(method, path, payload=None):
        if path.endswith("set_options"):
            ha["options"], ha["state"] = payload["options"], payload["options"][0]
        elif path.endswith("select_option"):
            ha["state"] = payload["option"]
        elif path.startswith("states/"):
            return {"state": ha["state"] if path.endswith("passage_watch") else "unknown"}
    built = []
    H.ha_api, H.write_index = fake_api, lambda: None
    H.publish = lambda t, days, force_alert=False: "published"
    sys.modules["full_report"] = types.SimpleNamespace(build_for_trip=lambda t: built.append(t.watch_depart))
    real_screen = sys.modules.get("screen")
    sys.modules["screen"] = types.SimpleNamespace(screen=lambda t, run_dir=None: (None, None, []))
    try:
        t = T.Trip(slug="dd", origin={"name": "A", "lat": -17.8, "lon": 177.4, "tz": "Pacific/Fiji"},
                   dest={"name": "B", "lat": -19.5, "lon": 169.5, "tz": "Pacific/Efate"}, earliest="2026-09-22", days=2)
        t.save()
        (t.dir / "runs" / "20260915T0000Z").mkdir(parents=True)
        (t.dir / "history.json").write_text(json.dumps([{"days": {
            "2026-09-22": {"verdict": "NO", "risk": 0.67, "best": "09:00"},
            "2026-09-23": {"verdict": "MARGINAL", "risk": 0.2, "best": "06:00"}}}]))
        expected, options = H.update_select()
        assert expected == "Not watching" and len(options) == 3, options
        assert H.sync_select(log=lambda m: None) == "no change"          # our own refill isn't the user's pick
        ha["state"] = "Wed 23 Sep 06:00 · MARGINAL 20%"                   # the user picks Wednesday
        result = H.sync_select(log=lambda m: None)
        t = T.Trip.load("dd")
        assert t.status == "watching" and t.watch_depart == "2026-09-23 06:00", (result, t)
        assert built == ["2026-09-23 06:00"], built
        H.update_select()                                                  # next screening keeps the pick
        assert ha["state"] == "Wed 23 Sep 06:00 · MARGINAL 20%", ha
        # a refill between the user's pick and the 5-minute check must not wipe the pick, even if the label changes
        ha["state"] = "Not watching"
        H.sync_select(log=lambda m: None)                                  # back to not watching
        (t.dir / "history.json").write_text(json.dumps([{"days": {
            "2026-09-22": {"verdict": "NO", "risk": 0.67, "best": "09:00"},
            "2026-09-23": {"verdict": "MARGINAL", "risk": 0.2, "best": "06:00"}}}]))
        H.update_select()
        ha["state"] = "Tue 22 Sep 09:00 · NO 67%"                          # the user picks Tuesday
        (t.dir / "history.json").write_text(json.dumps([{"days": {
            "2026-09-22": {"verdict": "WARNING", "risk": 0.36, "best": "09:00"},
            "2026-09-23": {"verdict": "NO", "risk": 0.39, "best": "06:00"}}}]))
        H.update_select()                                                  # a rebuild lands before the check
        assert ha["state"] == "Tue 22 Sep 09:00 · WARNING 36%", ha           # same departure, new label, still picked
        built.clear()
        H.sync_select(log=lambda m: None)
        assert T.Trip.load("dd").watch_depart == "2026-09-22 09:00" and built == ["2026-09-22 09:00"], built
        assert H.sync_select(log=lambda m: None) == "no change"
        ha["state"] = "Not watching"
        H.sync_select(log=lambda m: None)
        assert T.Trip.load("dd").status == "planning"
    finally:
        T.TRIPS, H.STATE_FILE, H.LOCK_FILE, H.ha_api, H.publish, H.write_index = saved
        sys.modules.pop("full_report", None)
        if real_screen:
            sys.modules["screen"] = real_screen
        else:
            sys.modules.pop("screen", None)


def test_ha_buttons_start_and_cancel():
    import json, sys, tempfile, types
    import trip as T
    import ha_publish as H
    tmp = Path(tempfile.mkdtemp())
    saved = (T.TRIPS, H.STATE_FILE, H.LOCK_FILE, H.ha_api, H.publish, H.write_index, H.notify, T.resolve_place)
    T.TRIPS, H.STATE_FILE, H.LOCK_FILE = tmp, tmp / "ha_watch.json", tmp / ".lock"
    ha = {"input_select.passage_watch": "Not watching", H.NEW_TRIP_BUTTON: "2026-09-10T00:00:00+00:00",
          H.CANCEL_BUTTON: "unknown", H.UPDATE_BUTTON: "unknown", H.FROM_TEXT: "Savusavu", H.TO_TEXT: "Neiafu, TO", H.EARLIEST_DATE: "2026-10-02"}
    def fake_api(method, path, payload=None):
        if path.endswith("set_options"):
            ha["input_select.passage_watch"] = payload["options"][0]
        elif path.endswith("select_option"):
            ha["input_select.passage_watch"] = payload["option"]
        elif path.startswith("states/"):
            return {"state": ha[path[len("states/"):]]}
    alerts, published = [], []
    H.ha_api, H.write_index = fake_api, lambda: None
    H.notify = lambda title, msg, url: alerts.append((title, msg)) or "200"
    H.publish = lambda t, days, **kw: published.append(t.slug) or "published"
    T.resolve_place = lambda text, *a, **k: ({"name": text.split(",")[0], "lat": -17.0, "lon": 179.0, "tz": "Pacific/Fiji"}, "stub")
    real_screen = sys.modules.get("screen")
    sys.modules["screen"] = types.SimpleNamespace(screen=lambda t, run_dir=None: (None, None, []))
    try:
        old = T.Trip(slug="old", origin={"name": "A", "lat": -17.8, "lon": 177.4, "tz": "Pacific/Fiji"},
                     dest={"name": "B", "lat": -19.5, "lon": 169.5, "tz": "Pacific/Efate"}, earliest="2026-09-22", status="watching")
        old.save()
        assert H.sync_select(log=lambda m: None) == "no change"            # first sight of the buttons: record, don't act
        assert T.Trip.load("old").status == "watching" and not published
        ha[H.NEW_TRIP_BUTTON] = "2026-09-15T01:00:00+00:00"                  # the user presses Start new trip
        r = H.sync_select(log=lambda m: None)
        assert T.Trip.load("old").status == "done", r
        new = [t for t in T.Trip.all() if t.status == "planning"]
        assert len(new) == 1 and new[0].earliest == "2026-10-02" and new[0].dest["name"] == "Neiafu", (r, new)
        assert published == [new[0].slug], published
        assert H.sync_select(log=lambda m: None) == "no change"            # same press isn't handled twice
        H.update_select()                                                  # a refill keeps the button records
        assert H.sync_select(log=lambda m: None) == "no change"
        ha[H.CANCEL_BUTTON] = "2026-09-15T02:00:00+00:00"                    # first sight recorded above, now pressed
        r = H.sync_select(log=lambda m: None)
        assert r.startswith("cancelled") and not [t for t in T.Trip.all() if t.status != "done"], r
        ha[H.FROM_TEXT] = ""                                                # a blank field: alert, change nothing
        ha[H.NEW_TRIP_BUTTON] = "2026-09-15T03:00:00+00:00"
        r = H.sync_select(log=lambda m: None)
        assert "not started" in r and "not started" in alerts[-1][0], (r, alerts)
    finally:
        T.TRIPS, H.STATE_FILE, H.LOCK_FILE, H.ha_api, H.publish, H.write_index, H.notify, T.resolve_place = saved
        if real_screen:
            sys.modules["screen"] = real_screen
        else:
            sys.modules.pop("screen", None)


def test_wave_warning_says_why():
    import screen
    def model(share, hs, why, no=0.0):
        return {"p_wave_ecmwf_wam025": share, "hs_ecmwf_wam025_p90": hs, "wave_why_ecmwf_wam025": why,
                "p_wave_no_ecmwf_wam025": no, "ratio_ecmwf_wam025_p10": 3.0,
                "p_wave_ncep_gfswave025": 0.0, "hs_ncep_gfswave025_p90": 2.0, "wave_why_ncep_gfswave025": [],
                "p_wave_no_ncep_gfswave025": 0.0, "ratio_ncep_gfswave025_p10": 5.0}
    row = {m: model(0.5, 3.4, ["over 3 m forward of the beam", "steeper than the feet rule"]) for m in windows.WIND_MODELS}
    assert screen.wave_warnings(row) == [
        "ECMWF wave model: over 3 m forward of the beam, steeper than the feet rule (to 3.4 m)"], screen.wave_warnings(row)
    row = {m: model(0.5, 4.3, ["over 4 m"], no=0.5) for m in windows.WIND_MODELS}
    assert "a no-go" in screen.wave_warnings(row)[0], screen.wave_warnings(row)
    row = {m: model(0.05, 3.4, ["over 3 m forward of the beam"]) for m in windows.WIND_MODELS}   # under the 10% share
    assert screen.wave_warnings(row) == []


def test_point_of_sail_shares():
    import comfort
    head = run(16, COURSE)                      # dead ahead on the long leg
    beam = run(16, (COURSE + 90) % 360)
    broad = run(16, (COURSE + 130) % 360)
    tail = run(16, (COURSE + 175) % 360)
    assert head["up_share"] > 0.9, head
    assert beam["beam_share"] > 0.9, beam
    assert broad["broad_share"] > 0.9, broad
    assert tail["run_share"] > 0.9, tail
    for r in (head, beam, broad, tail):
        assert abs(sum(r[f"{k}_share"] for k, _ in windows.POINTS_OF_SAIL) - 1) < 1e-9
    # sea sectors by where the seas come from relative to the heading
    assert [comfort.sea_side(a) for a in (10, 55, 100, 170, 180)] == ["bow", "fwd_quarter", "aft_quarter", "astern", "astern"]
    swell = run(16, (COURSE + 130) % 360, wave_from=(COURSE + 55) % 360)   # a swell 55 deg off the bow
    assert swell["sea_fwd_quarter_share"] > 0.9, swell


def test_verdict_tiers():
    def row(p_warn, p_no, wave=0.0, wave_no=0.0):
        m = {"p_any": p_warn, "p_no": p_no, "p_wave_ecmwf_wam025": wave, "p_wave_ncep_gfswave025": 0.0,
             "p_wave_no_ecmwf_wam025": wave_no, "p_wave_no_ncep_gfswave025": 0.0}
        return {k: m for k in windows.WIND_MODELS}
    assert windows.verdict(row(0.05, 0.0)) == "GO"
    assert windows.verdict(row(0.20, 0.05)) == "WARNING"
    assert windows.verdict(row(0.60, 0.12)) == "NO"
    assert windows.verdict(row(0.05, 0.0, wave=0.5)) == "WARNING"      # one wave model warns
    assert windows.verdict(row(0.05, 0.0, wave=0.5, wave_no=0.5)) == "NO"   # a wave no-go, wind fine


def test_comfort_depends_on_where_the_seas_come_from():
    import comfort
    astern, beam, bow = (comfort.vertical_accel(3, 8, 7, rel) for rel in (180, 90, 0))
    assert astern < beam < bow, (astern, beam, bow)
    assert comfort.encounter_period(8, 7, 180) > 8 > comfort.encounter_period(8, 7, 0)
    assert [comfort.LEVELS[comfort.level(15, a)] for a in (astern, beam, bow)] == ["Easy", "Coffee", "Rough"]
    assert comfort.LEVELS[comfort.level(26, 0.1)] == "Rough"             # wind over the deck alone
    # whole passages: same 3 m sea, running before it versus punching into it
    trade = (COURSE + 150) % 360
    following = run(16, trade, hs=3.0, period=8.0, wave_from=trade)
    head = run(16, (COURSE + 20) % 360, hs=3.0, period=8.0, wave_from=COURSE)
    worst = lambda r: min(i for i, h in enumerate(r["comfort_h"]) if h > 0)
    assert worst(following) > worst(head), (following["comfort_h"], head["comfort_h"])


def test_current_along_and_across_the_line():
    trade = (COURSE + 140) % 360
    still = run(16, trade)
    fair = run(16, trade, current_kt=1.0, current_towards=COURSE)            # flowing the way we're going
    foul = run(16, trade, current_kt=1.0, current_towards=(COURSE + 180) % 360)
    cross = run(16, trade, current_kt=1.0, current_towards=(COURSE + 90) % 360)
    assert fair["hours"] < still["hours"] < foul["hours"], (fair["hours"], still["hours"], foul["hours"])
    # about 1 kt on roughly 6.6 kt: the fair current passage is about 13% shorter
    assert 0.80 < fair["hours"] / still["hours"] < 0.92, fair["hours"] / still["hours"]
    assert abs(fair["current_along_kt"] - 1.0) < 0.1 and abs(foul["current_along_kt"] + 1.0) < 0.1
    # crabbing across a 1 kt set costs a little: sqrt(6.6^2 - 1) is about 1% slower
    assert still["hours"] < cross["hours"] < still["hours"] * 1.03, (still["hours"], cross["hours"])
    # Open-Meteo gives the direction a current flows towards; a fair current toward the course must speed us up
    e, n = 1.0 * math.sin(math.radians(COURSE)), 1.0 * math.cos(math.radians(COURSE))
    assert windows.current_effect(6.0, COURSE, e, n)[1] > 0.99


def test_currents_file_units_and_direction():
    import json, tempfile
    d = Path(tempfile.mkdtemp())
    (d / "currents.json").write_text(json.dumps([{"hourly": {
        "time": ["2026-09-22T00:00", "2026-09-22T01:00", "2026-09-22T02:00"],
        "ocean_current_velocity": [1.852, 3.704, None],        # km/h: 1 kt, 2 kt, then missing
        "ocean_current_direction": [90, 180, None]}}]))          # flowing towards east, then south
    c = windows.load_currents(d)
    assert abs(c["east"][0, 0] - 1.0) < 1e-9 and abs(c["north"][0, 0]) < 1e-9, c
    assert abs(c["north"][1, 0] + 2.0) < 1e-9, c                 # towards south is negative north
    assert abs(c["north"][2, 0] + 2.0) < 1e-9, c                 # a missing hour holds the last value


def test_full_brief_builds_from_saved_data(no_performance_polar=False):
    """Smoke test: build a real brief from the newest saved trip data. Skips when there's none.
    With no_performance_polar, no performance polar is configured, as for anyone who hasn't got
    one: the brief must still build from the conservative envelope alone."""
    import trip as T
    import polar
    import report_html
    trips = [t for t in T.Trip.all() if (t.dir / "synoptic").exists() and (t.dir / "runs").exists()]
    if not trips:
        print("  (skipped: no saved trip data)")
        return
    t = trips[0]
    t.use_route()
    run_dir = sorted((t.dir / "runs").iterdir())[-1]
    syn_dir = sorted((t.dir / "synoptic").iterdir())[-1]
    depart = t.window()[0]
    ctx = dict(trip=t.slug, origin=t.origin["name"], dest=t.dest["name"], origin_tz=t.origin_tz, dest_tz=t.dest_tz,
               origin_label="origin time", dest_label="destination time", chart_label="LT")
    saved_pw = polar.PW_FILE
    if no_performance_polar:
        polar.PW_FILE = None
    try:
        b = report_html.build(t.route_key(), depart, run_dir=run_dir, syn_dir=syn_dir, bulletin=None, ctx=ctx)
        page = report_html.render(b, "smoke test")
    finally:
        polar.PW_FILE = saved_pw
    assert "Trip estimates" in page and "Point of sail" in page and "Current" in page, page[:500]
    if (run_dir / "ncep_gfswave025.json").exists() and "swell_wave_height" in (run_dir / "ncep_gfswave025.json").read_text()[:5000]:
        assert 'id="sea"' in page and "Wind sea" in page and "Whole passage" in page, "sea section missing"
    assert b["story"], "no weather story days"
    names = [x["name"] for x in b["polars"]]
    if no_performance_polar:
        assert names == ["Conservative polar"], names
        assert "Performance polar" not in page
    else:
        assert len(names) == 2, names


def test_full_brief_without_a_performance_polar():
    test_full_brief_builds_from_saved_data(no_performance_polar=True)


def test_update_now_button():
    import json, sys, tempfile, types
    import trip as T
    import ha_publish as H
    tmp = Path(tempfile.mkdtemp())
    saved = (T.TRIPS, H.STATE_FILE, H.LOCK_FILE, H.ha_api, H.publish, H.write_index, H.notify)
    T.TRIPS, H.STATE_FILE, H.LOCK_FILE = tmp, tmp / "ha_watch.json", tmp / ".lock"
    ha = {"input_select.passage_watch": "Not watching", H.UPDATE_BUTTON: "unknown",
          H.NEW_TRIP_BUTTON: "unknown", H.CANCEL_BUTTON: "unknown"}
    def fake_api(method, path, payload=None):
        if path.endswith("set_options"):
            ha["input_select.passage_watch"] = payload["options"][0]
        elif path.endswith("select_option"):
            ha["input_select.passage_watch"] = payload["option"]
        elif path.startswith("states/"):
            return {"state": ha[path[len("states/"):]]}
    alerts, screened, briefs = [], [], []
    H.ha_api, H.write_index = fake_api, lambda: None
    H.notify = lambda title, msg, url: alerts.append(title) or "200"
    H.publish = lambda t, days, **kw: alerts.append("Passage: " + t.title()) or "published"
    real = {k: sys.modules.get(k) for k in ("screen", "full_report")}
    sys.modules["screen"] = types.SimpleNamespace(screen=lambda t, run_dir=None: screened.append(t.slug) or (None, None, []))
    sys.modules["full_report"] = types.SimpleNamespace(build_for_trip=lambda t, fresh=None: briefs.append((t.slug, fresh)))
    try:
        T.Trip(slug="w", origin={"name": "A", "lat": -17.8, "lon": 177.4, "tz": "Pacific/Fiji"},
               dest={"name": "B", "lat": -19.5, "lon": 169.5, "tz": "Pacific/Efate"}, earliest="2026-09-22",
               status="watching", watch_depart="2026-09-22 12:00").save()
        assert H.sync_select(log=lambda m: None) == "no change"            # first sight: record only
        ha[H.UPDATE_BUTTON] = "2026-09-15T07:00:00+00:00"
        r = H.sync_select(log=lambda m: None)
        assert screened == ["w"] and len(briefs) == 1 and briefs[0][1].total_seconds() == 3 * 3600, (r, screened, briefs)
        assert alerts == ["Passage: A to B"], alerts
        ha[H.UPDATE_BUTTON] = "2026-09-15T07:05:00+00:00"                   # pressed again straight away
        r = H.sync_select(log=lambda m: None)
        assert r == "update skipped, cooldown" and screened == ["w"] and alerts[-1] == "Passage: update skipped", (r, alerts)
    finally:
        T.TRIPS, H.STATE_FILE, H.LOCK_FILE, H.ha_api, H.publish, H.write_index, H.notify = saved
        for k, v in real.items():
            if v:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)


def test_sea_state_rules():
    import seastate as SS
    assert abs(SS.wavelength(8) - 99.9) < 0.1                               # 1.56 T^2
    assert round(SS.steepness_n(3, 8)) == 33 and SS.steepness_words(33) == "gentle"
    assert SS.steepness_words(10) == "very steep"
    train = lambda h, t, frm, heading=257: SS._train(h, t, frm, heading, 7.0)
    long_swell, short_swell = train(2.2, 15, 204), train(2.2, 8, 204)
    wind = train(1.4, 6, 106)
    assert not SS.cross_sea(wind, long_swell)        # a 15 s swell from 100 deg away is a slow roll, not confused
    assert SS.cross_sea(wind, short_swell)           # a steeper 8 s swell from there is a cross sea
    assert not SS.cross_sea(train(0.8, 6, 106), short_swell)   # under 1 m doesn't count
    assert SS._range(6.0, 6.2, ".0f", "s") == "6 s" and SS._range(1.1, 1.6, ".1f", "m") == "1.1 to 1.6 m"
    # a following wind sea overtakes; a head sea is met faster than its own period
    assert train(1.4, 6, 257 + 180)["enc"] > 6 > train(1.4, 6, 257)["enc"]
    rows = [dict(ecmwf=None, gfs=None, wind=wind, swell=long_swell, swell2=None, cross=False) for _ in range(6)]
    text = SS.describe(SS.summarize(rows))
    assert "Slow roll" in text and "Cross sea" not in text and "6 to 6" not in text and len(text) < 200, text
    rows = [dict(ecmwf=None, gfs=None, wind=wind, swell=short_swell, swell2=None, cross=True) for _ in range(6)]
    assert "Cross sea about 6 h" in SS.describe(SS.summarize(rows))


def test_position_formats():
    import trip as T
    want = (-19.5277, 169.4959)
    close = lambda got, exp=want, tol=0.002: got and abs(got[0] - exp[0]) < tol and abs(got[1] - exp[1]) < tol
    for text in ("-19.5277,169.4959", "-19.5277, 169.4959", "19.5277° S, 169.4959° E", "19.5277 S 169.4959 E",
                 "S 19.5277 E 169.4959", "19 31.66 S 169 29.75 E", "19°31.66'S 169°29.75'E",
                 "S19°31'40\" E169°29'45\"", "169.4959 E, 19.5277 S"):
        assert close(T.parse_position(text)), (text, T.parse_position(text))
    assert T.parse_position("Port Resolution") is None
    # a saved place is found with or without its country written after it
    saved = T.load_places()
    if "port resolution" in saved:
        for text in ("Port Resolution", "Port resolution, Vanuatu "):
            rec, _ = T.resolve_place(text)
            assert abs(rec["lat"] + 19.5277) < 0.01 and rec["tz"] == "Pacific/Efate", (text, rec)
    assert T.parse_position("Nadi, fiji") is None
    assert T.parse_position("95 S, 200 E") is None              # out of range
    north = T.parse_position("21 18.5 N, 157 51.5 W")           # Honolulu, the other hemisphere
    assert north and abs(north[0] - 21.308) < 0.002 and abs(north[1] + 157.858) < 0.002, north


def test_verdict_reason_names_the_check():
    import screen
    slot = dict(verdict="NO", p_tws=0.0, p_tws_no=0.0, p_gust=0.0, p_gust_no=0.0,
                wave_notes=["ECMWF wave model: over 4 m (to 4.3 m), a no-go"])
    text = screen.verdict_reason(slot)
    assert "waves" in text and "ECMWF over 4 m" in text and "Wind stays inside your limits" in text, text
    windy = dict(verdict="WARNING", p_tws=0.18, p_tws_no=0.0, p_gust=0.0, p_gust_no=0.0, wave_notes=[])
    text = screen.verdict_reason(windy)
    assert "Warning on wind" in text and "18% reach 25 kt" in text and "waves" not in text, text
    hard = dict(verdict="NO", p_tws=0.5, p_tws_no=0.2, p_gust=0.4, p_gust_no=0.15, wave_notes=[])
    text = screen.verdict_reason(hard)
    assert "20% of passages reach 30 kt" in text and "15% gust over 40 kt" in text, text
    assert screen.verdict_reason(dict(verdict="GO", p_tws=0.0, p_tws_no=0.0, p_gust=0.0, p_gust_no=0.0,
                                      wave_notes=[])) == "Wind and seas inside your limits."


def test_forecast_runs_out():
    wind, waves = synthetic(16, (COURSE + 140) % 360)
    late = T0 + timedelta(hours=N_T - 20)
    assert windows.sail(0, late, wind, waves, VMC, RT) is None


def test_sunrise_against_open_meteo():
    lat, lon = route.WAYPOINTS[-1][1:]
    d = requests.get("https://api.open-meteo.com/v1/forecast", params=dict(
        latitude=lat, longitude=lon, daily="sunrise,sunset", timezone="GMT",
        start_date="2026-09-25", end_date="2026-09-25"), timeout=60).json()["daily"]
    ref_rise = datetime.fromisoformat(d["sunrise"][0]).replace(tzinfo=timezone.utc)
    ref_set = datetime.fromisoformat(d["sunset"][0]).replace(tzinfo=timezone.utc)
    # Open-Meteo labels the event by the UTC day it falls in; sunrise at Tanna is the previous evening UTC
    rise, sset = None, None
    for day in (ref_rise.date() - timedelta(days=1), ref_rise.date(), ref_rise.date() + timedelta(days=1)):
        r_, s_ = windows.sun_times(day, lat, lon)
        rise = r_ if abs((r_ - ref_rise).total_seconds()) < 3600 else rise
        sset = s_ if abs((s_ - ref_set).total_seconds()) < 3600 else sset
    assert rise and abs((rise - ref_rise).total_seconds()) < 180, (rise, ref_rise)
    assert sset and abs((sset - ref_set).total_seconds()) < 180, (sset, ref_set)


def test_entry_window():
    local = lambda h, m=0: datetime(2026, 9, 25, h, m, tzinfo=windows.VUT).astimezone(timezone.utc)
    assert windows.entry_ok(local(10))
    assert not windows.entry_ok(local(3))
    assert not windows.entry_ok(local(21))


if __name__ == "__main__":
    import os
    os.environ["PASSAGE_NO_HA"] = "1"     # never touch the live Home Assistant from a test
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"pass  {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:   # network trouble, such as an Open-Meteo rate limit
                failed += 1
                print(f"ERROR {name}: {type(e).__name__} {e}")
    raise SystemExit(failed)
