"""Along-track synoptic diagnostics, regime calls and charts for one passage.

Follows the boat: at each 6-hourly model time the diagnostics are read at the
median position of the ensemble tracks from daily.py (conservative polar).

Rules of thumb used for the calls (tropical ocean, not calibrated on a logged passage):
  Squalls   HIGH      PWAT >= 50 mm, RH700 >= 60 %, and CAPE >= 1000 J/kg or K >= 32
            MODERATE  PWAT >= 45 mm, RH700 >= 50 %, and CAPE >= 500 J/kg or K >= 28
            ISOLATED  PWAT >= 35 mm, RH700 >= 40 %, K >= 15
            LOW       otherwise. RH700 under 20 % means the trade inversion caps showers.
  Surge     850 hPa cold advection of 3 K/day or more with MSLP rising: a front or
            trough passing to the south and a high building behind it
  Upper trough  500 hPa cyclonic geostrophic vorticity >= 2e-5 /s at the boat
  Squall gust   in a squall a downdraft can bring the 850 hPa wind to the surface, so
                the gust potential is at least the 850 hPa wind (momentum transfer idea)
"""
import math
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np

import route
import synoptic as S
import windows

HERE = Path(__file__).parent
FJT = windows.FJT

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8983", "#e4e3de", "#fcfcfb"
MODEL_COLORS = {"ECMWF": "#2a78d6", "GFS": "#eb6834", "ICON": "#1baf7a"}   # others muted gray
OTHER = "#b5b3ab"
TZ = {"zone": FJT, "label": "FJT"}     # set by set_timezone() for a trip


def set_timezone(tz, label):
    """Times on charts and in the discussion are shown in this zone (the origin's)."""
    TZ["zone"], TZ["label"] = tz, label


def fmt_pos(lat, lon):
    lon = lon % 360
    return (f"{abs(lat):.0f}{'S' if lat < 0 else 'N'} "
            f"{lon if lon <= 180 else 360 - lon:.0f}{'E' if lon <= 180 else 'W'}")

_STYLE = {"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
          "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
          "axes.titlecolor": INK, "axes.titleweight": "bold", "figure.facecolor": SURFACE,
          "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE}


def charts_available():
    """matplotlib is optional: without it the brief is built without maps."""
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(_STYLE)
    return plt


def median_track(members):
    """Function t -> distance along route, median across member tracks."""
    tracks = [tr for _, _, _, tr in members]
    def dist_at(t):
        ds = []
        for tr in tracks:
            ts = [x[0] for x in tr]
            if t <= ts[0]:
                ds.append(0.0)
            elif t >= ts[-1]:
                ds.append(route.total_nm())
            else:
                k = max(i for i, x in enumerate(ts) if x <= t)
                (t0, d0), (t1, d1) = tr[k], tr[k + 1]
                f = (t - t0).total_seconds() / max((t1 - t0).total_seconds(), 1)
                ds.append(d0 + (d1 - d0) * f)
        return float(np.median(ds))
    return dist_at


def squall_risk(pwat, rh700, cape, k):
    cape = 0 if cape is None or math.isnan(cape) else cape
    if pwat >= 50 and rh700 >= 60 and (cape >= 1000 or k >= 32):
        return "HIGH"
    if pwat >= 45 and rh700 >= 50 and (cape >= 500 or k >= 28):
        return "MODERATE"
    if pwat >= 35 and rh700 >= 40 and k >= 15:
        return "ISOLATED"
    return "LOW"


POINT_FIELDS = ["mslp", "wind10_kt", "gust10_kt", "wind850_kt", "pwat", "cape", "k_index", "rh700", "precip6", "t850"]
GRID_FIELDS = ["tadv850", "cyc500", "div250", "jet250_kt"]


def along_track(models, points, depart_utc, arrive_utc, dist_at):
    """{label: [row per 6-hourly step inside the passage]}.
    Point fields come from the 0.25 deg route points, gradient fields from the grid."""
    out = {}
    for key, m in models.items():
        pm = points.get(key)
        if pm is None:
            continue
        rows = []
        for ti, t in enumerate(m.times):
            if t < depart_utc - timedelta(hours=6) or t > arrive_utc + timedelta(hours=6) or not (m.valid[ti] and pm.valid[ti]):
                continue
            nm = dist_at(t)
            lat, lon, crs = route.position_at(nm)
            row = dict(t=t, nm=nm, lat=lat, lon=lon, ti=ti)
            for f in POINT_FIELDS:
                row[f] = pm.at(f, ti, nm)
            for f in GRID_FIELDS:
                row[f] = m.at(f, ti, lat, lon)
            row["wdir10"] = pm.wdir(ti, nm)
            row["squall"] = squall_risk(row["pwat"], row["rh700"], row["cape"], row["k_index"])
            # 24 h pressure change at the boat's position, which removes the twice-daily atmospheric tide
            if ti >= 4 and pm.valid[ti - 4]:
                row["dp24"] = pm.at("mslp", ti, nm) - pm.at("mslp", ti - 4, nm)
            else:
                row["dp24"] = float("nan")
            rows.append(row)
        out[m.label] = rows
    return out


def compass(deg):
    pts = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return pts[int((deg % 360) / 22.5 + 0.5) % 16]


def discussion(models, track, depart_fjt, n_days):
    """Plain-language synoptic discussion per passage day, built from the numbers."""
    ec, gf = track.get("ECMWF", []), track.get("GFS", [])
    days = []
    for k in range(n_days):
        t0 = (depart_fjt + timedelta(days=k)).astimezone(timezone.utc)
        t1 = t0 + timedelta(days=1)
        er = [r for r in ec if t0 <= r["t"] < t1]
        gr = [r for r in gf if t0 <= r["t"] < t1]
        if not er:
            break
        mid = er[len(er) // 2]
        m_ec = models["ecmwf_ifs025"]
        highs = m_ec.centers(mid["ti"], "high")
        south = mid["lat"] < 0
        highs = [h for h in highs if abs(h["lat"]) >= 20 and (h["lat"] < 0) == south]
        lows = [l for l in m_ec.centers(mid["ti"], "low") if l["hpa"] < 1012]
        jet = m_ec.jet_core(mid["ti"], mid["lon"], south=south)
        lines = []
        # surface pattern
        if highs:
            h = highs[0]
            lines.append(f"High {h['hpa']:.0f} hPa near {fmt_pos(h['lat'], h['lon'])}.")
        for l in lows[:2]:
            lines.append(f"Low {l['hpa']:.0f} hPa near {fmt_pos(l['lat'], l['lon'])}.")
        w = [r["wind10_kt"] for r in er]
        wd = [r["wdir10"] for r in er]
        lines.append(f"At the boat ECMWF has {min(w):.0f} to {max(w):.0f} kt from the {compass(np.median(wd))}"
                     + (f", GFS {min(r['wind10_kt'] for r in gr):.0f} to {max(r['wind10_kt'] for r in gr):.0f} kt." if gr else "."))
        w850 = max(r["wind850_kt"] for r in er)
        # moisture and stability
        pw, rh, kk = max(r["pwat"] for r in er), max(r["rh700"] for r in er), max(r["k_index"] for r in er)
        cape = max(r["cape"] for r in er)
        risk = max((r["squall"] for r in er), key=["LOW", "ISOLATED", "MODERATE", "HIGH"].index)
        if risk == "LOW" and rh < 20:
            lines.append(f"Dry above the trade inversion (700 hPa humidity up to {rh:.0f}%, precipitable water up to "
                         f"{pw:.0f} mm, K-index up to {kk:.0f}). Showers stay shallow. Squall risk low.")
        elif risk == "LOW":
            lines.append(f"Precipitable water up to {pw:.0f} mm, 700 hPa humidity up to {rh:.0f}%, K-index up to {kk:.0f}, "
                         f"CAPE up to {cape:.0f} J/kg. Squall risk low.")
        elif risk == "ISOLATED":
            lines.append(f"Moister mid levels for a while: precipitable water up to {pw:.0f} mm, 700 hPa humidity up to {rh:.0f}%, "
                         f"K-index up to {kk:.0f}, CAPE up to {cape:.0f} J/kg. An isolated shower or squall is possible.")
        else:
            lines.append(f"Moist, unstable column: precipitable water up to {pw:.0f} mm, 700 hPa humidity up to {rh:.0f}%, "
                         f"K-index up to {kk:.0f}, CAPE up to {cape:.0f} J/kg. Squall risk {risk.lower()}.")
        if risk != "LOW" and w850 > max(w) + 3:
            lines.append(f"The wind at 850 hPa reaches {w850:.0f} kt, stronger than at the surface, and a squall downdraft "
                         "can bring gusts near that speed down to the water.")
        # advection, pressure tendency, upper air
        cold = min(r["tadv850"] for r in er)
        rise = np.nanmax([r["dp24"] for r in er]) if any(np.isfinite(r["dp24"]) for r in er) else float("nan")
        if cold <= -3 and rise > 1:
            lines.append(f"Cold advection at 850 hPa ({cold:.0f} K/day) with pressure up {rise:.1f} hPa in 24 hours: "
                         "a front or trough passing to the south and a high building behind it. Expect the trade wind to freshen and back toward the south.")
        elif cold <= -3:
            lines.append(f"Cold advection at 850 hPa ({cold:.0f} K/day): cooler, drier air moving in from the south.")
        cyc = max(r["cyc500"] for r in er)
        if cyc >= 2:
            lines.append(f"Upper trough near the boat (500 hPa cyclonic vorticity {cyc:.1f}e-5/s), which lifts the column and helps showers grow.")
        elif min(r["cyc500"] for r in er) <= -1:
            lines.append("Ridge aloft at 500 hPa over the boat, which means sinking air and settled weather.")
        if jet and jet["kt"] >= 60:
            gap = abs(jet["lat"]) - abs(mid["lat"])
            where = (f"well {'south' if south else 'north'} of the route" if gap >= 5 else
                     "close to the route, where shear aloft can tilt or tear shower tops")
            lines.append(f"Jet at 250 hPa {jet['kt']:.0f} kt near {fmt_pos(jet['lat'], jet['lon'])}, {where}.")
        # model agreement at the boat
        spreads = []
        for r in er:
            vals = [rows_at(track[lbl], r["t"])["wind10_kt"] for lbl in track if rows_at(track[lbl], r["t"])]
            if len(vals) >= 3:
                spreads.append(max(vals) - min(vals))
        if spreads:
            n = len([lbl for lbl in track if any(t0 <= x["t"] < t1 for x in track[lbl])])
            lines.append(f"The {n} models are within {max(spreads):.0f} kt of each other at the boat."
                         if max(spreads) <= 6 else
                         f"The {n} models disagree by up to {max(spreads):.0f} kt at the boat, so treat the wind figures with caution.")
        days.append(dict(day=k + 1, start=t0, risk=risk, text=" ".join(lines), rows=er))
    return days


def arrival_note(track, arrive_utc):
    """Conditions at the destination around the median arrival, all models."""
    parts = []
    for lbl, rows in track.items():
        near = [r for r in rows if abs((r["t"] - arrive_utc).total_seconds()) <= 6 * 3600]
        if near:
            r = min(near, key=lambda x: abs((x["t"] - arrive_utc).total_seconds()))
            parts.append((lbl, r))
    if not parts:
        return ""
    ec = dict(parts).get("ECMWF", parts[0][1])
    winds = ", ".join(f"{lbl} {r['wind10_kt']:.0f}" for lbl, r in parts)
    txt = (f"Around arrival the models give {winds} kt, ECMWF from the {compass(ec['wdir10'])}.")
    colds = [r["tadv850"] for _, r in parts]
    rises = [r["dp24"] for _, r in parts if np.isfinite(r["dp24"])]
    if min(colds) <= -3 and rises and max(rises) > 1:
        txt += (f" Cold advection at 850 hPa (down to {min(colds):.0f} K/day) and pressure up to {max(rises):.1f} hPa "
                "higher than a day earlier: a front has passed to the south and the trade wind is freshening behind it.")
    return txt


def rows_at(rows, t):
    for r in rows:
        if r["t"] == t:
            return r
    return None


# ---------------------------------------------------------------- charts

def _lon_label(x, _):
    x = x % 360
    return f"{x:.0f}E" if x <= 180 else f"{360 - x:.0f}W"


def surface_chart(m, ti, path, boat=None, route_name=None):
    plt = _plt()
    d = m.d
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    lon2, lat2 = np.meshgrid(m.lons, m.lats)
    pw = ax.contourf(lon2, lat2, d["pwat"][ti], levels=np.arange(10, 66, 5), cmap="Blues", extend="both")
    cb = fig.colorbar(pw, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Precipitable water, mm", color=INK2)
    cb.outline.set_edgecolor(GRID)
    cs = ax.contour(lon2, lat2, d["mslp"][ti], levels=np.arange(990, 1040, 2), colors=INK, linewidths=0.8)
    ax.clabel(cs, fmt="%d", fontsize=7, inline=True)
    step = 1
    ax.barbs(lon2[::step, ::step], lat2[::step, ::step], d["u10"][ti][::step, ::step] * S.MS2KT,
             d["v10"][ti][::step, ::step] * S.MS2KT, length=5, linewidth=0.6, color=INK2, barb_increments=dict(half=5, full=10, flag=50))
    for kind, sym in (("high", "H"), ("low", "L")):
        for c in m.centers(ti, kind)[:4]:
            ax.text(c["lon"], c["lat"], sym, ha="center", va="center", fontsize=14, fontweight="bold",
                    color="#2a78d6" if sym == "H" else "#e34948")
            ax.text(c["lon"], c["lat"] - 1.2, f"{c['hpa']:.0f}", ha="center", va="top", fontsize=7, color=INK2)
    _route_and_places(ax, m, boat, route_name)
    ax.set_title(f"{m.label} surface: MSLP, 10 m wind, precipitable water\n"
                 f"valid {m.times[ti].astimezone(TZ['zone']):%a %d %b %H:%M} {TZ['label']}", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def upper_chart(m, ti, path, boat=None, route_name=None):
    plt = _plt()
    d = m.d
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    lon2, lat2 = np.meshgrid(m.lons, m.lats)
    jet = ax.contourf(lon2, lat2, d["jet250_kt"][ti], levels=np.arange(60, 181, 20), cmap="Purples", extend="max")
    cb = fig.colorbar(jet, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("250 hPa wind, kt", color=INK2)
    cb.outline.set_edgecolor(GRID)
    cs = ax.contour(lon2, lat2, d["z500"][ti], levels=np.arange(5400, 6000, 30), colors=INK, linewidths=0.8)
    ax.clabel(cs, fmt="%d", fontsize=7, inline=True)
    cyc = np.nan_to_num(d["cyc500"][ti])
    ax.contour(lon2, lat2, cyc, levels=[2, 4, 6], colors="#e34948", linewidths=0.9, linestyles="--")
    _route_and_places(ax, m, boat, route_name)
    ax.set_title(f"{m.label} upper air: 500 hPa height (m), 250 hPa jet\nred dashes: 500 hPa troughs, "
                 f"valid {m.times[ti].astimezone(TZ['zone']):%a %d %b %H:%M} {TZ['label']}", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _route_and_places(ax, m, boat, route_name):
    plt = _plt()
    for name in ([route_name] if route_name else [route.NAME]):
        wp = route.ROUTES[name]
        pts = []
        for (_, a1, o1), (_, a2, o2) in zip(wp, wp[1:]):
            pts += [route.gc_interp(a1, o1, a2, o2, f) for f in np.linspace(0, 1, 20)]
        ax.plot([p[1] % 360 for p in pts], [p[0] for p in pts], color="#eb6834", linewidth=2)
    wps = route.ROUTES[route_name or route.NAME]
    for name, la, lo in (wps[0], wps[-1]):
        ax.plot(lo % 360, la, "o", color=INK, markersize=3)
        ax.text(lo % 360 + 0.4, la + 0.3, name, fontsize=7, color=INK)
    if boat:
        ax.plot(boat[1] % 360, boat[0], "o", markersize=9, markerfacecolor="#eb6834", markeredgecolor=SURFACE,
                markeredgewidth=2)
    ax.set_xlim(m.lons[0], m.lons[-1])
    ax.set_ylim(m.lats[-1], m.lats[0])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_lon_label))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{abs(y):.0f}{'S' if y < 0 else 'N' if y > 0 else ''}"))
    ax.set_aspect(1 / math.cos(math.radians(abs(float(np.mean(m.lats))))))
    ax.grid(color=GRID, linewidth=0.5)


def track_panels(track, path, depart_utc, arrive_utc):
    """Small multiples along the boat's track, one measure per panel, shared time axis."""
    plt = _plt()
    panels = [("wind10_kt", "10 m wind, kt", None), ("wind850_kt", "850 hPa wind, kt", None),
              ("pwat", "Precipitable water, mm", (35, 50)), ("rh700", "700 hPa humidity, %", (40, 60)),
              ("k_index", "K-index", (15, 28)), ("tadv850", "850 hPa temperature advection, K/day", (-3,)),
              ("dp24", "24 h pressure change, hPa", (1,)), ("cyc500", "500 hPa cyclonic vorticity, 1e-5/s", (2,))]
    import matplotlib.dates as mdates
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 13), sharex=True)
    for ax, (key, label, thresholds) in zip(axes, panels):
        for lbl, rows in track.items():
            if not rows:
                continue
            color = MODEL_COLORS.get(lbl, OTHER)
            ax.plot([r["t"] for r in rows], [r[key] for r in rows], color=color,
                    linewidth=2 if lbl in MODEL_COLORS else 1, zorder=3 if lbl in MODEL_COLORS else 2, label=lbl)
        for th in thresholds or ():
            ax.axhline(th, color=MUTED, linewidth=0.8, linestyle=":")
        ax.axvspan(depart_utc, arrive_utc, color="#f1f0ec", zorder=0)
        ax.set_title(label, loc="left", fontsize=9)
        ax.grid(color=GRID, linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.0))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d\n%H:%M", tz=TZ["zone"]))
    axes[-1].set_xlabel(f"{TZ['label']}. Shaded: the passage. Dotted: thresholds used in the squall and surge calls.")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130)
    plt.close(fig)
