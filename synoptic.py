"""Synoptic analysis for the Fiji to Vanuatu passages: the upper and lower atmosphere.

Fetches a 2.5 deg grid (5S to 40S, 155E to 170W) every 6 hours from several
deterministic models and computes the diagnostics a forecaster would read:

  Surface   MSLP, high and low centers, the pressure gradient as a geostrophic
            wind, 10 m wind and its convergence
  850 hPa   wind (the momentum a squall downdraft can bring down), temperature
            advection (cold air behind a front or a southerly surge)
  700 hPa   humidity: dry air above the trade inversion suppresses deep showers
  500 hPa   heights and geostrophic vorticity: upper troughs lift and destabilize
  250 hPa   jet position and speed, divergence aloft
  Column    precipitable water, CAPE, K-index, Total Totals

The 2.5 deg grid is for the large-scale pattern and gradients. Surface and
column values at the boat come from each model at the 0.25 deg route points,
because a coarse grid point near Fiji or Vanuatu can sit on land.

Usage: python3 synoptic.py fetch [port_vila port_resolution ...]
"""
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests


def _window_extreme(a, size, biggest):
    """Largest (or smallest) value in each size x size window, edges clamped.
    Replaces scipy.ndimage.maximum_filter/minimum_filter so the app needs no scipy."""
    r = size // 2
    shifted = []
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            rows = np.clip(np.arange(a.shape[0]) + di, 0, a.shape[0] - 1)
            cols = np.clip(np.arange(a.shape[1]) + dj, 0, a.shape[1] - 1)
            shifted.append(a[np.ix_(rows, cols)])
    stack = np.stack(shifted)
    return stack.max(axis=0) if biggest else stack.min(axis=0)


HERE = Path(__file__).parent
LATS = np.arange(-5.0, -40.01, -2.5)
LONS = np.arange(155.0, 190.01, 2.5)            # 190 = 170W
MODELS = {
    "ecmwf_ifs025": "ECMWF",
    "gfs_global": "GFS",
    "icon_global": "ICON",
    "ukmo_global_deterministic_10km": "UKMO",
    "ecmwf_aifs025_single": "AIFS",
    "gem_global": "GEM",
}
VARS = ["pressure_msl", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
        "temperature_850hPa", "relative_humidity_850hPa", "wind_speed_850hPa", "wind_direction_850hPa",
        "temperature_700hPa", "relative_humidity_700hPa", "wind_speed_700hPa", "wind_direction_700hPa",
        "temperature_500hPa", "geopotential_height_500hPa", "wind_speed_250hPa", "wind_direction_250hPa",
        "cape", "lifted_index", "precipitation", "total_column_integrated_water_vapour"]
META = {"ecmwf_ifs025": "ecmwf_ifs025", "gfs_global": "ncep_gfs025", "icon_global": "dwd_icon",
        "ukmo_global_deterministic_10km": "ukmo_global_deterministic_10km",
        "ecmwf_aifs025_single": "ecmwf_aifs025_single", "gem_global": "cmc_gem_gdps"}

MS2KT = 1.943844
OMEGA, R_EARTH, G, RHO = 7.292e-5, 6.371e6, 9.80665, 1.2


# ---------------------------------------------------------------- fetch

def run_time(model):
    try:
        m = requests.get(f"https://api.open-meteo.com/data/{META[model]}/static/meta.json", timeout=30).json()
        t = m.get("last_run_initialisation_time")
        if not t or time.time() - t > 2 * 86400:   # stale metadata seen for GEM on 2026-09-15
            return None
        return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H UTC")
    except Exception:
        return None


def _post(lat_s, lon_s, model):
    # Open-Meteo weights a call by locations and variables; a full grid is a few hundred
    # calls against a 600 per minute allowance, so pause and retry on 429.
    for attempt in range(6):
        r = requests.post("https://api.open-meteo.com/v1/forecast", data=dict(
            latitude=lat_s, longitude=lon_s, hourly=",".join(VARS), models=model, forecast_days=16,
            temporal_resolution="hourly_6", timezone="GMT", wind_speed_unit="ms"), timeout=300)
        if r.status_code != 429:
            break
        print(f"  rate limited, waiting 70 s")
        time.sleep(70)
    r.raise_for_status()
    return r.json()


def _to_array(pts, shape_tail):
    times = pts[0]["hourly"]["time"]
    arr = np.full((len(VARS), len(times), len(pts)), np.nan, dtype=np.float32)
    for k, pt in enumerate(pts):
        for v, name in enumerate(VARS):
            col = pt["hourly"].get(name)
            if col:
                arr[v, :, k] = [np.nan if x is None else x for x in col]
    return arr.reshape((len(VARS), len(times)) + shape_tail), times


def domain_for(waypoints, max_points=256):
    """Grid for the synoptic pattern around a route: 20 deg poleward of it, where the highs and
    fronts that drive the trades sit, 12 deg equatorward, 15 deg either side in longitude.
    Latitudes run from north to south; longitudes are 0-360 and may pass 180."""
    lats = [w[1] for w in waypoints]
    lons = np.unwrap(np.radians([w[2] % 360 for w in waypoints]))
    lons = np.degrees(lons)
    south = np.mean(lats) < 0
    lat_n = min(max(lats) + (12 if south else 20), 80)
    lat_s = max(min(lats) - (20 if south else 12), -80)
    lon_w, lon_e = min(lons) - 15, max(lons) + 15
    step = 2.5
    while True:
        la = np.arange(math.ceil(lat_n / step) * step, lat_s - 1e-6, -step)
        lo = np.arange(math.floor(lon_w / step) * step, lon_e + 1e-6, step)
        if la.size * lo.size <= max_points:
            return la, lo % 360 if lo.min() < 0 else lo, step
        step += 0.5


def fetch(route_names=("port_vila", "port_resolution"), out=None):
    """Grid around the listed routes plus each model at the route points."""
    import route as R
    wps = [w for n in route_names for w in R.ROUTES[n]]
    LA, LO, step = domain_for(wps)
    la, lo = np.meshgrid(LA, LO, indexing="ij")
    lat_s = ",".join(f"{x:.2f}" for x in la.ravel())
    lon_s = ",".join(f"{((x + 180) % 360) - 180:.2f}" for x in lo.ravel())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    out = out or HERE / "data" / f"synoptic_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"fetched_utc": stamp, "lats": LA.tolist(), "lons": LO.tolist(), "step": step, "models": {}}
    for model, label in MODELS.items():
        print(f"fetching {label}")
        try:
            arr, times = _to_array(_post(lat_s, lon_s, model), (len(LA), len(LO)))
        except requests.RequestException as e:      # one model missing shouldn't sink the brief
            print(f"  {label} skipped: {e}")
            continue
        np.savez_compressed(out / f"{model}.npz", data=arr, times=np.array(times), lats=LA, lons=LO, step=step)
        valid = [t for n, t in enumerate(times) if np.isfinite(arr[0, n]).any()]
        manifest["models"][model] = {"label": label, "run": run_time(model), "last_valid": valid[-1] if valid else None,
                                     "vars_present": [n for v, n in enumerate(VARS) if np.isfinite(arr[v]).any()]}
    for name in route_names:
        R.use(name)
        pts = R.sample_points()
        manifest.setdefault("routes", {})[name] = pts
        for model, label in MODELS.items():
            print(f"fetching {label} along {name}")
            try:
                arr, times = _to_array(_post(",".join(f"{p[1]:.3f}" for p in pts), ",".join(f"{p[2]:.3f}" for p in pts),
                                             model), (len(pts),))
            except requests.RequestException as e:
                print(f"  {label} along {name} skipped: {e}")
                continue
            np.savez_compressed(out / f"route_{name}_{model}.npz", data=arr, times=np.array(times),
                                nm=np.array([p[0] for p in pts]))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"saved {out}")
    return out


# ---------------------------------------------------------------- load and derive

class Model:
    """One model's grid with derived fields. Arrays are [time, lat, lon]."""

    def __init__(self, run_dir, model):
        z = np.load(run_dir / f"{model}.npz")
        self.name, self.label = model, MODELS[model]
        # grids saved before domain_for existed used the fixed Fiji-Vanuatu grid
        self.lats = z["lats"] if "lats" in z else LATS
        self.lons = z["lons"] if "lons" in z else LONS
        self.step = float(z["step"]) if "step" in z else 2.5
        self.times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in z["times"]]
        self.raw = {n: z["data"][v].astype(float) for v, n in enumerate(VARS)}
        valid = np.isfinite(self.raw["pressure_msl"]).all(axis=(1, 2))
        self.valid = valid
        self._derive()

    def _uv(self, level):
        spd, d = self.raw[f"wind_speed_{level}"], np.radians(self.raw[f"wind_direction_{level}"])
        return -spd * np.sin(d), -spd * np.cos(d)

    def _derive(self):
        lat = np.radians(self.lats)[None, :, None]
        f = 2 * OMEGA * np.sin(lat)                                   # negative in the south
        dy = np.radians(self.step) * R_EARTH                          # latitudes decrease with index
        dx = np.radians(self.step) * R_EARTH * np.cos(lat)

        def ddx(a):
            return np.gradient(a, axis=2) / dx

        def ddy(a):
            return -np.gradient(a, axis=1) / dy                      # index increases southward

        r = self.raw
        d = {}
        d["mslp"] = r["pressure_msl"]
        # geostrophic wind from the MSLP field
        p = r["pressure_msl"] * 100
        ug, vg = -ddy(p) / (RHO * f), ddx(p) / (RHO * f)
        d["geo_kt"] = np.hypot(ug, vg) * MS2KT
        u10, v10 = self._uv("10m")
        d["wind10_kt"], d["gust10_kt"] = r["wind_speed_10m"] * MS2KT, r["wind_gusts_10m"] * MS2KT
        d["u10"], d["v10"] = u10, v10
        d["conv10"] = -(ddx(u10) + ddy(v10)) * 1e5                   # 1e-5 /s, positive = convergence
        u85, v85 = self._uv("850hPa")
        u70, v70 = self._uv("700hPa")
        d["wind850_kt"], d["wind700_kt"] = r["wind_speed_850hPa"] * MS2KT, r["wind_speed_700hPa"] * MS2KT
        t85 = r["temperature_850hPa"]
        d["tadv850"] = -(u85 * ddx(t85) + v85 * ddy(t85)) * 86400     # K per day, negative = cold advection
        d["t850"] = t85
        # dew points by the Magnus formula
        def dew(t, rh):
            g = np.log(np.clip(rh, 1, 100) / 100) + 17.625 * t / (243.04 + t)
            return 243.04 * g / (17.625 - g)
        td85, td70 = dew(t85, r["relative_humidity_850hPa"]), dew(r["temperature_700hPa"], r["relative_humidity_700hPa"])
        t70, t50 = r["temperature_700hPa"], r["temperature_500hPa"]
        d["k_index"] = (t85 - t50) + td85 - (t70 - td70)
        d["total_totals"] = t85 + td85 - 2 * t50
        d["rh700"] = r["relative_humidity_700hPa"]
        d["t500"] = t50
        # 500 hPa geostrophic relative vorticity, signed so cyclonic is positive in either hemisphere
        z5 = r["geopotential_height_500hPa"]
        d["z500"] = z5
        ug5, vg5 = -G * ddy(z5) / f, G * ddx(z5) / f
        zeta = ddx(vg5) - ddy(ug5)
        d["cyc500"] = zeta * np.sign(f) * 1e5                         # 1e-5 /s, positive = cyclonic (trough)
        u25, v25 = self._uv("250hPa")
        d["jet250_kt"] = r["wind_speed_250hPa"] * MS2KT
        d["div250"] = (ddx(u25) + ddy(v25)) * 1e5                     # 1e-5 /s, positive = divergence aloft
        d["cape"], d["li"] = r["cape"], r["lifted_index"]
        d["pwat"] = r["total_column_integrated_water_vapour"]
        d["precip6"] = r["precipitation"]
        # f is small near the equator, so geostrophic fields are masked within 10 deg of it
        for k in ("geo_kt", "cyc500"):
            d[k] = np.where(np.abs(self.lats)[None, :, None] < 10, np.nan, d[k])
        self.d = d

    def at(self, field, t_index, lat, lon):
        """Bilinear value of a field at a point."""
        a = self.d[field][t_index]
        fi = (self.lats[0] - lat) / self.step
        fj = ((lon - self.lons[0]) % 360) / self.step
        i0 = int(np.clip(math.floor(fi), 0, len(self.lats) - 2))
        j0 = int(np.clip(math.floor(fj), 0, len(self.lons) - 2))
        wi, wj = fi - i0, fj - j0
        return float(a[i0, j0] * (1 - wi) * (1 - wj) + a[i0 + 1, j0] * wi * (1 - wj)
                     + a[i0, j0 + 1] * (1 - wi) * wj + a[i0 + 1, j0 + 1] * wi * wj)

    def centers(self, t_index, kind="high", size=3):
        """High or low pressure centers: local extremes that stand out from their surroundings."""
        p = self.d["mslp"][t_index]
        if not np.isfinite(p).all():
            return []
        ext = (p == _window_extreme(p, size, kind == "high"))
        out = []
        for i, j in zip(*np.nonzero(ext)):
            if 0 < i < len(self.lats) - 1 and 0 < j < len(self.lons) - 1:
                ring = p[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3]
                prominence = p[i, j] - ring.mean() if kind == "high" else ring.mean() - p[i, j]
                if prominence > 0.5:
                    out.append(dict(lat=float(self.lats[i]), lon=float(self.lons[j]), hpa=float(p[i, j])))
        return sorted(out, key=lambda c: -c["hpa"] if kind == "high" else c["hpa"])

    def jet_core(self, t_index, lon, south=True, half_width=10):
        """Strongest 250 hPa wind poleward of 15 deg within half_width deg of longitude."""
        a = self.d["jet250_kt"][t_index]
        js = np.abs((self.lons - lon % 360 + 180) % 360 - 180) <= half_width
        is_ = self.lats <= -15 if south else self.lats >= 15
        sub = a[np.ix_(is_, js)]
        if not np.isfinite(sub).any():
            return None
        i, j = np.unravel_index(np.nanargmax(sub), sub.shape)
        return dict(lat=float(self.lats[is_][i]), lon=float(self.lons[js][j]), kt=float(sub[i, j]))


def dew_point(t, rh):
    """Magnus formula, deg C."""
    g = np.log(np.clip(rh, 1, 100) / 100) + 17.625 * t / (243.04 + t)
    return 243.04 * g / (17.625 - g)


class RoutePoints:
    """One model at the route sample points. Arrays are [time, point]."""

    def __init__(self, run_dir, route_name, model):
        z = np.load(run_dir / f"route_{route_name}_{model}.npz")
        self.label = MODELS[model]
        self.times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in z["times"]]
        self.nm = z["nm"]
        r = {n: z["data"][v].astype(float) for v, n in enumerate(VARS)}
        t85, t70, t50 = r["temperature_850hPa"], r["temperature_700hPa"], r["temperature_500hPa"]
        td85, td70 = dew_point(t85, r["relative_humidity_850hPa"]), dew_point(t70, r["relative_humidity_700hPa"])
        d = dict(mslp=r["pressure_msl"], wind10_kt=r["wind_speed_10m"] * MS2KT, gust10_kt=r["wind_gusts_10m"] * MS2KT,
                 wind850_kt=r["wind_speed_850hPa"] * MS2KT, pwat=r["total_column_integrated_water_vapour"],
                 cape=r["cape"], li=r["lifted_index"], rh700=r["relative_humidity_700hPa"], precip6=r["precipitation"],
                 t850=t85, k_index=(t85 - t50) + td85 - (t70 - td70), total_totals=t85 + td85 - 2 * t50)
        a = np.radians(r["wind_direction_10m"])
        d["sin10"], d["cos10"] = np.sin(a), np.cos(a)
        self.d = d
        self.valid = np.isfinite(r["pressure_msl"]).all(axis=1)

    def at(self, field, ti, nm):
        return float(np.interp(nm, self.nm, self.d[field][ti]))

    def wdir(self, ti, nm):
        return (math.degrees(math.atan2(self.at("sin10", ti, nm), self.at("cos10", ti, nm))) + 360) % 360


def load_route(run_dir, route_name):
    return {m: RoutePoints(run_dir, route_name, m) for m in MODELS
            if (run_dir / f"route_{route_name}_{m}.npz").exists()}


def load_all(run_dir):
    return {m: Model(run_dir, m) for m in MODELS if (run_dir / f"{m}.npz").exists()}


def latest_dir():
    return sorted((HERE / "data").glob("synoptic_*"))[-1]


if __name__ == "__main__":
    if sys.argv[1:2] == ["fetch"]:
        fetch(tuple(sys.argv[2:]) or ("port_vila", "port_resolution"))
