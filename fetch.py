"""Download ensemble wind and wave forecasts for the route sample points.

Source: Open-Meteo, which decodes the official GRIB output of each center.
  ecmwf_ifs025           ECMWF ENS, 51 members, 15 days
  gfs_seamless           NOAA GEFS, 31 members, 0.25 deg to day 10 then 0.5 deg
  ecmwf_wam025, ncep_gfswave025  deterministic waves. Open-Meteo lists wave
    ensembles (ecmwf_wam025_ensemble, ncep_gefswave025) but on 2026-09-15 every
    value they returned was null, so waves come from the two deterministic models.
Each run is saved under data/<UTC fetch time>_<route>/ with the model run times.

Usage: python3 fetch.py [port_resolution|port_vila]
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

import route

HERE = Path(__file__).parent
WIND_MODELS = {"ecmwf_ifs025": "ECMWF ENS", "gfs_seamless": "GEFS"}
WAVE_MODELS = {"ecmwf_wam025": "ECMWF waves", "ncep_gfswave025": "GFS waves"}
META = {
    "ecmwf_ifs025": "https://api.open-meteo.com/data/ecmwf_ifs025_ensemble/static/meta.json",
    "gfs_seamless": "https://api.open-meteo.com/data/ncep_gefs025/static/meta.json",
}


def get(url, params, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            print(f"  retry after {e}", file=sys.stderr)
            time.sleep(5 * (i + 1))


def run_time(model):
    if model not in META:  # no metadata endpoint published for the wave ensemble
        return None
    try:
        m = requests.get(META[model], timeout=30).json()
        t = m.get("last_run_initialisation_time")
        return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H UTC") if t else None
    except Exception:
        return None


def main():
    route.use(sys.argv[1] if len(sys.argv) > 1 else route.NAME)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    fetch_into(HERE / "data" / f"{stamp}_{route.NAME}")


def fetch_into(out, verbose=True):
    """Fetch the wind ensembles and wave models along the current route into out/."""
    pts = route.sample_points()
    lats = ",".join(f"{p[1]:.3f}" for p in pts)
    lons = ",".join(f"{p[2]:.3f}" for p in pts)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"fetched_utc": stamp, "route": route.NAME, "waypoints": route.WAYPOINTS, "points": pts, "models": {}}

    for model, label in WIND_MODELS.items():
        print(f"fetching {label} wind")
        d = get("https://ensemble-api.open-meteo.com/v1/ensemble", {
            "latitude": lats, "longitude": lons, "models": model,
            "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
            "wind_speed_unit": "kn", "forecast_days": 16, "timezone": "GMT"})
        (out / f"{model}.json").write_text(json.dumps(d))
        manifest["models"][model] = {"label": label, "run": run_time(model)}

    # ECMWF gives the combined sea with mean and peak period. GFS-Wave also splits it into wind sea,
    # primary swell and secondary swell (checked 2026-09-16: ECMWF's split fields come back empty).
    wave_vars = {
        "ecmwf_wam025": "wave_height,wave_direction,wave_period,wave_peak_period",
        "ncep_gfswave025": ("wave_height,wave_direction,wave_period,"
                            "wind_wave_height,wind_wave_period,wind_wave_direction,"
                            "swell_wave_height,swell_wave_period,swell_wave_direction,"
                            "secondary_swell_wave_height,secondary_swell_wave_period,secondary_swell_wave_direction"),
    }
    for model, label in WAVE_MODELS.items():
        print(f"fetching {label} waves")
        d = get("https://marine-api.open-meteo.com/v1/marine", {
            "latitude": lats, "longitude": lons, "models": model,
            "hourly": wave_vars.get(model, "wave_height,wave_direction,wave_period"),
            "forecast_days": 16, "timezone": "GMT"})
        (out / f"{model}.json").write_text(json.dumps(d))
        manifest["models"][model] = {"label": label, "run": run_time(model)}

    # Surface currents: Meteo-France SMOC (0.08 deg, hourly, 10 days, tides included) through Open-Meteo.
    # RTOFS was the first choice, but NOMADS retired OPeNDAP subsetting and NOAA's open bucket has only
    # depth-averaged currents in the global 2D files (checked 2026-09-15).
    print("fetching SMOC currents")
    d = get("https://marine-api.open-meteo.com/v1/marine", {
        "latitude": lats, "longitude": lons,
        "hourly": "ocean_current_velocity,ocean_current_direction", "forecast_days": 10, "timezone": "GMT"})
    (out / "currents.json").write_text(json.dumps(d))
    manifest["models"]["currents"] = {"label": "SMOC currents", "run": None}

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if verbose:
        print(f"saved {out}")
        print(json.dumps(manifest["models"], indent=1))
    return out


if __name__ == "__main__":
    main()
