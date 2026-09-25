# Passage weather

Ensemble departure planning for a cruising sailboat. You give it two places and
the first day you could leave. It sails every member of two weather ensembles
along the great circle between them, once per candidate day, and reports what
each departure would actually be like: wind, gusts, seas, point of sail, motoring
hours, fuel, arrival time and how rough the ride would be.

It runs on your own machine, in a browser tab, with no account and no paid data.

This is a planning guide, not a navigation tool. The captain develops the
detailed route for every passage.

## What you get

Two documents per trip.

**The screening** is one card per candidate departure day, GO, WARNING or NO,
with the reason on the card, plus a history table so you can see whether a day is
getting better or worse with each forecast run.

**The full brief** covers the day you pick: trip estimates from two boat speed
models, a day by day table, the sea state, a synoptic weather story with charts,
along-track diagnostic panels, the official text forecast for the area, and a
method section that states every threshold used.

## Running it

Python 3.10 or newer, numpy and requests. Charts need matplotlib.

    pip install -e ".[charts]"
    passage-weather

That opens http://127.0.0.1:8765. Everything happens there: new trip, screen the
window, pick a day, build the brief, and edit the boat's polars and limits.

Without installing, `python3 webapp.py` does the same thing. `--lan` listens on
the local network so a tablet can reach it, `--port` moves it, `--no-browser`
leaves the browser alone.

There's also a CLI, if you'd rather script it or run it from a timer:

    python3 passage.py plan "Nadi" "Port Resolution" --earliest 2026-09-22
    python3 passage.py screen <trip>
    python3 passage.py watch <trip> 2026-09-23
    python3 passage.py report <trip>
    python3 passage.py install-timer      # updates at 08:30 and 20:30 UTC

`python3 passage.py` on its own lists every command.

## The forecast

| Input | Source | Members |
|---|---|---|
| Wind, gusts | ECMWF ENS 0.25° | 51 |
| Wind, gusts | NOAA GEFS 0.25°, 0.5° after day 10 | 31 |
| Waves, periods, partitions | ECMWF WAM and GFS-Wave, deterministic | 1 each |
| Surface current | Météo-France SMOC | 1 |
| Synoptic fields | ECMWF IFS and AIFS, GFS, ICON, UKMO, GEM | 1 each |

All of it comes from Open-Meteo, which decodes the centers' GRIB. The free tier
is enough: a screening of four days is a few hundred kilobytes.

Open-Meteo's values were checked against the raw GRIB from ECMWF Open Data and
NOAA NOMADS on 2026-09-15. For 10 m wind at +96 h, ECMWF ENS, GEFS and ECMWF
HRES all agreed within 0.1 kt and 1°. GFS speeds differ by up to 2 kt because
Open-Meteo serves the 0.125° grid and NOMADS the 0.25°.

Two differences in the source data:

- **Gusts aren't the same measure in the two ensembles.** ECMWF reports the
  highest gust in the previous 3 hours, GEFS the gust at that instant, so the
  model fields aren't comparable. Both are replaced with 1.23 times the sustained
  wind, which is the WMO at-sea ratio of a 3 second gust to the 10 minute mean.
- **Wave direction conventions differ** between the wave models, and the period
  fields aren't the same quantity either. ECMWF's mean period reads 0.3 to 2.5 s
  shorter than its peak period at the points checked, so a display showing peak
  period will make the same sea look less steep.

## The boat

Drop your polar files into the app folder and tick them on the settings page.
`.pol` and `.csv` in the usual qtVlm and OpenCPN layouts both parse. Included
here are the polars for a Lagoon 42, which you should replace with your own.

The **conservative polar** is the lowest speed in each cell across the files you
tick, times a cruise factor (0.90 by default). Polars collected at different
times and in different conditions disagree, and the lowest of them is what this plans against. A **performance polar** is optional: if you point the
settings at one it runs alongside as a faster bound, with separate upwind,
downwind and night factors.

Speeds are turned into a VMC table, the best speed toward the next waypoint at
each wind angle, taken from the convex hull of the polar in velocity space. That
covers the case where tacking or gybing beats sailing the straight line, which is
the only tactic modeled. Everything else is rhumb line, with the engine on when
sailing speed drops under a threshold you set.

## Limits and the verdict

Each departure gets the worst of three checks. A tier counts when more than 10%
of the ensemble passages reach it.

| Check | WARNING | NO |
|---|---|---|
| Sustained wind | over 25 kt | over 30 kt |
| Gust | over 35 kt | over 40 kt |
| Waves | over 3.0 m forward of the beam, or the period rule broken | over 4.0 m from any direction |

The period rule is that the period in seconds should be at least the wave height
in feet, which is 3.28 times the height in meters. Height alone doesn't decide
anything: 3 m from astern with a long period is a different passage from 3 m on
the bow, so the wave checks read height, period and which quarter the sea comes
from together. Every number above is editable on the settings page.

The screening card names the check that set the verdict, and the brief's method
section prints the thresholds that produced it.

## Comfort

Comfort is rated every 15 minutes of every simulated passage, then reported as
the level that holds for most of the day and the worst level lasting 3 hours or
more.

Two measures, and the worse one wins. **Wind over the deck** uses apparent wind
against the boat's reef points. **Motion** uses the wave encounter period,
`Te = T / |1 - V cos(theta) / c|`, and from it the RMS vertical acceleration,
`(Hs / 4) * (2 pi / Te)^2`, banded on ISO 2631-1. Multihull factors go on top:
beam-on seas and head seas both make the ride worse than the bare number says.

The result reads as Champagne, Easy, Coffee or Uncomfortable, with the reason
attached: the seas' quarter, the strongest apparent wind, the roughest motion and
what caused the worst spell.

## The sea

The wave section separates wind sea, primary swell and secondary swell where the
model provides partitions, gives the steepness of each as "1 in N", and flags a
cross sea when two trains arrive from different directions with the swell period
under 11 s. A short swell crossing a wind sea is the setup that makes a boat roll
badly at a wave height that looks harmless in a single number.

## Weather story and along-track panels

Six deterministic models are read through the layers a forecaster works with:
the surface pressure pattern with highs and lows located, geostrophic wind, 850
hPa temperature advection, 700 hPa humidity and the trade inversion, K-index,
CAPE, precipitable water, 500 hPa vorticity and the 250 hPa jet. The same
diagnostics are read at the boat's median position every 6 hours and drawn as
small multiples, so you can see the trend the passage sails into rather than one
snapshot.

Route point values are fetched on the 0.25° grid rather than read off the
coarse synoptic grid, because a 2.5° grid point can sit on an island and report
land wind for an offshore leg.

## What it doesn't do

- Route optimization or weather routing. It sails the rhumb line.
- Squalls and convection as events. CAPE and K-index are reported as ingredients.
- Wave-induced speed loss.
- Tides, currents in passes, or swell at the anchorage.
- Anything that should be trusted without a look at the official forecast, which
  the brief includes for that reason.

## Using it from other software

The modules are plain Python with no framework and no global state beyond
`settings.json`, so the engine can be driven directly:

    import trip as T, screen, report_html

`polar.py` parses polars and builds VMC tables. `windows.py` sails the ensembles
and scores a departure. `comfort.py` and `seastate.py` are self-contained
calculators. `screen.py` and `report_html.py` render the two pages. `daily.py`
returns the day by day numbers as data, which is the place to start if you want
to feed them somewhere else, a plugin or an instrument display included.

The optional Home Assistant publisher in `ha_publish.py` is one example of that:
off by default, and it copies the pages to an HA box and sends an alert when a
verdict changes. Fill in the host and notify service on the settings page if you
want it.

## Tests

    python3 test_windows.py

Runs without network access. The suite covers the polar envelope, the VMC hull,
the wind and wave tiers, comfort banding, sea state rules, position parsing,
timers, and a smoke test that builds a complete brief from saved trip data.

## License

MIT. See LICENSE.
