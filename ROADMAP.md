# Roadmap

Ordered by how much each one changes the answers the tool gives.

## 1. Sum the wave trains for the motion estimate

The comfort model treats the sea as one height and one period. A passage through a long
swell with a local wind sea on top is two trains, and the short one drives the
accelerations, because acceleration goes as 1 over the period squared. Measured against a
real forecast, collapsing the two understates RMS vertical acceleration by a median of
2.8 times, 5.2 at the 90th percentile.

The data is already downloaded: GFS-Wave gives the wind sea, primary swell and secondary
swell separately. Summing variances across the trains is the fix:

    a_rms^2 = sum over trains of ((Hs_i / 4) * (2 pi / Te_i)^2 * response_i)^2

Do it together with item 2, since on its own it moves every passage a band rougher.

## 2. Re-anchor the comfort bands

`comfort.MOTION_BANDS` compares an unweighted acceleration against the comfort reactions
in ISO 2631-1, which assume Wk weighting over 0.5 to 80 Hz. Wave motion sits at 0.06 to
0.25 Hz, where the standard uses the Wf weighting and a motion sickness dose value
instead. For scale, NORDFORSK 1987's limit for passengers is roughly ten times looser
than the bands here.

The bands work as an internal index because they were tuned against the inputs as they
are. Change the inputs in item 1 and they have to move with them. Re-anchor against
passages actually sailed, not against the standard.

## 3. Alert when a run refuses to build on stale data

A run that refuses writes `FAILED ... too old for a report` to `trips/auto.log` and stops.
Nothing else says so, and a silent update looks the same as a quiet forecast. Send one
alert per refusal, at most one a day.

## 4. Check what ECMWF publishes as its peak period

The feet rule now reads the dominant swell's period. GFS-Wave's field was verified against
raw GRIB (`perpw`), but ECMWF's `wave_peak_period` came back at 9 to 11 s on a sea where
GFS had a 13 to 14 s primary swell, with ECMWF also about a meter higher on significant
height. Either the models genuinely disagree about which train dominates, which is
possible in a bimodal sea, or the field isn't what it looks like. Verify against an ECMWF
GRIB that carries `pp1d`.

## 5. Packaging for other machines

A portable Windows bundle, PDF output on a machine with no Chrome or Edge to print with,
and a first run that walks someone through pointing at their own polar files.

## 6. Calibrate against logged passages

Everything about comfort is a first guess until it's checked against recorded motion. Log
acceleration and the crew's own rating on passage, then fit the bands to it.

## Recently finished

- The feet rule judged on the dominant swell's peak period, in bands: 3:1 or better is
  clear, under 2:1 warns, under 1:1 is a square sea. Between 2:1 and 3:1 it warns only in
  big seas, and nothing flags until the sea stays steep for three hours.
- Steepness as the physical cross-check, wavelength over height as 1 in N, warning at
  1 in 40 and stopping at 1 in 20.
- Every model cycle recorded, its age printed on every page, and a refusal to build a
  report on data past the freshness limits.
