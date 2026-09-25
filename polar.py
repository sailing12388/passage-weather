"""Build a conservative polar from the boat's polar files in this folder.

Method:
  1. Read every source onto one TWA x TWS grid. Blanks and 0.01 placeholders
     are treated as missing, not as zero speed.
  2. Take the cell-wise minimum across the sources that have a value.
  3. Multiply by CRUISE_FACTOR for a loaded cruising boat, fouling, sea state.
  4. Build a velocity-made-good-on-course (VMC) table from that polar, so a
     course the boat can't point at directly is sailed as tacks or gybes.

Sources used for the envelope: sabado.pol, sabado.csv, sabado_polar_v2.csv,
sabado(1).csv. The OpenCPN NMEA polar (logged, sparse) is a cross-check only.
"""
import csv
import io
import math
import zipfile
from pathlib import Path

import numpy as np

import settings

HERE = Path(__file__).parent
CRUISE_FACTOR = settings.get()["polar"]["cruise_factor"]
TWS = [6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 36]
TWA = list(range(60, 181, 10))
SOURCES = settings.get()["polar"]["sources"]
# Optional extra polar exported from OpenCPN, used by __main__ as a sanity check if present.
NMEA_ZIP = HERE / "OpenCPN Polar.zip"
NMEA_MEMBER = "Sabado_NMEA_V2.pol"
_pf = settings.get()["polar"]["performance_file"]
PW_FILE = (HERE / _pf) if _pf else None
# Performance polar adjustments from settings. Split: upwind below 90 TWA, downwind from 90.
# Night replaces the day factor.
PW_UPWIND = settings.get()["polar"]["performance"]["upwind"]
PW_DOWNWIND = settings.get()["polar"]["performance"]["downwind"]
PW_NIGHT = settings.get()["polar"]["performance"]["night"]


def parse_polar(text):
    """Return {(twa, tws): speed} from a tab, semicolon or comma separated polar."""
    rows = [r for r in text.splitlines() if r.strip()]
    delim = "\t" if "\t" in rows[0] else (";" if ";" in rows[0] else ",")
    table = list(csv.reader(io.StringIO("\n".join(rows)), delimiter=delim))
    header = [float(x) for x in table[0][1:]]
    out = {}
    for row in table[1:]:
        twa = float(row[0])
        for tws, cell in zip(header, row[1:]):
            cell = cell.strip()
            if not cell:
                continue
            v = float(cell)
            if v <= 0.05:  # OpenCPN placeholder for "no data"
                continue
            out[(twa, tws)] = v
    return out


def load_sources():
    return {name: parse_polar((HERE / name).read_text()) for name in SOURCES}


def load_nmea():
    if not NMEA_ZIP.exists():
        return None
    with zipfile.ZipFile(NMEA_ZIP) as z:
        return parse_polar(z.read(NMEA_MEMBER).decode())


def envelope(sources):
    """Cell-wise minimum. Returns (grid, which-source-set-the-minimum)."""
    grid = np.full((len(TWA), len(TWS)), np.nan)
    who = {}
    for i, a in enumerate(TWA):
        for j, s in enumerate(TWS):
            vals = {n: p[(a, s)] for n, p in sources.items() if (a, s) in p}
            if vals:
                lo = min(vals.values())
                grid[i, j] = lo
                who[(a, s)] = sorted(n for n, v in vals.items() if v == lo)
    return grid, who


def conservative_polar():
    grid, _ = envelope(load_sources())
    return grid * CRUISE_FACTOR


def pw_raw_grid():
    """A predefined performance polar on the TWA x TWS grid used here.
    File rows: TWS, then TWA/speed pairs from 0 to 170. 180 takes the 170 value."""
    rows = {}
    for line in PW_FILE.read_text().splitlines():
        if not line.strip():
            continue
        v = [float(x) for x in line.split("\t")]
        rows[v[0]] = dict(zip(v[1::2], v[2::2]))
    tws_rows = sorted(rows)
    grid = np.zeros((len(TWA), len(TWS)))
    for i, a in enumerate(TWA):
        a_src = min(a, 170)
        speeds = [np.interp(a_src, sorted(rows[t]), [rows[t][k] for k in sorted(rows[t])]) for t in tws_rows]
        grid[i] = np.interp(TWS, tws_rows, speeds)
    return grid


def pw_polars():
    """(day grid, night grid) with the performance-polar adjustments from Settings applied,
    or (None, None) when no performance polar is configured. It is optional: the conservative
    envelope is the one the estimates rely on."""
    if PW_FILE is None or not PW_FILE.exists():
        return None, None
    raw = pw_raw_grid()
    day = raw * np.array([[PW_UPWIND if a < 90 else PW_DOWNWIND] for a in TWA])
    return day, raw * PW_NIGHT


def boat_speed(grid, twa, tws):
    """Bilinear lookup. Below 60 TWA returns 0 (not a sailing angle).
    Above 36 kt TWS holds the 36 kt row; below 6 kt scales linearly to 0."""
    twa = abs(twa)
    if twa < TWA[0]:
        return 0.0
    ai = np.interp(twa, TWA, range(len(TWA)))
    if tws < TWS[0]:
        scale, tws = tws / TWS[0], TWS[0]
    else:
        scale = 1.0
    si = np.interp(min(tws, TWS[-1]), TWS, range(len(TWS)))
    a0, s0 = int(math.floor(ai)), int(math.floor(si))
    a1, s1 = min(a0 + 1, len(TWA) - 1), min(s0 + 1, len(TWS) - 1)
    fa, fs = ai - a0, si - s0
    v = (grid[a0, s0] * (1 - fa) * (1 - fs) + grid[a1, s0] * fa * (1 - fs)
         + grid[a0, s1] * (1 - fa) * fs + grid[a1, s1] * fa * fs)
    return float(v) * scale


def vmc_table(grid, tws_values=range(0, 41), twa_step=1):
    """VMC[tws][course_twa]: best speed along a course at course_twa off the
    true wind, allowing a mix of two headings (tacking or gybing).
    Computed as the convex hull of the polar in boat-velocity space."""
    angles = np.arange(60, 181, 2)
    table = np.zeros((len(tws_values), 181 // twa_step + 1))
    for k, tws in enumerate(tws_values):
        pts = []
        for a in angles:
            v = boat_speed(grid, a, tws)
            r = math.radians(a)
            pts.append((v * math.sin(r), v * math.cos(r)))   # x: across wind, y: toward wind
            pts.append((-v * math.sin(r), v * math.cos(r)))
        pts = np.array(pts)
        for c_idx, c in enumerate(range(0, 181, twa_step)):
            u = np.array([math.sin(math.radians(c)), math.cos(math.radians(c))])
            best = 0.0
            # single heading
            proj_perp = pts[:, 0] * u[1] - pts[:, 1] * u[0]
            proj_par = pts @ u
            on_line = np.abs(proj_perp) < 1e-9
            if on_line.any():
                best = max(best, proj_par[on_line].max())
            # two headings on opposite sides of the course line
            pos, neg = proj_perp > 1e-9, proj_perp < -1e-9
            if pos.any() and neg.any():
                P, N = pts[pos], pts[neg]
                pp, pn = proj_perp[pos][:, None], proj_perp[neg][None, :]
                w = pp / (pp - pn)                     # weight on the N point
                par = proj_par[pos][:, None] * (1 - w) + proj_par[neg][None, :] * w
                best = max(best, float(par.max()))
            table[k, c_idx] = max(best, 0.0)
    return table


def write_outputs():
    sources = load_sources()
    grid, who = envelope(sources)
    cons = grid * CRUISE_FACTOR
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    with open(out / "sabado_conservative.pol", "w") as f:
        f.write("twa/tws\t" + "\t".join(str(s) for s in TWS) + "\n")
        for i, a in enumerate(TWA):
            f.write(f"{a}\t" + "\t".join(f"{v:.1f}" for v in cons[i]) + "\n")

    counts = {}
    for names in who.values():
        for n in names:
            counts[n] = counts.get(n, 0) + 1
    print("cells where each source is the minimum:", counts)
    print(f"cells with a value: {np.isfinite(grid).sum()} of {grid.size}")

    nmea = load_nmea()
    if nmea:
        ratios = []
        for (a, s), v in nmea.items():
            if a in TWA and s in TWS and np.isfinite(grid[TWA.index(a), TWS.index(s)]):
                ratios.append(v / cons[TWA.index(a), TWS.index(s)])
        r = np.array(ratios)
        print(f"NMEA logged / conservative, {len(r)} matching cells: "
              f"min {r.min():.2f}  median {np.median(r):.2f}  max {r.max():.2f}  "
              f"cells where logged is below conservative: {(r < 1).sum()}")
    return cons


if __name__ == "__main__":
    cons = write_outputs()
    print((HERE / "output" / "sabado_conservative.pol").read_text())
