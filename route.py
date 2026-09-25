"""Planning routes from Nadi. Not for navigation.

Both start at a planning point offshore of the barrier reef southwest of Nadi.
The passage out from Denarau isn't simulated; EXIT_HOURS is added to every ETA.

port_resolution: 8 nm north of Futuna (19.533S 170.217E), then about 15 nm
  south of Aniwa (19.240S 169.604E) to off Port Resolution (Ireupuow 19.533S 169.500E).
port_vila: about 50 nm north of Erromango's center, then south of Efate to a
  point south of Pango Point (Pango 17.783S 168.283E). Harbor entry not simulated.
"""
import math

ROUTES = {
    "port_resolution": [
        ("Offshore Nadi", -18.10, 177.00),
        ("N of Futuna", -19.40, 170.25),
        ("Port Resolution", -19.51, 169.52),
    ],
    "port_vila": [
        ("Offshore Nadi", -18.10, 177.00),
        ("S of Efate", -17.95, 168.55),
        ("S of Pango Point", -17.85, 168.30),
    ],
}
NAME = "port_resolution"
WAYPOINTS = ROUTES[NAME]


def define(name, waypoints):
    """Register a route: a list of (label, lat, lon)."""
    ROUTES[name] = list(waypoints)


def use(name):
    global NAME, WAYPOINTS
    NAME, WAYPOINTS = name, ROUTES[name]


EXIT_HOURS = 3.0          # Denarau to WP0
SAMPLE_NM = 30.0          # forecast sample spacing along the route
R_NM = 3440.065


def gc_distance(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_NM * math.asin(math.sqrt(a))


def gc_bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def gc_interp(lat1, lon1, lat2, lon2, f):
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    d = gc_distance(lat1, lon1, lat2, lon2) / R_NM
    if d == 0:
        return lat1, lon1
    a, b = math.sin((1 - f) * d) / math.sin(d), math.sin(f * d) / math.sin(d)
    x = a * math.cos(p1) * math.cos(l1) + b * math.cos(p2) * math.cos(l2)
    y = a * math.cos(p1) * math.sin(l1) + b * math.cos(p2) * math.sin(l2)
    z = a * math.sin(p1) + b * math.sin(p2)
    return math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))


def legs():
    out, start = [], 0.0
    for (n1, a1, o1), (n2, a2, o2) in zip(WAYPOINTS, WAYPOINTS[1:]):
        d = gc_distance(a1, o1, a2, o2)
        out.append(dict(frm=n1, to=n2, start_nm=start, length_nm=d,
                        lat1=a1, lon1=o1, lat2=a2, lon2=o2))
        start += d
    return out


def total_nm():
    return sum(l["length_nm"] for l in legs())


def position_at(nm):
    """(lat, lon, course_true) at a distance along the route."""
    for leg in legs():
        if nm <= leg["start_nm"] + leg["length_nm"] or leg is legs()[-1]:
            f = min(max((nm - leg["start_nm"]) / leg["length_nm"], 0.0), 1.0)
            lat, lon = gc_interp(leg["lat1"], leg["lon1"], leg["lat2"], leg["lon2"], f)
            to = (leg["lat2"], leg["lon2"])
            if f >= 0.999:
                crs = gc_bearing(leg["lat1"], leg["lon1"], *to)
            else:
                crs = gc_bearing(lat, lon, *to)
            return lat, lon, crs


def sample_points():
    n = int(math.ceil(total_nm() / SAMPLE_NM))
    step = total_nm() / n
    return [(i * step, *position_at(i * step)[:2]) for i in range(n + 1)]


if __name__ == "__main__":
    import sys
    use(sys.argv[1] if len(sys.argv) > 1 else NAME)
    for leg in legs():
        print(f"{leg['frm']} -> {leg['to']}: {leg['length_nm']:.0f} nm, "
              f"initial course {gc_bearing(leg['lat1'], leg['lon1'], leg['lat2'], leg['lon2']):.0f}T")
    print(f"total {total_nm():.0f} nm, {len(sample_points())} forecast sample points")
