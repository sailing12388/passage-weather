"""Comfort from wind over the deck and boat motion in the sea, allowing for point of sail.

Two parts, and the worse sets the level on Sereno's scale (Sick, Rough, Coffee, Easy, Champagne):

1. Motion. Waves met from astern arrive slowly, waves met on the bow arrive fast. The encounter
   period for a wave of period T (deep water, phase speed c = gT/2pi) with the boat making V
   through the water at angle theta between its heading and the direction the waves travel is
       Te = T / |1 - V cos(theta) / c|
   For waves long compared with the hull the boat rides the surface, so vertical acceleration
   follows the sea: RMS elevation is Hs/4, giving RMS acceleration (Hs/4) (2 pi / Te)^2.
   Shorter waves move the boat less (scaled by wavelength against three hull lengths).
   Catamaran factors: beam seas add quick roll, bow seas add bridgedeck slamming.
   Levels use the comfort bands of ISO 2631-1 for RMS acceleration (under 0.315 m/s2 not
   uncomfortable, 0.315-0.63 a little, 0.5-1 fairly, 0.8-1.6 uncomfortable, 1.25-2.5 very).
   ISO 2631-1 was written for vibration above the band where seasickness sits, so these are a
   starting point until Sereno's comfort presses can calibrate them.

2. Wind over the deck, by apparent wind, from the Lagoon 42 manual's reef points: first reef at
   23 kt apparent on a reach (20 close hauled), second reef around 30-33 kt.

Not calibrated against a logged passage.
"""
import math

G = 9.81
KT = 0.514444               # m/s per knot
LOA_M = 12.8                # Lagoon 42
LEVELS = ["Sick", "Rough", "Coffee", "Easy", "Champagne"]    # index 0 is worst

# RMS vertical acceleration, m/s2, at or above which each level is lost (Champagne, Easy, Coffee, Rough)
MOTION_BANDS = (0.15, 0.315, 0.63, 1.25)
# apparent wind, kt, above which each level is lost (Champagne, Easy, Coffee, Rough)
WIND_BANDS = (12, 18, 23, 33)
BEAM_ROLL = 1.25            # beam seas: quick catamaran roll
BOW_SLAM = 1.3              # seas within 45 deg of the bow: bridgedeck slamming, once the sea is over 1.5 m


def apparent_wind(tws, sailing_twa, boat_kt):
    return math.sqrt(tws * tws + boat_kt * boat_kt + 2 * tws * boat_kt * math.cos(math.radians(sailing_twa)))


def relative_sea(wave_from, heading):
    """0 = seas on the bow, 180 = seas from astern."""
    return abs((wave_from - heading + 180) % 360 - 180)


def encounter_period(period_s, boat_kt, rel_sea_deg):
    """Te for a boat meeting waves at rel_sea_deg (0 on the bow, 180 from astern)."""
    c = G * period_s / (2 * math.pi)
    theta = math.radians(180 - rel_sea_deg)          # angle between heading and wave travel
    factor = abs(1 - boat_kt * KT * math.cos(theta) / c)
    return period_s / max(factor, 0.15)                # surfing along with a wave: cap the stretch


def vertical_accel(hs_m, period_s, boat_kt, rel_sea_deg):
    """RMS vertical acceleration, m/s2, with catamaran roll and slam factors."""
    te = encounter_period(period_s, boat_kt, rel_sea_deg)
    wavelength = G * period_s ** 2 / (2 * math.pi)
    response = min(1.0, wavelength / (3 * LOA_M))
    a = (hs_m / 4) * (2 * math.pi / te) ** 2 * response
    if 60 <= rel_sea_deg <= 120:
        a *= BEAM_ROLL
    elif rel_sea_deg < 45 and hs_m > 1.5:
        a *= BOW_SLAM
    return a


def _band_level(value, bands):
    for i, limit in enumerate(bands):
        if value < limit:
            return 4 - i
    return 0


def level(aws_kt, accel):
    """Index into LEVELS: the worse of wind over the deck and motion."""
    return min(_band_level(aws_kt, WIND_BANDS), _band_level(accel, MOTION_BANDS))


SEA_SECTORS = ("bow", "fwd_quarter", "aft_quarter", "astern")
SEA_SECTOR_NAMES = {"bow": "on the bow", "fwd_quarter": "on the forward quarter",
                    "aft_quarter": "on the aft quarter", "astern": "from astern"}


def sea_side(rel_sea_deg):
    """Sector the seas come from: bow 0-45, forward quarter 45-90, aft quarter 90-135, astern 135-180."""
    return SEA_SECTORS[min(int(rel_sea_deg // 45), 3)]


def iso_words(accel):
    """ISO 2631-1 comfort wording for an RMS acceleration."""
    for limit, words in ((0.315, "not uncomfortable"), (0.63, "a little uncomfortable"), (1.0, "fairly uncomfortable"),
                         (1.6, "uncomfortable")):
        if accel < limit:
            return words
    return "very uncomfortable"


def summarize_passage(members):
    """Comfort across the ensemble passages from daily.run: (main, worst, details dict, sentence)."""
    import numpy as np
    shares, worst, causes, aws, acc = [], [], [], [], []
    sides = {k: [] for k in SEA_SECTORS}
    for _, _, days, _ in members:
        days = [d for d in days if d]
        hours = np.sum([d["hours_at"] for d in days], axis=0)
        shares.append(hours / hours.sum() if hours.sum() else hours)
        w = min(days, key=lambda d: d["level_worst3"])
        worst.append(w["level_worst3"])
        causes.append(w["worst_cause"])
        for k in sides:
            sides[k].append(np.mean([d[f"sea_{k}"] for d in days]))
        aws.append(max(d["aws_max"] for d in days))
        acc.append(max(d["accel_max"] for d in days))
    share = np.mean(shares, axis=0)
    main = int(np.argmax(share))
    worst_level = int(np.floor(np.median(worst)))
    cause = max(set(causes), key=causes.count)
    side = {k: float(np.mean(v)) for k, v in sides.items()}
    aws_med, acc_med = float(np.median(aws)), float(np.median(acc))
    order = sorted(range(len(LEVELS)), key=lambda i: -share[i])
    time_txt = ", ".join(f"{share[i]:.0%} {LEVELS[i]}" for i in order if share[i] >= 0.05)
    main_side = max(side, key=side.get)
    sentence = (f"Time at each level: {time_txt}. Worst 3 hours: {LEVELS[worst_level]}, from {cause}. "
                "Seas " + ", ".join(f"{side[k]:.0%} {SEA_SECTOR_NAMES[k]}" for k in SEA_SECTORS if side[k] >= 0.05) + ". "
                f"Peak apparent wind {aws_med:.0f} kt. Roughest motion {acc_med:.2f} m/s², "
                f"{iso_words(acc_med)} by ISO 2631.")
    return LEVELS[main], LEVELS[worst_level], dict(share=share.tolist(), cause=cause, side=side, main_side=main_side,
                                                   aws=aws_med, accel=acc_med), sentence
