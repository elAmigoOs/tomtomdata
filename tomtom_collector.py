"""
Masar - TomTom Traffic Flow Collector (Jeddah)
===============================================
Builds the 6-month per-road dataset the supervisor asked for, by polling
TomTom's Flow Segment Data endpoint for sample points along 3 Jeddah roads
and appending every reading to a CSV.

There is no ready-made dataset to download here - this script IS the dataset.
Every day it doesn't run is a day of data you can never recover.

--------------------------------------------------------------------------
STEP 1 - GET AN API KEY (5 minutes, no credit card)
--------------------------------------------------------------------------
  1. Go to https://developer.tomtom.com/ -> Register (free "Freemium" plan)
  2. Dashboard -> My Keys -> your default key works for Traffic APIs
  3. Paste it below, or better: set it as an environment variable
         export TOMTOM_API_KEY="your_key_here"

  Free tier limits are PER MONTH (not per day), no card needed:
      Traffic Flow (Flow Segment Data): 20,000 requests / month
      Incident Details:                  2,500 requests / month
  Over the limit requests are rejected (HTTP 403 / 429), NOT billed.
  Check the budget with: python tomtom_collector.py --budget

--------------------------------------------------------------------------
STEP 2 - VERIFY YOUR POINTS LAND ON THE RIGHT ROADS  (do this first!)
--------------------------------------------------------------------------
      python tomtom_collector.py --verify

  The coordinates below are APPROXIMATE SEEDS I could not verify myself.
  --verify snaps each one to its nearest road and prints:
      - the road class (FRC0=motorway ... FRC6=local road)
      - the free-flow speed (a 100+ km/h free-flow = highway, ~60 = city street)
      - a Google Maps link to the exact segment TomTom matched

  Open each link and confirm it's really on Al Haramain / Al Madinah /
  King Abdulaziz. If not, right-click the right spot in Google Maps, copy
  the coordinates, and replace the point below. Sanity check on class:
      Al Haramain Rd   -> expect FRC0 or FRC1 (it's a highway)
      Al Madinah Rd    -> expect FRC1 or FRC2 (major arterial)
      King Abdulaziz   -> expect FRC1 or FRC2 (major arterial)

  Getting this wrong means 6 months of data for the wrong roads.

--------------------------------------------------------------------------
STEP 3 - RUN A 24-48 HOUR PILOT (as the supervisor said - do not skip)
--------------------------------------------------------------------------
      python tomtom_collector.py --once          # single test reading
      python tomtom_collector.py                 # continuous loop

  After ~24h, open the CSV and check you can actually see rush hour in the
  numbers (speed_ratio should dip in the morning/evening). If the numbers
  never move, your points are on a road with no real traffic variation and
  you should pick better ones BEFORE committing to 6 months.

--------------------------------------------------------------------------
STEP 4 - RUN IT FOR 6 MONTHS
--------------------------------------------------------------------------
  Do NOT run this on your laptop - it has to survive reboots and closed
  lids. Use the GitHub Actions workflow (tomtom_collect.yml) that runs it
  on GitHub's servers every 30 minutes for free.

  --once exits with a non-zero code when no row was written (e.g. the key
  was rejected or the monthly quota is used up), so the GitHub run turns red.

Requires: pip install requests
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("TOMTOM_API_KEY", "PASTE_YOUR_KEY_HERE")

BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData"
STYLE = "absolute"   # absolute = real km/h speeds (what we want for ML)
ZOOM = 12            # higher zoom = finer road matching; 10-14 is sensible here
UNIT = "kmph"

CSV_PATH = "jeddah_tomtom_flow.csv"
POLL_INTERVAL_MINUTES = 30   # see budget_report(): must fit the MONTHLY free limits

# TomTom free-tier limits, per calendar month.
FLOW_MONTHLY_LIMIT = 20_000        # Traffic Flow requests (one per point per run)
INCIDENTS_MONTHLY_LIMIT = 2_500    # Incident Details requests (one per run, tomtom_incidents.py)
INCIDENT_REQUESTS_PER_RUN = 1
DAYS_PER_MONTH = 31                # budget for the longest month

SAUDI_TZ = timezone(timedelta(hours=3))

# Points picked by hand on Google Maps (Osama). Re-run --verify after any change.
# Several points per road so one reading isn't a single spot's quirk.
ROADS = {
    "Al Haramain Road": [
        (21.72336, 39.19394),
        (21.62897, 39.21308),
        (21.58853, 39.22808),
    ],
    "Al Madinah Road": [
        (21.67808, 39.12614),
        (21.64356, 39.14481),
        (21.61511, 39.15536),
        (21.55194, 39.17472),
    ],
    "King Abdulaziz Road": [
        (21.55642, 39.12564),
        (21.59236, 39.12375),
        (21.62064, 39.11597),
    ],
}

# Congestion label from speed_ratio = currentSpeed / freeFlowSpeed.
# Starting cutoffs - revisit after the pilot using the real distribution.
FREE_FLOW_MIN_RATIO = 0.85   # >= 85% of free-flow speed -> Low
MODERATE_MIN_RATIO = 0.60    # 60-85% -> Medium ; below 60% -> High


# ---------------------------------------------------------------------------
# CORE
# ---------------------------------------------------------------------------

def classify(speed_ratio):
    if speed_ratio is None:
        return ""
    if speed_ratio >= FREE_FLOW_MIN_RATIO:
        return "Low"
    if speed_ratio >= MODERATE_MIN_RATIO:
        return "Medium"
    return "High"


class FetchError(RuntimeError):
    """A failed request. status is the HTTP status code (None for network errors).
    The message never contains the API key."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def fetch_point(lat, lon, max_retries=3):
    """One Flow Segment Data call. Returns the parsed dict, or raises FetchError."""
    url = f"{BASE_URL}/{STYLE}/{ZOOM}/json"
    params = {"key": API_KEY, "point": f"{lat},{lon}", "unit": UNIT}

    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=20)

            if r.status_code == 429:
                # Quota used up, or too fast. Back off, don't hammer.
                wait = 60 * (attempt + 1)
                print(f"    HTTP 429 rate-limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue

            if r.status_code == 403:
                raise FetchError("HTTP 403 Forbidden - the key was rejected: invalid, Traffic API "
                                 "not enabled, or the monthly free quota is used up", 403)

            if r.status_code == 400:
                raise FetchError("HTTP 400 - TomTom found no road near this point. "
                                 "Move the point onto the road itself.", 400)

            if r.status_code != 200:
                raise FetchError(f"HTTP {r.status_code} from TomTom", r.status_code)

            return r.json()["flowSegmentData"]

        except requests.RequestException as e:
            # Never print the raw error: it contains the URL, and the URL
            # contains the API key.
            msg = str(e).replace(API_KEY, "***")
            if attempt == max_retries - 1:
                raise FetchError(f"network error: {msg}") from None
            time.sleep(5 * (attempt + 1))

    raise FetchError(f"HTTP 429 - still rate-limited after {max_retries} tries "
                     "(monthly quota used up, or too many requests per second)", 429)


def segment_mid(d, lat, lon):
    """Midpoint of the road segment TomTom matched (falls back to the input)."""
    coords = d.get("coordinates", {}).get("coordinate", [])
    if not coords:
        return lat, lon
    m = coords[len(coords) // 2]
    return m["latitude"], m["longitude"]


# These three are major roads, so a local-road class (FRC4+) means the point
# snapped to a side street. FRC0-FRC3 are all plausible - confirm on the map.
LOCAL_CLASSES = {"FRC4", "FRC5", "FRC6"}

# Two matched segments whose midpoints are closer than this are treated as
# the same stretch of road (e.g. both carriageways of one long highway segment).
NEAR_DUP_KM = 0.5


def haversine_km(a, b):
    from math import radians, sin, cos, asin, sqrt
    la1, lo1, la2, lo2 = map(radians, (a[0], a[1], b[0], b[1]))
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * asin(sqrt(h))


def segment_length_km(d):
    c = d.get("coordinates", {}).get("coordinate", [])
    return sum(haversine_km((p["latitude"], p["longitude"]),
                            (q["latitude"], q["longitude"]))
               for p, q in zip(c, c[1:]))


def verify():
    """Snap each configured point to its road and report what it matched.
    Flags duplicate segments, unexpected road classes and points with no road."""
    print("Verifying each point snaps to the road you intend.\n")

    seen = {}       # matched-segment midpoint -> first point that landed there
    problems = []

    for road, points in ROADS.items():
        print(f"{'='*70}\n{road}\n{'='*70}")
        for i, (lat, lon) in enumerate(points, 1):
            label = f"{road} p{i}"
            try:
                d = fetch_point(lat, lon)
                mlat, mlon = segment_mid(d, lat, lon)
                link = f"https://www.google.com/maps?q={mlat},{mlon}"
                frc = d.get("frc")
                print(f"  point {i}: {lat},{lon}")
                print(f"    road class  : {frc}")
                print(f"    free-flow   : {d.get('freeFlowSpeed')} km/h")
                print(f"    current     : {d.get('currentSpeed')} km/h")
                print(f"    confidence  : {d.get('confidence')}")
                print(f"    segment len : {segment_length_km(d):.1f} km "
                      "(the whole stretch this one reading covers)")
                print(f"    CHECK HERE  : {link}  (middle of that segment)")

                dup = next((lbl for (pl, pn), lbl in seen.items()
                            if haversine_km((mlat, mlon), (pl, pn)) < NEAR_DUP_KM), None)
                if dup:
                    print(f"    !! DUPLICATE: same stretch of road as {dup} "
                          "- move this point to a different part of the road")
                    problems.append(f"{label} duplicates {dup}")
                else:
                    seen[(mlat, mlon)] = label

                if frc in LOCAL_CLASSES:
                    print(f"    !! CLASS {frc} is a local road - probably a side "
                          "street, check the map link")
                    problems.append(f"{label} local road class {frc}")
            except Exception as e:
                print(f"  point {i}: {lat},{lon}  -> FAILED: {e}")
                problems.append(f"{label} failed")
            print()
            time.sleep(0.3)   # stay under the default QPS limit

    print("=" * 70)
    if problems:
        print(f"{len(problems)} problem(s) - fix these points before collecting:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("All points are on distinct stretches of road and none is a local street.")
        print("Now open every CHECK HERE link once to confirm it's the right road.")


FIELDNAMES = [
    "timestamp_utc", "timestamp_local", "date_local", "hour_local",
    "day_of_week", "is_weekend", "road", "point_id", "lat", "lon",
    "seg_lat", "seg_lon",
    "frc", "current_speed_kmh", "free_flow_speed_kmh",
    "current_travel_time_s", "free_flow_travel_time_s",
    "speed_ratio", "delay_s", "confidence", "road_closure", "congestion_level",
]


def collect_once():
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(SAUDI_TZ)
    new_file = not os.path.exists(CSV_PATH)

    if not new_file:
        with open(CSV_PATH, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if header != FIELDNAMES:
            sys.exit(f"{CSV_PATH} has a different column layout than this script. "
                     "Delete it (or rename it) and run again, otherwise rows will misalign.")

    rows, ok, failed = [], 0, 0
    statuses = {}          # HTTP status (or "network") -> number of failed points
    key_rejected = False   # after a 403/429 every other point would fail the same way

    for road, points in ROADS.items():
        for i, (lat, lon) in enumerate(points, 1):
            if key_rejected:
                failed += 1
                statuses["skipped"] = statuses.get("skipped", 0) + 1
                continue
            try:
                d = fetch_point(lat, lon)

                cur = d.get("currentSpeed")
                free = d.get("freeFlowSpeed")
                ratio = round(cur / free, 4) if (cur and free) else None

                cur_tt = d.get("currentTravelTime")
                free_tt = d.get("freeFlowTravelTime")
                delay = (cur_tt - free_tt) if (cur_tt is not None and free_tt is not None) else None
                seg_lat, seg_lon = segment_mid(d, lat, lon)

                rows.append({
                    "timestamp_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_local": now_local.strftime("%Y-%m-%d %H:%M:%S"),
                    "date_local": now_local.strftime("%Y-%m-%d"),
                    "hour_local": now_local.hour,
                    "day_of_week": now_local.strftime("%A"),
                    # Friday/Saturday is the Saudi weekend
                    "is_weekend": int(now_local.weekday() in (4, 5)),
                    "road": road,
                    "point_id": f"{road.replace(' ', '_')}_{i}",
                    "lat": lat,
                    "lon": lon,
                    "seg_lat": seg_lat,
                    "seg_lon": seg_lon,
                    "frc": d.get("frc"),
                    "current_speed_kmh": cur,
                    "free_flow_speed_kmh": free,
                    "current_travel_time_s": cur_tt,
                    "free_flow_travel_time_s": free_tt,
                    "speed_ratio": ratio,
                    "delay_s": delay,
                    "confidence": d.get("confidence"),
                    "road_closure": d.get("roadClosure"),
                    "congestion_level": classify(ratio),
                })
                ok += 1

            except Exception as e:
                status = getattr(e, "status", None)
                print(f"  {road} p{i}: FAILED (HTTP {status or 'n/a'}) - {e}", file=sys.stderr)
                failed += 1
                key = status or "network"
                statuses[key] = statuses.get(key, 0) + 1
                if status in (403, 429):
                    # The key or its quota is the problem: the remaining points
                    # would fail too, and each 429 retry waits minutes.
                    key_rejected = True
                    print("  Skipping the remaining points this run (key rejected or quota used up).",
                          file=sys.stderr)

            time.sleep(0.3)   # keep under the free-tier QPS limit

    if rows:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if new_file:
                w.writeheader()
            w.writerows(rows)

    total = ok + failed
    print(f"[{now_local:%Y-%m-%d %H:%M}] wrote {ok} of {total} rows"
          + (f", {failed} failed ({', '.join(f'{k}: {v}' for k, v in statuses.items())})" if failed else ""))
    return ok, failed


def budget_report():
    """Requests per month against TomTom's MONTHLY free limits (31-day month)."""
    points = sum(len(p) for p in ROADS.values())
    runs_per_day = 24 * 60 // POLL_INTERVAL_MINUTES
    runs_per_month = runs_per_day * DAYS_PER_MONTH
    budgets = [
        ("Traffic Flow", points, FLOW_MONTHLY_LIMIT),
        ("Incident Details", INCIDENT_REQUESTS_PER_RUN, INCIDENTS_MONTHLY_LIMIT),
    ]
    print(f"One run every {POLL_INTERVAL_MINUTES} min = {runs_per_day} runs/day "
          f"= {runs_per_month:,} runs in a {DAYS_PER_MONTH}-day month")
    over = False
    for name, per_run, limit in budgets:
        per_month = per_run * runs_per_month
        headroom = limit - per_month
        print(f"  {name:<17}: {per_run} request(s)/run x {runs_per_month:,} = {per_month:,} requests/month "
              f"(free limit {limit:,}/month) -> headroom {headroom:,}")
        over |= headroom < 0
    if over:
        print("  !! OVER A FREE LIMIT - increase POLL_INTERVAL_MINUTES or remove some points, "
              "or the quota runs out before the month ends and every request is rejected.")
    else:
        print("  OK - both fit the monthly free limits.")
    print(f"  6 months of collection ~= {runs_per_day * 180 * points:,} flow requests, "
          f"{runs_per_day * 180:,} readings per point.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="snap each point and report what road it matched")
    ap.add_argument("--once", action="store_true",
                    help="collect a single round and exit (use this with cron)")
    ap.add_argument("--budget", action="store_true",
                    help="print the monthly request budget and exit (needs no API key)")
    args = ap.parse_args()

    if args.budget:
        budget_report()
        sys.exit(0)

    if API_KEY == "PASTE_YOUR_KEY_HERE":
        sys.exit("No API key. Set TOMTOM_API_KEY or edit API_KEY in this file.")

    if args.verify:
        verify()
    elif args.once:
        written, failed = collect_once()
        if written == 0:
            # Nothing collected: fail the run so GitHub Actions shows it in red.
            sys.exit(f"ERROR: 0 rows written ({failed} point(s) failed) - nothing was collected.")
        if failed:
            print(f"WARNING: {failed} of {written + failed} points failed this run; "
                  f"{written} rows were written.")
    else:
        budget_report()
        print(f"\nCollecting every {POLL_INTERVAL_MINUTES} min into {CSV_PATH}. Ctrl+C to stop.\n")
        while True:
            collect_once()
            time.sleep(POLL_INTERVAL_MINUTES * 60)