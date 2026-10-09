"""
Masar - TomTom traffic incidents collector (Jeddah).

One request per run covers all monitored roads (a bounding box), so it adds
only 144 requests/day at a 10-minute interval.

Files it maintains:
  jeddah_incidents.csv       one row per incident (no duplicates). Each run
                             updates last_seen_utc / times_seen / end time of
                             incidents that are still active.
  jeddah_incidents_runs.csv  one row per run: how many incidents of each type
                             were active. Shows that a run happened even when
                             nothing was found (so "no accidents" is real data,
                             not a missed run).

Usage:
  export TOMTOM_API_KEY="..."
  python tomtom_incidents.py
"""

import csv
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

import requests

from tomtom_collector import ROADS, haversine_km

API_KEY = os.environ.get("TOMTOM_API_KEY", "")
URL = "https://api.tomtom.com/traffic/services/5/incidentDetails"
INCIDENTS_CSV = "jeddah_incidents.csv"
RUNS_CSV = "jeddah_incidents_runs.csv"
SAUDI_TZ = timezone(timedelta(hours=3))
BBOX_PAD_DEG = 0.02  # ~2 km around the monitored points

CATEGORIES = {
    0: "Unknown", 1: "Accident", 2: "Fog", 3: "Dangerous Conditions", 4: "Rain",
    5: "Ice", 6: "Jam", 7: "Lane Closed", 8: "Road Closed", 9: "Road Works",
    10: "Wind", 11: "Flooding", 14: "Broken Down Vehicle",
}
MAGNITUDE = {0: "Unknown", 1: "Minor", 2: "Moderate", 3: "Major", 4: "Undefined"}

FIELDS = ("{incidents{type,geometry{type,coordinates},properties{id,iconCategory,"
          "magnitudeOfDelay,events{description,code,iconCategory},startTime,endTime,"
          "from,to,length,delay,roadNumbers,probabilityOfOccurrence,numberOfReports,"
          "lastReportTime}}}")

COLUMNS = [
    "incident_id", "category_code", "category", "description", "magnitude_of_delay",
    "start_time_utc", "end_time_utc", "first_seen_utc", "last_seen_utc", "times_seen",
    "from_road", "to_road", "length_m", "delay_s", "road_numbers",
    "start_lat", "start_lon", "end_lat", "end_lon",
    "nearest_point_id", "nearest_point_km",
    "probability", "number_of_reports", "last_report_time_utc",
]
RUN_COLUMNS = ["timestamp_utc", "timestamp_local", "total"] + [
    CATEGORIES[k].lower().replace(" ", "_") for k in sorted(CATEGORIES)]

POINTS = [(f"{road.replace(' ', '_')}_{i}", lat, lon)
          for road, pts in ROADS.items() for i, (lat, lon) in enumerate(pts, 1)]


def bbox():
    lats = [p[1] for p in POINTS]
    lons = [p[2] for p in POINTS]
    return (f"{min(lons) - BBOX_PAD_DEG:.4f},{min(lats) - BBOX_PAD_DEG:.4f},"
            f"{max(lons) + BBOX_PAD_DEG:.4f},{max(lats) + BBOX_PAD_DEG:.4f}")


def fetch():
    params = {"key": API_KEY, "bbox": bbox(), "fields": FIELDS,
              "language": "en-GB", "timeValidityFilter": "present"}
    try:
        r = requests.get(URL, params=params, timeout=30)
    except requests.RequestException as e:
        sys.exit("Incidents request failed: " + str(e).replace(API_KEY, "***"))
    if r.status_code != 200:
        sys.exit(f"Incidents request failed: HTTP {r.status_code} "
                 f"{r.text[:200].replace(API_KEY, '***')}")
    return r.json().get("incidents", [])


def coords_of(geom):
    """Return a list of (lat, lon) from a GeoJSON Point or LineString."""
    c = (geom or {}).get("coordinates") or []
    if not c:
        return []
    if isinstance(c[0], (int, float)):
        return [(c[1], c[0])]
    return [(pt[1], pt[0]) for pt in c]


def nearest_point(coords):
    best = (None, None)
    for pid, lat, lon in POINTS:
        for la, lo in coords:
            d = haversine_km((la, lo), (lat, lon))
            if best[1] is None or d < best[1]:
                best = (pid, d)
    return best[0], (round(best[1], 3) if best[1] is not None else "")


def to_row(inc, now_utc):
    p = inc.get("properties", {})
    coords = coords_of(inc.get("geometry"))
    pid, dist = nearest_point(coords) if coords else ("", "")
    code = p.get("iconCategory")
    return {
        "incident_id": p.get("id", ""),
        "category_code": code,
        "category": CATEGORIES.get(code, str(code)),
        "description": "; ".join(e.get("description", "") for e in p.get("events") or []),
        "magnitude_of_delay": MAGNITUDE.get(p.get("magnitudeOfDelay"), p.get("magnitudeOfDelay", "")),
        "start_time_utc": p.get("startTime") or "",
        "end_time_utc": p.get("endTime") or "",
        "first_seen_utc": now_utc,
        "last_seen_utc": now_utc,
        "times_seen": 1,
        "from_road": p.get("from") or "",
        "to_road": p.get("to") or "",
        "length_m": p.get("length", ""),
        "delay_s": p.get("delay", ""),
        "road_numbers": "|".join(p.get("roadNumbers") or []),
        "start_lat": coords[0][0] if coords else "",
        "start_lon": coords[0][1] if coords else "",
        "end_lat": coords[-1][0] if coords else "",
        "end_lon": coords[-1][1] if coords else "",
        "nearest_point_id": pid,
        "nearest_point_km": dist,
        "probability": p.get("probabilityOfOccurrence", ""),
        "number_of_reports": p.get("numberOfReports", ""),
        "last_report_time_utc": p.get("lastReportTime") or "",
    }


def load(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["incident_id"]: r for r in csv.DictReader(f)}


def write_atomic(path, columns, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main():
    if not API_KEY:
        sys.exit("No API key. Set TOMTOM_API_KEY.")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    now_utc = now.strftime("%Y-%m-%d %H:%M:%S")
    incidents = fetch()

    table = load(INCIDENTS_CSV)
    new = 0
    for inc in incidents:
        row = to_row(inc, now_utc)
        iid = row["incident_id"]
        if not iid:
            continue
        if iid in table:
            old = table[iid]
            old["last_seen_utc"] = now_utc
            old["times_seen"] = int(old.get("times_seen") or 0) + 1
            for k in ("end_time_utc", "description", "magnitude_of_delay", "length_m",
                      "delay_s", "probability", "number_of_reports", "last_report_time_utc"):
                if row[k] not in ("", None):
                    old[k] = row[k]
        else:
            table[iid] = row
            new += 1
    rows = sorted(table.values(), key=lambda r: (r["first_seen_utc"], r["incident_id"]))
    write_atomic(INCIDENTS_CSV, COLUMNS, rows)

    counts = Counter(inc.get("properties", {}).get("iconCategory") for inc in incidents)
    run = {"timestamp_utc": now_utc,
           "timestamp_local": now.astimezone(SAUDI_TZ).strftime("%Y-%m-%d %H:%M:%S"),
           "total": len(incidents)}
    for k in sorted(CATEGORIES):
        run[CATEGORIES[k].lower().replace(" ", "_")] = counts.get(k, 0)
    exists = os.path.exists(RUNS_CSV)
    with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow(run)

    print(f"[{run['timestamp_local']}] {len(incidents)} active incidents, {new} new; "
          f"accidents now: {counts.get(1, 0)}")


if __name__ == "__main__":
    main()
