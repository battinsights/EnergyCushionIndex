#!/usr/bin/env python3
"""
EPRINC Oil and Gas Cushion Index
Gas storage region weight builder

Builds the five regional weights used by the gas storage adequacy
component, from state-level January residential plus commercial natural
gas consumption, aggregated to EIA storage regions.

Why residential plus commercial rather than total consumption
--------------------------------------------------------------
The component asks whether gas is located where it will be needed during
winter stress. Total consumption includes industrial and power burn,
which run year round and are heavily concentrated in Texas and Louisiana.
Weighting by total would give South Central a large share for reasons
unrelated to cold weather vulnerability. Residential plus commercial is
the weather sensitive portion of demand, and is the relevant measure for
a cold snap security question.

This choice materially changes the weights and must be stated explicitly
in methodology Section 4.1, not left as "January consumption share".

Method
------
1. Pull state level residential (N3010) and commercial (N3020) monthly
   consumption for every state.
2. Take January values only, averaged across the calibration window.
3. Map each state to its EIA storage region.
4. Weight each region by its share of national January res+comm demand.

Usage
-----
    export EIA_API_KEY=your_key_here
    python build_region_weights.py

Output
------
    Prints the weights block ready to paste into config/thresholds.json,
    plus the full state to region mapping table for the methodology
    appendix, and a coverage check showing any unmapped states.
"""

import json
import os
import sys
import time
from collections import defaultdict

import pandas as pd
import requests

API_ROOT = "https://api.eia.gov/v2"

# Calibration window for the weights. Uses the same frozen window as the
# rest of the index. Weights are recomputed annually and versioned.
JAN_YEARS = list(range(2015, 2026))

# ---------------------------------------------------------------------------
# State to EIA storage region mapping.
#
# EIA's five storage regions are defined for the Weekly Natural Gas Storage
# Report. The mapping below follows EIA's published region definitions.
# VERIFY this against EIA's current region definitions before publishing --
# region boundaries have been redefined once before (2015).
#
# Source to check:
#   https://www.eia.gov/naturalgas/storage/basics/
# ---------------------------------------------------------------------------

STATE_REGION = {
    # East
    "CT": "east", "DE": "east", "DC": "east", "ME": "east", "MD": "east",
    "MA": "east", "NH": "east", "NJ": "east", "NY": "east", "NC": "east",
    "OH": "east", "PA": "east", "RI": "east", "SC": "east", "VT": "east",
    "VA": "east", "WV": "east", "GA": "east", "TN": "east", "KY": "east",
    # Midwest
    "IL": "midwest", "IN": "midwest", "IA": "midwest", "KS": "midwest",
    "MI": "midwest", "MN": "midwest", "MO": "midwest", "NE": "midwest",
    "ND": "midwest", "OK": "midwest", "SD": "midwest", "WI": "midwest",
    # South Central
    "AL": "south_central", "AR": "south_central", "FL": "south_central",
    "LA": "south_central", "MS": "south_central", "NM": "south_central",
    "TX": "south_central",
    # Mountain
    "CO": "mountain", "ID": "mountain", "MT": "mountain", "UT": "mountain",
    "WY": "mountain",
    # Pacific
    "AK": "pacific", "AZ": "pacific", "CA": "pacific", "NV": "pacific",
    "OR": "pacific", "WA": "pacific", "HI": "pacific",
}

RES_PREFIX = "N3010"   # residential consumption, MMcf
COM_PREFIX = "N3020"   # commercial consumption, MMcf


def api_key():
    key = os.environ.get("EIA_API_KEY")
    if not key:
        sys.exit("EIA_API_KEY not set.")
    return key


def fetch_state(prefix, state, key, retries=3):
    """Pull one state's monthly consumption series. Returns None if absent."""
    series_id = f"{prefix}{state}2"
    url = f"{API_ROOT}/natural-gas/cons/sum/data/"
    params = {
        "api_key": key, "frequency": "monthly", "data[0]": "value",
        "facets[series][]": series_id,
        "start": f"{min(JAN_YEARS)}-01", "end": f"{max(JAN_YEARS)}-12",
        "sort[0][column]": "period", "sort[0][direction]": "asc", "length": 5000,
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            rows = r.json().get("response", {}).get("data", [])
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
            continue
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.set_index("period")["value"].sort_index()
        s.index = pd.to_datetime(s.index + "-01")
        return s
    return None


def main():
    key = api_key()
    states = sorted(STATE_REGION)

    print(f"pulling residential and commercial consumption for "
          f"{len(states)} states, January {min(JAN_YEARS)} to {max(JAN_YEARS)}...\n")

    january_by_state = {}
    missing = []

    for st in states:
        res = fetch_state(RES_PREFIX, st, key)
        com = fetch_state(COM_PREFIX, st, key)
        if res is None and com is None:
            missing.append(st)
            print(f"  {st}  NO DATA")
            continue
        total = 0.0
        parts = []
        for label, s in (("res", res), ("com", com)):
            if s is None:
                parts.append(f"{label}:missing")
                continue
            jan = s[s.index.month == 1]
            jan = jan[jan.index.year.isin(JAN_YEARS)]
            if len(jan):
                total += jan.mean()
                parts.append(f"{label}:{jan.mean():,.0f}")
            else:
                parts.append(f"{label}:no-jan")
        january_by_state[st] = total
        print(f"  {st}  {total:>12,.0f} MMcf   ({', '.join(parts)})")

    if missing:
        print(f"\nWARNING: no data returned for {missing}. These states are "
              f"excluded from the weights. Confirm whether that is correct "
              f"before publishing.", file=sys.stderr)

    # -- aggregate to regions --
    region_totals = defaultdict(float)
    for st, val in january_by_state.items():
        region_totals[STATE_REGION[st]] += val

    grand_total = sum(region_totals.values())
    weights = {r: v / grand_total for r, v in region_totals.items()}

    print("\n" + "=" * 68)
    print("REGION WEIGHTS, January residential plus commercial consumption")
    print("=" * 68)
    print(f"calibration window: January {min(JAN_YEARS)} to {max(JAN_YEARS)}\n")
    for r in ["east", "midwest", "south_central", "mountain", "pacific"]:
        print(f"  {r:16s} {region_totals[r]:>14,.0f} MMcf   {weights[r]:.4f}")
    print(f"  {'TOTAL':16s} {grand_total:>14,.0f} MMcf   {sum(weights.values()):.4f}")

    print("\n" + "=" * 68)
    print("PASTE INTO config/thresholds.json, replacing region_weights")
    print("=" * 68)
    block = {
        "_note": (f"January residential plus commercial consumption share, "
                  f"averaged {min(JAN_YEARS)} to {max(JAN_YEARS)}. "
                  f"Recompute annually and version. See methodology 4.1."),
        **{r: round(weights[r], 4) for r in
           ["east", "midwest", "south_central", "mountain", "pacific"]},
    }
    print(json.dumps({"region_weights": block}, indent=2))

    print("\n" + "=" * 68)
    print("STATE TO REGION MAPPING, for methodology appendix")
    print("=" * 68)
    by_region = defaultdict(list)
    for st, r in STATE_REGION.items():
        by_region[r].append(st)
    for r in ["east", "midwest", "south_central", "mountain", "pacific"]:
        print(f"\n{r}:")
        print("  " + ", ".join(sorted(by_region[r])))

    print("""
BEFORE PUBLISHING
-----------------
Verify the state to region mapping in this script against EIA's current
published storage region definitions. EIA redefined these regions once
before, in 2015, and the mapping here was built from the standard
definitions rather than read programmatically from EIA.
""")


if __name__ == "__main__":
    main()
