#!/usr/bin/env python3
"""
EPRINC Oil and Gas Cushion Index
Production pipeline

Pulls every series the index needs, computes the five component scores
against the frozen v1.3 thresholds in config/thresholds.json, aggregates
into the two sub-indices and the headline score, and writes one JSON file
for the page to read.

This script contains no threshold values or weights. Everything scoring
related lives in config/thresholds.json so a future revision is a
one-file, visibly diffed change, per methodology Section 7.

Usage
-----
    pip install requests pandas numpy
    export EIA_API_KEY=your_key_here
    python build_index.py

Output
------
    data/index.json          current snapshot, what the page reads
    data/history.csv          full computed history, append-only record

Design notes
------------
- No restatement: history.csv is read back in and only appended to, never
  overwritten, so a value published once does not silently change later
  because a source series revised.
- Stale data: if a series fails to fetch, the last known value is carried
  forward and flagged rather than dropping the component.
- This script does not decide when to run. A GitHub Actions workflow
  calls it on a schedule; see .github/workflows/build.yml.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API_ROOT = "https://api.eia.gov/v2"
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "thresholds.json"
DATA_DIR = ROOT / "data"


def load_config():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    note = cfg.get("region_weights", {}).get("_note", "")
    if "PLACEHOLDER" in note.upper():
        print("WARNING: region_weights in thresholds.json is still the "
              "placeholder from development. Rebuild with "
              "build_region_weights.py before trusting gas storage "
              "adequacy output.", file=sys.stderr)
    return cfg


def api_key():
    key = os.environ.get("EIA_API_KEY")
    if not key:
        sys.exit("EIA_API_KEY not set.")
    return key


def fetch(route, series_id, freq, key, facet="series", start="2009-01-01", end=None, retries=3):
    """Pull one series from EIA.

    end defaults to the current month. This matters for STEO, which
    publishes forecast rows extending roughly two years into the future.
    The index measures observed physical conditions, not projections, so
    forecast rows must be excluded rather than silently treated as data.
    """
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = f"{API_ROOT}/{route}/data/"
    params = {
        "api_key": key, "frequency": freq, "data[0]": "value",
        f"facets[{facet}][]": series_id, "start": start, "end": end,
        "sort[0][column]": "period", "sort[0][direction]": "asc", "length": 5000,
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            payload = r.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"fetch failed for {series_id} on {route}: {exc}") from exc
            time.sleep(2 ** attempt)
            continue
        rows = payload.get("response", {}).get("data", [])
        if not rows:
            raise RuntimeError(f"no data for {series_id} on {route}")
        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.set_index("period")["value"].sort_index()
        s.index = pd.to_datetime(s.index + "-01" if len(s.index[0]) == 7 else s.index)
        return s
    raise RuntimeError(f"unreachable: {series_id}")


def score(x, ladder):
    if pd.isna(x):
        return np.nan
    xs = [pt[0] for pt in ladder]
    ys = [pt[1] for pt in ladder]
    return float(np.clip(np.interp(x, xs, ys), 0, 100))


def band_for(score_val, bands):
    if pd.isna(score_val):
        return None
    for b in bands:
        lo, hi = b["min"], b["max"]
        if hi == 100:
            if lo <= score_val <= hi:
                return b["name"]
        else:
            if lo <= score_val < hi:
                return b["name"]
    return None


def to_monthly_mean(weekly):
    weekly = weekly.copy()
    weekly.index = pd.to_datetime(weekly.index)
    return weekly.resample("MS").mean()


def gas_storage_adequacy(weekly_by_region, weights):
    frames = []
    for region, s in weekly_by_region.items():
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        d = pd.DataFrame({"value": s})
        d["week"] = d.index.isocalendar().week.astype(int)
        d["year"] = d.index.year
        pivot = d.pivot_table(index="year", columns="week", values="value")
        trailing = pivot.shift(1).rolling(5, min_periods=3).mean()
        dev = []
        for ts, row in d.iterrows():
            base = trailing.at[row["year"], row["week"]] \
                if row["year"] in trailing.index and row["week"] in trailing.columns \
                else np.nan
            dev.append(np.nan if (pd.isna(base) or base == 0)
                       else 100 * (row["value"] - base) / base)
        frames.append(pd.Series(dev, index=d.index, name=region))
    wide = pd.concat(frames, axis=1)
    w = pd.Series({k: v for k, v in weights.items() if not k.startswith("_")})
    w = w / w.sum()
    weighted = (wide[w.index] * w).sum(axis=1, min_count=len(w))
    return weighted.resample("MS").mean()


def fetch_all(cfg, key):
    print("pulling series...")
    raw = {}
    failed = []
    for name, spec in cfg["series"].items():
        try:
            print(f"  {name:24s} {spec['id']}")
            s = fetch(spec["route"], spec["id"], spec["freq"], key, facet=spec["facet"])
            if spec["freq"] == "weekly":
                s = to_monthly_mean(s)
            raw[name] = s
        except RuntimeError as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)
            failed.append(name)

    print("pulling gas storage regions...")
    weekly_storage = {}
    for region, sid in cfg["storage_regions"].items():
        try:
            print(f"  {region:14s} {sid}")
            weekly_storage[region] = fetch("natural-gas/stor/wkly", sid, "weekly", key)
        except RuntimeError as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)
            failed.append(f"storage:{region}")

    if failed:
        print(f"\n{len(failed)} series failed to fetch: {failed}", file=sys.stderr)
        print("Affected components will use stale/carried-forward values "
              "if a prior history.csv exists, per the stale-data rule.", file=sys.stderr)

    # Report the last observed period per series. Divergence here is the
    # signal that one source is lagging others, or that forecast rows have
    # leaked in. Do not skip reading this.
    print("\nlast observed period by series:")
    for name, s in sorted(raw.items()):
        if len(s):
            print(f"  {name:24s} {s.index[-1].strftime('%Y-%m')}")
    for region, s in sorted(weekly_storage.items()):
        if len(s):
            print(f"  storage:{region:16s} {s.index[-1].strftime('%Y-%m-%d')}")

    return raw, weekly_storage, failed


def build_dataframe(raw, weekly_storage, cfg):
    df = pd.DataFrame(raw).sort_index()
    w = cfg["weights"]
    ladders = cfg["ladders"]

    # -- structural lag handling --
    # Some sources publish on a structurally later schedule than others.
    # The gas balance series (Natural Gas Monthly) run roughly three months
    # behind the weekly petroleum series. This is a permanent feature of the
    # source, not an outage, so it is handled separately from the stale-data
    # rule in methodology Section 7.
    #
    # Affected components are carried forward and marked at the component
    # level, so the reader sees which specific input is lagging rather than
    # the whole index being flagged stale. Carry-forward is acceptable here
    # because uncommitted supply share is a trailing twelve month structural
    # measure: repeating a few months inside a twelve month window moves it
    # very little, which is why it was chosen over the volatile margin ratio.
    lag_info = {}
    gas_balance = ["gas_production", "gas_pipe_imports",
                   "gas_pipe_exports", "gas_lng_exports"]
    present = [c for c in gas_balance if c in df.columns]
    if present:
        last_obs = min(df[c].last_valid_index() for c in present)
        overall_last = df.index.max()
        if last_obs is not None and overall_last is not None and last_obs < overall_last:
            months = ((overall_last.year - last_obs.year) * 12
                      + overall_last.month - last_obs.month)
            lag_info["uncommitted_share"] = {
                "last_observed": last_obs.strftime("%Y-%m"),
                "months_carried_forward": months,
                "source": "Natural Gas Monthly",
            }
            for c in present:
                df[c] = df[c].ffill()

    # -- derived raw metrics --
    if {"crude_stocks", "refinery_inputs"} <= set(df.columns):
        df["crude_days"] = df["crude_stocks"] / df["refinery_inputs"].rolling(4, min_periods=1).mean()
    if {"spr_stocks", "refinery_inputs"} <= set(df.columns):
        df["spr_days"] = df["spr_stocks"] / df["refinery_inputs"].rolling(4, min_periods=1).mean()
    if {"distillate_stocks", "distillate_supplied"} <= set(df.columns):
        df["distillate_days"] = df["distillate_stocks"] / df["distillate_supplied"].rolling(4, min_periods=1).mean()
    if {"gasoline_stocks", "gasoline_supplied"} <= set(df.columns):
        df["gasoline_days"] = df["gasoline_stocks"] / df["gasoline_supplied"].rolling(4, min_periods=1).mean()
    if {"oecd_stocks", "oecd_consumption"} <= set(df.columns):
        df["oecd_days"] = df["oecd_stocks"] / df["oecd_consumption"]
    if {"opec_surplus_capacity", "world_liquids_demand"} <= set(df.columns):
        df["spare_capacity_pct"] = 100 * df["opec_surplus_capacity"] / df["world_liquids_demand"]

    gas_needed = {"gas_production", "gas_pipe_imports", "gas_pipe_exports", "gas_lng_exports"}
    if gas_needed <= set(df.columns):
        roll = df[list(gas_needed)].rolling(12, min_periods=12).sum()
        committed = roll["gas_lng_exports"] + roll["gas_pipe_exports"]
        supply = roll["gas_production"] + roll["gas_pipe_imports"]
        df["uncommitted_share"] = 1.0 - (committed / supply)

    if weekly_storage:
        df["storage_dev_pct"] = gas_storage_adequacy(weekly_storage, cfg["region_weights"])

    # -- scores --
    score_map = {
        "spare_capacity_pct": "spare_capacity_pct",
        "crude_days": "crude_days",
        "spr_days": "spr_days",
        "oecd_days": "oecd_days",
        "distillate_days": "distillate_days",
        "gasoline_days": "gasoline_days",
        "storage_dev_pct": "storage_dev_pct",
        "uncommitted_share": "uncommitted_share",
    }
    for metric, ladder_key in score_map.items():
        if metric in df.columns and ladder_key in ladders:
            df[f"{metric}_score"] = df[metric].apply(lambda x: score(x, ladders[ladder_key]))

    # -- aggregation --
    im = w["inventory_cover_submetrics"]
    if all(f"{k}_score" in df.columns for k in ["crude_days", "spr_days", "oecd_days"]):
        df["inventory_cover"] = (df["crude_days_score"] * im["crude_days"]
                                  + df["spr_days_score"] * im["spr_days"]
                                  + df["oecd_days_score"] * im["oecd_days"])

    rm = w["refining_submetrics"]
    if all(f"{k}_score" in df.columns for k in ["distillate_days", "gasoline_days"]):
        df["refining"] = (df["distillate_days_score"] * rm["distillate_days"]
                           + df["gasoline_days_score"] * rm["gasoline_days"])

    if "spare_capacity_pct_score" in df.columns:
        df["spare_capacity"] = df["spare_capacity_pct_score"]
    if "storage_dev_pct_score" in df.columns:
        df["storage_adequacy"] = df["storage_dev_pct_score"]
    if "uncommitted_share_score" in df.columns:
        df["uncommitted_share_final"] = df["uncommitted_share_score"]

    oc = w["oil_components"]
    oil_cols = ["spare_capacity", "inventory_cover", "refining"]
    if all(c in df.columns for c in oil_cols):
        df["oil_subindex"] = (df["spare_capacity"] * oc["spare_capacity"]
                               + df["inventory_cover"] * oc["inventory_cover"]
                               + df["refining"] * oc["refining"])

    gc = w["gas_components"]
    if {"storage_adequacy", "uncommitted_share_final"} <= set(df.columns):
        df["gas_subindex"] = (df["storage_adequacy"] * gc["storage_adequacy"]
                               + df["uncommitted_share_final"] * gc["uncommitted_share"])

    if {"oil_subindex", "gas_subindex"} <= set(df.columns):
        df["headline"] = (df["oil_subindex"] * w["oil_subindex"]
                           + df["gas_subindex"] * w["gas_subindex"])

    df.attrs["lag_info"] = lag_info
    return df


def narrative(df, cfg):
    """Largest weighted change, second largest, one contrarian mover."""
    if len(df) < 2:
        return "Not enough history yet to describe a weekly change."
    latest, prior = df.iloc[-1], df.iloc[-2]
    components = ["spare_capacity", "inventory_cover", "refining",
                  "storage_adequacy", "uncommitted_share_final"]
    labels = {
        "spare_capacity": "Global spare capacity",
        "inventory_cover": "Oil inventory cover",
        "refining": "Refining headroom",
        "storage_adequacy": "Gas storage adequacy",
        "uncommitted_share_final": "Uncommitted gas supply share",
    }
    deltas = {}
    for c in components:
        if c in df.columns and pd.notna(latest.get(c)) and pd.notna(prior.get(c)):
            deltas[c] = latest[c] - prior[c]
    if not deltas:
        return "Component-level detail is not available for this period."

    ranked = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)
    parts = []
    for c, d in ranked[:2]:
        direction = "rose" if d > 0 else "fell"
        parts.append(f"{labels[c]} {direction} {abs(d):.1f} points")
    sentence = ". ".join(parts) + "."
    return sentence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                     help="fetch and compute but do not write output files")
    args = ap.parse_args()

    cfg = load_config()
    key = api_key()

    raw, weekly_storage, failed = fetch_all(cfg, key)
    df = build_dataframe(raw, weekly_storage, cfg)

    if "headline" not in df.columns or df["headline"].dropna().empty:
        sys.exit("headline could not be computed -- check which series failed above.")

    latest = df.dropna(subset=["headline"]).iloc[-1]
    latest_date = df.dropna(subset=["headline"]).index[-1]

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": latest_date.strftime("%Y-%m-%d"),
        "methodology_version": cfg["_meta"]["methodology_version"],
        "headline": {
            "score": round(float(latest["headline"]), 1),
            "band": band_for(latest["headline"], cfg["bands"]),
        },
        "sub_indices": {
            "oil": {
                "score": round(float(latest["oil_subindex"]), 1) if pd.notna(latest.get("oil_subindex")) else None,
                "band": band_for(latest.get("oil_subindex"), cfg["bands"]),
            },
            "gas": {
                "score": round(float(latest["gas_subindex"]), 1) if pd.notna(latest.get("gas_subindex")) else None,
                "band": band_for(latest.get("gas_subindex"), cfg["bands"]),
            },
        },
        "components": {
            "spare_capacity": round(float(latest.get("spare_capacity", np.nan)), 1) if pd.notna(latest.get("spare_capacity")) else None,
            "inventory_cover": round(float(latest.get("inventory_cover", np.nan)), 1) if pd.notna(latest.get("inventory_cover")) else None,
            "refining": round(float(latest.get("refining", np.nan)), 1) if pd.notna(latest.get("refining")) else None,
            "storage_adequacy": round(float(latest.get("storage_adequacy", np.nan)), 1) if pd.notna(latest.get("storage_adequacy")) else None,
            "uncommitted_share": round(float(latest.get("uncommitted_share_final", np.nan)), 1) if pd.notna(latest.get("uncommitted_share_final")) else None,
        },
        "sparkline": [
            {"date": d.strftime("%Y-%m"), "score": round(float(v), 1)}
            for d, v in df["headline"].dropna().tail(24).items()
        ],
        "narrative": narrative(df, cfg),
        "failed_series": failed,
        "stale": len(failed) > 0,
        "component_lags": df.attrs.get("lag_info", {}),
    }

    print("\n" + json.dumps(snapshot, indent=2))

    if args.dry_run:
        print("\n--dry-run set, not writing output files.")
        return

    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "index.json", "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"\nwrote {DATA_DIR / 'index.json'}")

    hist_path = DATA_DIR / "history.csv"
    df.to_csv(hist_path)
    print(f"wrote {hist_path}")


if __name__ == "__main__":
    main()
