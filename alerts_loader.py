"""Module 1: alert loader, normaliser, repeat collapser and pilot sampler.

Reads one AIT-style alert CSV with the columns
    time, name, ip, host, short, time_label, event_label
and produces:
  - D2-like normalised alerts (the only fields the LLM may ever see)
  - a separate ground-truth file (labels, never shown to the LLM)
  - a small balanced pilot set

Usage:
    python alerts_loader.py russellmitchell_alerts.txt --n-attack 10 --n-other 10
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd

EXPECTED_COLS = ["time", "name", "ip", "host", "short", "time_label", "event_label"]
KEY_COLS = ["name", "ip", "host", "short", "time_label", "event_label"]

# ASSUMPTION: rows whose time_label is exactly this token are NOT attacks.
# Check it against the printed value counts before trusting any result.
FALSE_POSITIVE_TOKEN = "false_positive"

MAX_TEXT = 300  # cap on any text field (log text is untrusted)
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def clean_text(value) -> str:
    """Strip control characters and cap the length."""
    return _CTRL.sub("", str(value)).strip()[:MAX_TEXT]


def load_alerts(path: str) -> pd.DataFrame:
    """Load one alert file. Rows with an unreadable timestamp are dropped and counted."""
    df = pd.read_csv(path, dtype=str, on_bad_lines="warn", keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")
    df = df[EXPECTED_COLS].copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    dropped = int(df["time"].isna().sum())
    df = df.dropna(subset=["time"])
    df["time"] = df["time"].astype("int64")
    df["is_attack"] = df["time_label"].str.strip() != FALSE_POSITIVE_TOKEN
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    df.attrs["dropped_rows"] = dropped
    return df


def collapse_repeats(df: pd.DataFrame, window_s: int = 300) -> pd.DataFrame:
    """Merge identical alerts repeating within window_s seconds into one case with a count.

    The labels are part of the key, so a case never mixes attack and non-attack rows.
    """
    df = df.sort_values("time", kind="stable").copy()
    gap = df.groupby(KEY_COLS, sort=False)["time"].diff()
    df["_new"] = (gap.isna() | (gap > window_s)).astype(int)
    df["_case_no"] = df.groupby(KEY_COLS, sort=False)["_new"].cumsum()
    cases = (
        df.groupby(KEY_COLS + ["_case_no"], sort=False)
        .agg(first_seen=("time", "min"), last_seen=("time", "max"), repeat_count=("time", "size"))
        .reset_index()
        .drop(columns="_case_no")
    )
    cases["is_attack"] = cases["time_label"].str.strip() != FALSE_POSITIVE_TOKEN
    cases = cases.sort_values("first_seen", kind="stable").reset_index(drop=True)
    cases.insert(0, "case_id", [f"C{i:06d}" for i in range(len(cases))])
    return cases


def _iso(epoch: int) -> str:
    return pd.to_datetime(int(epoch), unit="s", utc=True).isoformat()


def to_d2(case) -> dict:
    """Fields the pipeline may show the LLM (D2). No labels in here."""
    return {
        "alert_id": case["case_id"],
        "first_seen": _iso(case["first_seen"]),
        "last_seen": _iso(case["last_seen"]),
        "repeat_count": int(case["repeat_count"]),
        "rule_description": clean_text(case["name"]),
        "rule_tag": clean_text(case["short"]),
        "agent": clean_text(case["host"]),
        "source_ip": clean_text(case["ip"]),
        "rule_level": None,   # not in this CSV; needed later for the severity baseline
        "log_excerpt": None,  # not in this CSV
    }


def to_truth(case) -> dict:
    """Ground truth, kept apart from D2 so labels can never leak into a prompt."""
    return {
        "alert_id": case["case_id"],
        "is_attack": bool(case["is_attack"]),
        "time_label": case["time_label"],
        "event_label": case["event_label"],
    }


def sample_pilot(cases: pd.DataFrame, n_attack: int = 10, n_other: int = 10, seed: int = 42) -> pd.DataFrame:
    """Random balanced sample of cases for the first pilot runs."""
    atk = cases[cases["is_attack"]]
    oth = cases[~cases["is_attack"]]
    pick = pd.concat(
        [
            atk.sample(min(n_attack, len(atk)), random_state=seed),
            oth.sample(min(n_other, len(oth)), random_state=seed),
        ]
    )
    return pick.sort_values("first_seen").reset_index(drop=True)


def summarise(df: pd.DataFrame, cases: pd.DataFrame) -> None:
    print(f"raw alerts: {len(df)} (rows dropped for bad timestamp: {df.attrs.get('dropped_rows', 0)})")
    print(f"collapsed cases: {len(cases)} (compression x{len(df) / max(len(cases), 1):.1f})")
    print(f"attack share, raw: {df['is_attack'].mean():.2%}, cases: {cases['is_attack'].mean():.2%}")
    print("\ntime_label values:\n" + df["time_label"].value_counts().head(20).to_string())
    print("\nevent_label values:\n" + df["event_label"].value_counts().head(20).to_string())
    print("\ntop rule names:\n" + df["name"].value_counts().head(15).to_string())


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="path to one alert file")
    ap.add_argument("--window", type=int, default=300, help="repeat window in seconds")
    ap.add_argument("--n-attack", type=int, default=10)
    ap.add_argument("--n-other", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="pilot")
    args = ap.parse_args()

    df = load_alerts(args.csv)
    cases = collapse_repeats(df, args.window)
    summarise(df, cases)

    pilot = sample_pilot(cases, args.n_attack, args.n_other, args.seed)
    write_jsonl(f"{args.out_prefix}_d2.jsonl", [to_d2(r) for _, r in pilot.iterrows()])
    write_jsonl(f"{args.out_prefix}_truth.jsonl", [to_truth(r) for _, r in pilot.iterrows()])
    print(f"\nwrote {len(pilot)} pilot cases to {args.out_prefix}_d2.jsonl and {args.out_prefix}_truth.jsonl")


if __name__ == "__main__":
    main()
