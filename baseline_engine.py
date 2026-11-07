"""Module 2: baseline engine (no LLM).

Turns each normalised alert (D2) into a baseline assessment (D3):
  - known_false_positive : rule seen before and almost always harmless
  - risk_score           : 0-100 from severity, asset criticality and repetition
  - priority             : level (critical/high/medium/low) plus the score
  - classification       : true_positive / false_positive (binary, like the ground truth)
  - severity_only_level  : rule.level alone, if the alert carries one

Keep this file in the same folder as alerts_loader.py.

Usage (after running alerts_loader.py, with the SAME file and --window):
    python baseline_engine.py russellmitchell_alerts.txt \
        --pilot pilot_d2.jsonl --truth pilot_truth.jsonl --out pilot_d3.jsonl

IMPORTANT: the rule-prior and known-false-positive list are learned from a
development set only. In this script the development set is every case that is
NOT in the pilot. That is fine as a smoke test. For the real evaluation, split
by time (earlier alerts for learning, later alerts for testing).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import pandas as pd

from alerts_loader import clean_text, collapse_repeats, load_alerts

# ASSUMPTIONS: edit these, then FIX them before the real evaluation.
DEFAULT_ASSET_KEYWORDS = {
    "firewall": 0.9, "dns": 0.9, "domain": 0.9,
    "web": 0.8, "mail": 0.7, "file": 0.6, "share": 0.6,
}
DEFAULT_ASSET_SCORE = 0.5
PRIORITY_BANDS = [(80, "critical"), (60, "high"), (40, "medium"), (0, "low")]
SEVERITY_ONLY_BANDS = [(13, "critical"), (10, "high"), (7, "medium"), (0, "low")]  # rule.level 0-15
UNSEEN_RULE_PRIOR = 0.5  # a rule never seen in the development set gets a neutral prior, not a benign one


@dataclass
class Weights:
    severity: float = 0.6
    asset: float = 0.2
    repetition: float = 0.2


class RulePrior:
    """Per-rule attack rate learned from a labelled development set only."""

    def __init__(self, min_support: int = 5, fp_threshold: float = 0.98):
        self.min_support = min_support
        self.fp_threshold = fp_threshold
        self.stats: dict[str, tuple[int, int]] = {}  # rule -> (cases, attack cases)

    def fit(self, dev_cases: pd.DataFrame) -> "RulePrior":
        keys = dev_cases["name"].map(clean_text)
        g = dev_cases.assign(_k=keys).groupby("_k")["is_attack"].agg(["size", "sum"])
        self.stats = {k: (int(r["size"]), int(r["sum"])) for k, r in g.iterrows()}
        return self

    def attack_prior(self, rule: str) -> float:
        n, a = self.stats.get(rule, (0, 0))
        return (a + 1) / (n + 2) if n else UNSEEN_RULE_PRIOR

    def is_known_fp(self, rule: str) -> bool:
        n, a = self.stats.get(rule, (0, 0))
        return n >= self.min_support and (1 - a / n) >= self.fp_threshold


def asset_criticality(agent: str, table: dict[str, float] = DEFAULT_ASSET_KEYWORDS) -> float:
    a = (agent or "").lower()
    return max([v for k, v in table.items() if k in a], default=DEFAULT_ASSET_SCORE)


def repetition_component(repeat_count: int) -> float:
    """About 99 repeats saturates at 1.0 (log scale)."""
    return min(1.0, math.log10(1 + max(int(repeat_count), 1)) / 2.0)


def severity_component(d2: dict, prior: RulePrior | None) -> tuple[float, str]:
    """Use rule.level when the alert has one; otherwise fall back to the learned rule prior."""
    level = d2.get("rule_level")
    if level is not None:
        return min(max(float(level), 0.0), 15.0) / 15.0, "rule_level"
    if prior is not None:
        return prior.attack_prior(d2["rule_description"]), "rule_prior"
    return 0.5, "none"


def band(value: float, bands: list[tuple[float, str]]) -> str:
    for threshold, name in bands:
        if value >= threshold:
            return name
    return bands[-1][1]


def baseline_assess(
    d2: dict,
    prior: RulePrior | None = None,
    weights: Weights = Weights(),
    tp_threshold: float = 50.0,
) -> dict:
    """Build the D3 object for one normalised alert."""
    known_fp = bool(prior and prior.is_known_fp(d2["rule_description"]))
    sev, sev_source = severity_component(d2, prior)
    asset = asset_criticality(d2.get("agent", ""))
    rep = repetition_component(d2.get("repeat_count", 1))
    score = round(100 * (weights.severity * sev + weights.asset * asset + weights.repetition * rep), 1)
    if known_fp:
        score = min(score, 10.0)  # a known false positive sinks to the bottom of the queue
    classification = "false_positive" if known_fp or score < tp_threshold else "true_positive"
    level = d2.get("rule_level")
    return {
        "alert_id": d2["alert_id"],
        "known_false_positive": known_fp,
        "severity_source": sev_source,
        "components": {"severity": round(sev, 3), "asset": round(asset, 3), "repetition": round(rep, 3)},
        "risk_score": score,
        "priority": {"level": band(score, PRIORITY_BANDS), "score": score},
        "classification": classification,
        "severity_only_level": band(float(level), SEVERITY_ONLY_BANDS) if level is not None else None,
    }


def binary_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="the same alert file used for the pilot")
    ap.add_argument("--pilot", required=True, help="pilot_d2.jsonl from alerts_loader.py")
    ap.add_argument("--truth", help="pilot_truth.jsonl, only to print a quick sanity check")
    ap.add_argument("--window", type=int, default=300, help="must match the window used for the pilot")
    ap.add_argument("--out", default="pilot_d3.jsonl")
    args = ap.parse_args()

    cases = collapse_repeats(load_alerts(args.csv), args.window)
    pilot = read_jsonl(args.pilot)
    pilot_ids = {p["alert_id"] for p in pilot}
    dev = cases[~cases["case_id"].isin(pilot_ids)]
    prior = RulePrior().fit(dev)
    print(f"development cases: {len(dev)} | pilot cases: {len(pilot)} | rules learned: {len(prior.stats)}")

    d3 = [baseline_assess(p, prior) for p in pilot]
    with open(args.out, "w", encoding="utf-8") as f:
        for row in d3:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(d3)} baseline assessments to {args.out}")
    print("known false positives flagged:", sum(r["known_false_positive"] for r in d3))
    print("severity source used:", pd.Series([r["severity_source"] for r in d3]).value_counts().to_dict())

    if args.truth:
        truth = {t["alert_id"]: t["is_attack"] for t in read_jsonl(args.truth)}
        y_true = [truth[r["alert_id"]] for r in d3]
        y_pred = [r["classification"] == "true_positive" for r in d3]
        print("sanity check vs truth:", binary_metrics(y_true, y_pred))


if __name__ == "__main__":
    main()
