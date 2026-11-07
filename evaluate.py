"""Module 7: evaluation.

Reads the experiment files and the run log, and compares every LLM condition with
the baseline on the SAME alerts.

Quality
  precision, recall, F1 (with a bootstrap 95% interval), attack miss rate, false-positive rate.
  Invalid answers count as errors (an invalid answer is a failure, not a free pass).
  Priority: how well the priority score ranks attacks above non-attacks (AUC and
  R-precision), correlation with the baseline score, and weighted kappa when the truth
  file has a priority_level column.
Reliability
  strict-JSON rate, valid rate on the first attempt and after the retry, failure rate,
  retry rate, run-to-run agreement across repetitions.
Efficiency
  latency per alert (all attempts), tokens per second, token counts, RAM in use.
Safety screens (automatic, heuristic; they do NOT replace the human rubric)
  invented IP rate, mean evidence grounding, disruptive-command rate.
Statistics
  paired exact McNemar test on correctness (repetition 1) for every pair of conditions
  including the baseline, Holm-corrected; paired Wilcoxon on latency if scipy is installed.
  A non-significant result is NOT proof of no difference.

Usage:
  python evaluate.py --dir experiment
Writes results_summary.csv, results_pairs.csv and results_report.md into --dir.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_logger import RunLogger

BASELINE = "BASELINE risk_score"


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------------------- statistics helpers
def prf(y: np.ndarray, p: np.ndarray) -> dict:
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "miss_rate": fn / (tp + fn) if tp + fn else float("nan"),
            "fpr": fp / (fp + tn) if fp + tn else float("nan")}


def bootstrap_f1(y: np.ndarray, p: np.ndarray, b: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (b, len(y)))
    yy, pp = y[idx], p[idx]
    tp = (yy & pp).sum(1)
    fp = (~yy & pp).sum(1)
    fn = (yy & ~pp).sum(1)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))


def auc(scores, labels) -> float | None:
    pos = np.asarray(labels, bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return None
    ranks = pd.Series(np.asarray(scores, float)).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def r_precision(scores, labels) -> float | None:
    pos = np.asarray(labels, bool)
    k = int(pos.sum())
    if k == 0:
        return None
    order = np.argsort(-np.asarray(scores, float), kind="stable")
    return float(pos[order[:k]].mean())


def weighted_kappa(a: list[str], b: list[str], levels: list[str]) -> float | None:
    k = len(levels)
    idx = {l: i for i, l in enumerate(levels)}
    if not a:
        return None
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    den = (w * exp).sum()
    return float(1 - (w * obs).sum() / den) if den else None


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# ----------------------------------------------------------------------------- data shaping
def build(dir_: Path, log: str):
    truth = {t["alert_id"]: t for t in read_jsonl(dir_ / "exp_truth.jsonl")}
    base = {b["alert_id"]: b for b in read_jsonl(dir_ / "baseline_d3.jsonl")}
    runs = RunLogger.read(log)
    return truth, base, runs


def outcome(rec: dict | None, is_attack: bool) -> tuple[bool, float | None, str | None]:
    """(predicted attack, priority score, priority level). Invalid answer -> wrong prediction, no score."""
    if rec is not None and rec["valid"]:
        parsed = rec["parsed"]
        return (parsed["classification"] == "true_positive",
                float(parsed["priority"]["score"]), parsed["priority"]["level"])
    return (not is_attack), None, None


def condition_rows(cond: str, runs: list[dict], truth: dict, base: dict) -> tuple[dict, dict]:
    """Return (summary row, per-alert outcome for repetition 1)."""
    mine = [r for r in runs if r["condition"] == cond]
    reps = sorted({r["repetition"] for r in mine})
    finals = {(r["repetition"], r["alert_id"]): r for r in mine if r["final"]}
    firsts = {(r["repetition"], r["alert_id"]): r for r in mine if r["attempt"] == 1}
    per_rep, rep1_outcomes = [], {}
    pri = {"auc": [], "rprec": [], "spearman": []}
    pri_pairs: list[tuple[str, str]] = []
    for rep in reps:
        ids = [a for a in truth if (rep, a) in finals]
        y = np.array([truth[a]["is_attack"] for a in ids], bool)
        outs = [outcome(finals[(rep, a)], truth[a]["is_attack"]) for a in ids]
        p = np.array([o[0] for o in outs], bool)
        per_rep.append(prf(y, p))
        scored = [(o[1], truth[a]["is_attack"], base[a]["risk_score"], o[2], truth[a].get("priority_level"))
                  for a, o in zip(ids, outs) if o[1] is not None]
        if scored:
            sc, lab, bs, lv, tl = zip(*scored)
            pri["auc"].append(auc(sc, lab))
            pri["rprec"].append(r_precision(sc, lab))
            pri["spearman"].append(pd.Series(sc).corr(pd.Series(bs), method="spearman"))
            pri_pairs += [(a, b) for a, b in zip(lv, tl) if b]
        if rep == reps[0]:
            rep1_outcomes = {a: (o[0] == truth[a]["is_attack"]) for a, o in zip(ids, outs)}
            rep1_y, rep1_p = y, p
    row = {"condition": cond, "n_alerts": len(rep1_outcomes), "repetitions": len(reps)}
    for key in ("precision", "recall", "f1", "miss_rate", "fpr"):
        vals = [m[key] for m in per_rep]
        row[key], row[key + "_sd"] = float(np.nanmean(vals)), (float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0)
    row["f1_ci_low"], row["f1_ci_high"] = bootstrap_f1(rep1_y, rep1_p)
    for key, name in (("auc", "priority_auc"), ("rprec", "priority_r_precision"), ("spearman", "priority_vs_baseline_spearman")):
        vals = [v for v in pri[key] if v is not None and not (isinstance(v, float) and math.isnan(v))]
        row[name] = float(np.mean(vals)) if vals else None
    levels = ["low", "medium", "high", "critical"]
    pp = [(a, b) for a, b in pri_pairs if a in levels and b in levels]
    row["priority_weighted_kappa"] = weighted_kappa([a for a, _ in pp], [b for _, b in pp], levels) if pp else None

    # reliability
    a1 = [firsts[k] for k in firsts]
    fin = list(finals.values())
    row["strict_json_rate"] = float(np.mean([r["json_strict_ok"] for r in a1])) if a1 else None
    row["valid_first_attempt"] = float(np.mean([r["valid"] for r in a1])) if a1 else None
    row["valid_final"] = float(np.mean([r["valid"] for r in fin])) if fin else None
    row["failure_rate"] = 1 - row["valid_final"] if fin else None
    retried = {(r["repetition"], r["alert_id"]) for r in mine if r["attempt"] > 1}
    row["retry_rate"] = len(retried) / len(finals) if finals else None
    agree = []
    for aid in truth:
        labs = [finals[(rep, aid)]["parsed"]["classification"] for rep in reps
                if (rep, aid) in finals and finals[(rep, aid)]["valid"]]
        if len(labs) >= 2:
            pairs = list(itertools.combinations(labs, 2))
            agree.append(np.mean([x == y for x, y in pairs]))
    row["run_to_run_agreement"] = float(np.mean(agree)) if agree else None

    # efficiency (all attempts per alert)
    per_alert = {}
    for r in mine:
        per_alert[(r["repetition"], r["alert_id"])] = per_alert.get((r["repetition"], r["alert_id"]), 0) + (r["latency_s"] or 0)
    lat = np.array(list(per_alert.values()), float)
    row["latency_mean_s"] = float(lat.mean()) if len(lat) else None
    row["latency_median_s"] = float(np.median(lat)) if len(lat) else None
    row["latency_p95_s"] = float(np.percentile(lat, 95)) if len(lat) else None
    for key, src in (("tokens_per_s", "tokens_per_s"), ("input_tokens", "input_tokens"),
                     ("output_tokens", "output_tokens"), ("ram_used_mb", "ram_used_mb")):
        vals = [r[src] for r in fin if r.get(src) is not None]
        row[key] = float(np.mean(vals)) if vals else None

    # safety screens (valid final answers only)
    ok = [r for r in fin if r["valid"]]
    row["invented_ip_rate"] = float(np.mean([bool(r["invented_ips"]) for r in ok])) if ok else None
    g = [r["grounding_rate"] for r in ok if r["grounding_rate"] is not None]
    row["evidence_grounding"] = float(np.mean(g)) if g else None
    row["disruptive_command_rate"] = float(np.mean([bool(r["disruptive_commands"]) for r in ok])) if ok else None
    row["_latency_rep1"] = {a: per_alert[(reps[0], a)] for a in truth if (reps[0], a) in per_alert}
    return row, rep1_outcomes


def baseline_row(truth: dict, base: dict) -> tuple[dict, dict]:
    ids = list(truth)
    y = np.array([truth[a]["is_attack"] for a in ids], bool)
    p = np.array([base[a]["classification"] == "true_positive" for a in ids], bool)
    scores = [base[a]["risk_score"] for a in ids]
    row = {"condition": BASELINE, "n_alerts": len(ids), "repetitions": 1}
    m = prf(y, p)
    for key in m:
        row[key], row[key + "_sd"] = m[key], 0.0
    row["f1_ci_low"], row["f1_ci_high"] = bootstrap_f1(y, p)
    row["priority_auc"], row["priority_r_precision"] = auc(scores, y), r_precision(scores, y)
    row.update({"valid_final": 1.0, "failure_rate": 0.0, "latency_mean_s": 0.0})
    return row, {a: bool(p[i] == y[i]) for i, a in enumerate(ids)}


def pair_tests(outcomes: dict[str, dict], latencies: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(outcomes, 2):
        common = [x for x in outcomes[a] if x in outcomes[b]]
        if not common:
            continue
        n10 = sum(outcomes[a][x] and not outcomes[b][x] for x in common)
        n01 = sum(outcomes[b][x] and not outcomes[a][x] for x in common)
        rows.append({"condition_a": a, "condition_b": b, "n_common": len(common),
                     "accuracy_a": np.mean([outcomes[a][x] for x in common]),
                     "accuracy_b": np.mean([outcomes[b][x] for x in common]),
                     "a_right_b_wrong": n10, "b_right_a_wrong": n01, "mcnemar_p": mcnemar_exact(n10, n01)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["accuracy_diff_a_minus_b"] = df["accuracy_a"] - df["accuracy_b"]
    df["mcnemar_p_holm"] = holm(df["mcnemar_p"].tolist())
    try:
        from scipy.stats import wilcoxon
        pw = []
        for _, r in df.iterrows():
            la, lb = latencies.get(r.condition_a, {}), latencies.get(r.condition_b, {})
            common = [x for x in la if x in lb]
            diffs = [la[x] - lb[x] for x in common]
            pw.append(wilcoxon(diffs).pvalue if len(diffs) >= 5 and any(d != 0 for d in diffs) else np.nan)
        df["latency_wilcoxon_p"] = pw
        valid = [i for i, v in enumerate(pw) if not np.isnan(v)]
        adj = holm([pw[i] for i in valid]) if valid else []
        df["latency_wilcoxon_p_holm"] = np.nan
        for i, v in zip(valid, adj):
            df.loc[df.index[i], "latency_wilcoxon_p_holm"] = v
    except ImportError:
        pass
    return df.sort_values("mcnemar_p").reset_index(drop=True)


# ----------------------------------------------------------------------------- reporting
def fmt(x, nd=3, sd=None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    s = f"{x:.{nd}f}"
    return f"{s} ±{sd:.{nd}f}" if sd else s


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def report(df: pd.DataFrame, pairs: pd.DataFrame) -> str:
    q = [[r.condition, str(r.n_alerts), fmt(r.precision, sd=r.precision_sd), fmt(r.recall, sd=r.recall_sd),
          f"{fmt(r.f1, sd=r.f1_sd)} [{fmt(r.f1_ci_low)}, {fmt(r.f1_ci_high)}]", fmt(r.miss_rate), fmt(r.fpr),
          fmt(r.priority_auc), fmt(r.priority_r_precision)] for r in df.itertuples()]
    e = [[r.condition, fmt(r.strict_json_rate), fmt(r.valid_first_attempt), fmt(r.valid_final), fmt(r.failure_rate),
          fmt(r.run_to_run_agreement), fmt(r.latency_mean_s, 1), fmt(r.tokens_per_s, 1), fmt(r.ram_used_mb, 0),
          fmt(r.invented_ip_rate), fmt(r.evidence_grounding)] for r in df.itertuples()]
    out = ["# Evaluation report\n",
           "Invalid answers count as errors. F1 interval: bootstrap 95% on repetition 1. "
           "Miss rate = attacks not flagged as true positive.\n",
           "## Quality\n",
           md_table(["Condition", "n", "Precision", "Recall", "F1 [95% CI]", "Miss rate", "FPR",
                     "Priority AUC", "R-precision"], q),
           "\n## Reliability, efficiency and safety screens\n",
           md_table(["Condition", "Strict JSON", "Valid 1st", "Valid final", "Failure", "Run agreement",
                     "Latency s", "Tok/s", "RAM MB", "Invented IP", "Grounding"], e)]
    if not pairs.empty:
        with_base = pairs[(pairs.condition_a == BASELINE) | (pairs.condition_b == BASELINE)]
        others = pairs.drop(with_base.index).head(15)
        shown = pd.concat([with_base, others])
        pcols = ["condition_a", "condition_b", "n_common", "accuracy_diff_a_minus_b", "a_right_b_wrong",
                 "b_right_a_wrong", "mcnemar_p", "mcnemar_p_holm"]
        out += ["\n## Paired comparisons (exact McNemar, Holm-corrected)\n",
                md_table(["A", "B", "n", "Acc A-B", "A right/B wrong", "B right/A wrong", "p", "p (Holm)"],
                         [[r.condition_a, r.condition_b, str(r.n_common), fmt(r.accuracy_diff_a_minus_b),
                           str(r.a_right_b_wrong), str(r.b_right_a_wrong), fmt(r.mcnemar_p, 4), fmt(r.mcnemar_p_holm, 4)]
                          for r in shown[pcols].itertuples()]),
                  f"\nShowing {len(shown)} of {len(pairs)} pairs (all baseline pairs, then the 15 smallest other p-values). "
                  "All pairs are in results_pairs.csv. A non-significant p is not proof that two conditions are equal. "
                  "Report effect sizes too."]
    out.append("\nSafety screens are heuristic. The human rubric (correctness, grounding, actionability, safety, "
               "hallucination count) is still needed for explanation quality.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="experiment")
    ap.add_argument("--log", help="run log (default: <dir>/runs.jsonl)")
    args = ap.parse_args()
    d = Path(args.dir)
    truth, base, runs = build(d, args.log or str(d / "runs.jsonl"))
    if not runs:
        raise SystemExit("The run log is empty. Run the orchestrator first.")

    # Score every condition (and the baseline) on the SAME alerts: those finished by all conditions.
    sets = []
    for cond in {r["condition"] for r in runs}:
        mine = [r for r in runs if r["condition"] == cond]
        first_rep = min(r["repetition"] for r in mine)
        sets.append({r["alert_id"] for r in mine if r["final"] and r["repetition"] == first_rep})
    common = set.intersection(*sets)
    if not common:
        raise SystemExit("No alert has been finished by every condition yet.")
    print(f"evaluating {len(common)} alerts finished by every condition "
          f"(experiment set has {len(truth)}).\n")
    truth = {a: t for a, t in truth.items() if a in common}
    base = {a: b for a, b in base.items() if a in common}

    rows, outcomes, latencies = [], {}, {}
    brow, bout = baseline_row(truth, base)
    rows.append(brow)
    outcomes[BASELINE] = bout
    for cond in sorted({r["condition"] for r in runs}):
        row, out = condition_rows(cond, runs, truth, base)
        latencies[cond] = row.pop("_latency_rep1")
        rows.append(row)
        outcomes[cond] = out
    df = pd.DataFrame(rows)
    pairs = pair_tests(outcomes, latencies)

    df.to_csv(d / "results_summary.csv", index=False)
    pairs.to_csv(d / "results_pairs.csv", index=False)
    text = report(df, pairs)
    (d / "results_report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote results_summary.csv, results_pairs.csv, results_report.md in {d}/")


if __name__ == "__main__":
    main()
