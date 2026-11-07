"""Module 6: orchestrator. The glue between the other modules.

Three commands:

  prepare  build the experiment set from one alert file
           - collapse repeats into cases
           - split into a DEVELOPMENT part (learning the known-false-positive list) and a
             TEST pool (chronological by default), then sample a balanced test set
           - compute the baseline (D3) for every test alert
           writes: exp_d2.jsonl, exp_truth.jsonl, baseline_d3.jsonl, dev_cases.csv, prepare_summary.json

  run      send every test alert to the LLM under each condition, validate (one retry),
           and log every attempt. Safe to stop (Ctrl-C) and re-run: finished work is skipped.

  status   show how much of the planned work is finished

Rules enforced here:
  - the LLM only receives D2 (never D3, never labels)
  - nothing the LLM writes is executed
  - every attempt is logged (Module 5)

Examples:
  python orchestrator.py prepare russellmitchell_alerts.txt --n 200
  python orchestrator.py run --models llama3.2:3b --prompts zero_shot,few_shot --limit 20
  python orchestrator.py status --models llama3.2:3b --prompts zero_shot,few_shot
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from alerts_loader import collapse_repeats, load_alerts, to_d2, to_truth
from baseline_engine import RulePrior, baseline_assess
from llm_engine import PROMPT_FORMATS, LLMConfig
from output_validator import assess_validated
from run_logger import RunLogger, environment_info, make_condition, model_details


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- prepare
def cmd_prepare(a) -> None:
    out = Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_alerts(a.csv)
    cases = collapse_repeats(df, a.window)
    if a.split == "chronological":
        cut = cases["first_seen"].quantile(a.dev_frac)
        dev, pool = cases[cases["first_seen"] <= cut], cases[cases["first_seen"] > cut]
    else:
        dev = cases.sample(frac=a.dev_frac, random_state=a.seed)
        pool = cases.drop(dev.index)

    atk, oth = pool[pool["is_attack"]], pool[~pool["is_attack"]]
    n_atk = min(a.n // 2, len(atk))
    n_oth = min(a.n - n_atk, len(oth))
    picked = pd.concat([atk.sample(n_atk, random_state=a.seed), oth.sample(n_oth, random_state=a.seed)])
    picked = picked.sort_values("first_seen").reset_index(drop=True)

    prior = RulePrior().fit(dev)
    d2 = [to_d2(r) for _, r in picked.iterrows()]
    write_jsonl(out / "exp_d2.jsonl", d2)
    write_jsonl(out / "exp_truth.jsonl", [to_truth(r) for _, r in picked.iterrows()])
    write_jsonl(out / "baseline_d3.jsonl", [baseline_assess(x, prior) for x in d2])
    dev[["name", "is_attack"]].to_csv(out / "dev_cases.csv", index=False)

    summary = {"source_file": a.csv, "window_s": a.window, "split": a.split, "dev_frac": a.dev_frac,
               "seed": a.seed, "raw_alerts": len(df), "cases": len(cases),
               "dev_cases": len(dev), "dev_attack_cases": int(dev["is_attack"].sum()),
               "pool_cases": len(pool), "pool_attack_cases": int(len(atk)),
               "experiment_cases": len(picked), "experiment_attack_cases": int(n_atk)}
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for k, v in summary.items():
        print(f"{k}: {v}")
    if n_atk < a.n // 2:
        print(f"\nWARNING: the test pool has only {len(atk)} attack cases, so the set is not balanced. "
              "Try --split random, a smaller --dev-frac, or a bigger scenario file.")
    if int(dev["is_attack"].sum()) == 0:
        print("WARNING: the development part has no attack cases, so the learned rule priors only know harmless rules.")
    print(f"\nfiles written to {out}/")


# ----------------------------------------------------------------------------- run
def _csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def cmd_run(a) -> None:
    if a.mode != "A":
        sys.exit("Only mode A (raw alert) exists so far. Modes B and C need the context retriever (Module 8).")
    d = Path(a.dir)
    alerts = read_jsonl(d / "exp_d2.jsonl")[: a.limit]
    models, prompts = _csv_list(a.models), _csv_list(a.prompts)
    temps = [float(t) for t in _csv_list(a.temperatures)]
    bad = [p for p in prompts if p not in PROMPT_FORMATS]
    if bad:
        sys.exit(f"Unknown prompt format(s) {bad}. Choose from {PROMPT_FORMATS}")
    log_path = a.log or str(d / "runs.jsonl")
    done = RunLogger.completed(log_path)

    planned = 0
    for t in temps:
        reps = a.reps if t > 0 else 1
        planned += len(models) * len(prompts) * reps * len(alerts)
    print(f"planned attempts-to-finish: {planned} (already done in log: {len(done)})")

    start, finished, durations = time.time(), 0, []
    try:
        with RunLogger(log_path, a.experiment_id) as logger:
            logger.write_manifest({"experiment_id": a.experiment_id, "command": " ".join(sys.argv),
                                   "environment": environment_info(a.host, models),
                                   "alerts": len(alerts), "models": models, "prompts": prompts,
                                   "temperatures": temps, "reps": a.reps, "mode": a.mode})
            for model in models:
                quant = model_details(a.host, model).get("quantization_level")
                for fmt in prompts:
                    for t in temps:
                        reps = a.reps if t > 0 else 1
                        if t == 0 and a.reps > 1:
                            print(f"note: temperature 0 makes repetitions identical, so {model}|{fmt} runs once.")
                        cond = make_condition(model, fmt, t, a.mode, a.top_p, a.format)
                        for rep in range(1, reps + 1):
                            cfg = LLMConfig(model=model, host=a.host, temperature=t, top_p=a.top_p,
                                            max_tokens=a.max_tokens, seed=a.seed + 100 * (rep - 1),
                                            num_ctx=a.num_ctx, output_format=a.format)
                            for alert in alerts:
                                if (cond, alert["alert_id"], rep) in done:
                                    continue
                                res = assess_validated(alert, fmt, cfg, repetition=rep)
                                if res["connection_error"]:
                                    print(f"\nSTOPPED: {res['connection_error']}\nNothing was logged for this alert. "
                                          "Fix the problem and run the same command again to resume.")
                                    return
                                last = len(res["attempts"]) - 1
                                for i, att in enumerate(res["attempts"]):
                                    logger.log_attempt(alert_id=alert["alert_id"], condition=cond, repetition=rep,
                                                       attempt=att, final=(i == last), mode=a.mode, quantisation=quant)
                                finished += 1
                                durations.append(res["total_latency_s"])
                                recent = durations[-10:]
                                eta_min = (planned - len(done) - finished) * (sum(recent) / len(recent)) / 60
                                state = "valid" if res["valid"] else "FAILED"
                                print(f"[{finished}] {cond} r{rep} {alert['alert_id']} {state} "
                                      f"{res['total_latency_s']}s | eta ~{max(eta_min, 0):.0f} min")
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved. Run the same command again to resume.")
        return
    print(f"\nfinished {finished} alerts this session in {(time.time() - start) / 60:.1f} min. "
          f"Next: python evaluate.py --dir {a.dir}")


# ----------------------------------------------------------------------------- status
def cmd_status(a) -> None:
    d = Path(a.dir)
    n = len(read_jsonl(d / "exp_d2.jsonl")[: a.limit])
    done = RunLogger.completed(a.log or str(d / "runs.jsonl"))
    counts: dict[tuple[str, int], int] = {}
    for cond, _alert, rep in done:
        counts[(cond, rep)] = counts.get((cond, rep), 0) + 1
    if not counts:
        print(f"nothing logged yet ({n} alerts in the experiment set)")
        return
    for (cond, rep), c in sorted(counts.items()):
        print(f"{cond} r{rep}: {c}/{n} alerts finished")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="build the experiment set")
    p.add_argument("csv")
    p.add_argument("--dir", default="experiment")
    p.add_argument("--window", type=int, default=300)
    p.add_argument("--n", type=int, default=200, help="total experiment cases (balanced when possible)")
    p.add_argument("--split", choices=["chronological", "random"], default="chronological")
    p.add_argument("--dev-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_prepare)

    for name, fn in (("run", cmd_run), ("status", cmd_status)):
        q = sub.add_parser(name)
        q.add_argument("--dir", default="experiment")
        q.add_argument("--log", help="run log path (default: <dir>/runs.jsonl)")
        q.add_argument("--models", default="llama3.2:3b", help="comma-separated Ollama model names")
        q.add_argument("--prompts", default="zero_shot", help=f"comma-separated, from {PROMPT_FORMATS}")
        q.add_argument("--temperatures", default="0")
        q.add_argument("--reps", type=int, default=1, help="repetitions (only useful when temperature > 0)")
        q.add_argument("--limit", type=int, help="only the first N alerts")
        q.add_argument("--mode", default="A", help="enrichment mode (only A for now)")
        q.add_argument("--host", default=LLMConfig.host)
        q.add_argument("--top-p", type=float, default=LLMConfig.top_p)
        q.add_argument("--max-tokens", type=int, default=LLMConfig.max_tokens)
        q.add_argument("--num-ctx", type=int, default=LLMConfig.num_ctx)
        q.add_argument("--seed", type=int, default=LLMConfig.seed)
        q.add_argument("--format", choices=["schema", "json", "none"], default="schema")
        q.add_argument("--experiment-id", default=time.strftime("exp%Y%m%d"))
        q.set_defaults(func=fn)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
